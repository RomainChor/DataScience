
import time
import numpy as np
import pandas as pd
from tqdm import tnrange
from copy import deepcopy
from sklearn.linear_model import SGDRegressor, SGDClassifier
import sklearn.metrics as metrics




def theta_risk(outputs, targets, theta=0):
    """
    Computes the 0-1 margin risk.
    Args:
        outputs (array-like): outputs of the decision function of the classifier.
        targets (array-like): targets to predict.
        classes (tuple-like): labels of the classes (binary classification).
        theta (float >= 0): margin value, default = 0.
    Returns: risk value (float)
    """
    if theta < 0:
        raise ValueError("theta must be >= 0!")
        
    labels = deepcopy(targets)
    classes = np.unique(labels)
    labels[labels == classes[0]] = 0
    labels[labels == classes[1]] = 1
    
    return (2*(labels-0.5)*outputs.transpose() < theta).mean()



def load_local_model(params):
    if params["task"] == "regression":
        mod = SGDRegressor
    else:
        mod = SGDClassifier
    model = mod(
        loss=params["loss"],
        penalty=None,
        max_iter=1,
        fit_intercept=True,
        shuffle=True,
        random_state=params["seed"],
        learning_rate='adaptive',
        tol=1e-2,
        n_iter_no_change=10,
        eta0=params["lr"]
    )

    return model



def load_global_model(params):
    if params["task"] == "regression":
        mod = SGDRegressor
    else:
        mod = SGDClassifier
    model = mod(
        loss=params["loss"],
        penalty=None,
        max_iter=1,
        fit_intercept=True,
        shuffle=False,
        random_state=params["seed"],
        learning_rate='constant',
        eta0=1e-32 
    )
    
    return model



def get_loss_pred_fn(name):
    if name == "squared_error":
        return metrics.mean_squared_error, "predict"
    elif name == "hinge":
        return metrics.hinge_loss, "decision_function"
    elif name == "log":
        return metrics.log_loss, "predict_proba"
    else:
        raise ValueError()


        
class Client:
    """
    Class emulating each client of the network. 
    """
    def __init__(self, X, y, params, generator=None):
        # self.X = np.array_split(X, params["n_rounds"])
        # self.y = np.array_split(y, params["n_rounds"])
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.X_place, self.y_place = self.compute_placeholders(X, y)
        self.model = load_local_model(params)
        self.loss_fn, pred = get_loss_pred_fn(params["loss"])
        self.pred_fn = getattr(self.model, pred)
        self.epochs = params["client_epochs"]
        self.round = 0
        self.generator = generator if generator is not None else np.random.default_rng(0)
        
    def compute_placeholders(self, X, y):
        # classes = np.unique(y)
        idx_1 = np.where(y == self.classes[0])[0][0]
        idx_2 = np.where(y == self.classes[1])[0][0]

        return np.array([X[idx_1], X[idx_2]], ndmin=2), np.array([y[idx_1], y[idx_2]])
        
    def initial_fit(self, coef=None, intercept=None):
        self.model.fit(
            self.X, 
            self.y,
            coef_init=coef, 
            intercept_init=intercept
        )

    def partial_fit(self):
        self.model.partial_fit(self.X, self.y)
        
    def run_SGD_iterations(self):
        for e in range(self.epochs-1):
            self.partial_fit()

    def compute_emp_risk(self):
        outputs = self.pred_fn(self.X)
        return self.loss_fn(self.y, outputs)



class Server:
    """
    Class emulating the central server. Instructs the clients to train their models, 
    then aggregates their parameters and computes risks values.
    """
    def __init__(self, X_test, y_test, params):
        self.X_test = X_test
        self.y_test = y_test
        self.X_placeholder, self.y_placeholder = self.compute_placeholders(X_test, y_test)

        self.global_coef_ = None
        # self.global_coef_ = np.random.normal(scale=0.1, size=X_test.shape[1])
        self.global_intercept_ = None
        self.n_clients = 0

        self.distribution = []   
        self.test_risks, self.emp_risks = [], []
        self.barnes_risks = []
        self.avg_local_emp_risk = 0
        self.global_model = load_global_model(params) 
        self.loss_fn, pred = get_loss_pred_fn(params["loss"])
        self.pred_fn = getattr(self.global_model, pred)
        self.round = 0

    def compute_placeholders(self, X, y):
        classes = np.unique(y)
        idx_1 = np.where(y == classes[0])[0][0]
        idx_2 = np.where(y == classes[1])[0][0]

        return np.array([X[idx_1], X[idx_2]], ndmin=2), np.array([y[idx_1], y[idx_2]])

    def add_participant(self, participant):
        self.distribution = np.append(self.distribution, participant)
        self.n_clients += 1

    def _aggregate_models(self):
        temp_coef = 0
        temp_inter = 0
        for client in self.distribution:
            temp_coef += client.model.coef_
            temp_inter += client.model.intercept_
        temp_coef /= self.n_clients
        temp_inter /= self.n_clients

        self.global_coef_ = temp_coef
        self.global_intercept_ = temp_inter

        # Avg of emp risks on local models before aggregation
        self.avg_local_emp_risk = np.mean([client.compute_emp_risk() for client in self.distribution])

        self.global_model.fit(
            X=self.X_placeholder,
            y=self.y_placeholder,
            coef_init=self.global_coef_,
            intercept_init=self.global_intercept_
        ) # Sets global model's parameters to the aggregated ones
    
    def compute_emp_risk(self):
        emp_risk = 0
        for client in self.distribution:
            outputs = self.pred_fn(client.X)
            emp_risk += self.loss_fn(client.y, outputs)
        emp_risk /= self.n_clients
        
        return emp_risk
    
    def compute_barnes_risk(self):
        barnes_risk = 0
        for client in self.distribution:
            outputs = self.pred_fn(client.X[client.round])
            barnes_risk += self.loss_fn(client.y[client.round], outputs) 
        barnes_risk /= self.n_clients
        
        return barnes_risk
    
    def compute_test_risk(self):
        outputs = self.pred_fn(self.X_test)
        
        return self.loss_fn(self.y_test, outputs)
    
    def compute_01_risks(self):
        risk = 1 - self.global_model.score(self.X_test, self.y_test)
        
        emp_risk = 0
        for client in self.distribution:
            # l = 0
            # for r in range(client.round+1):
            #     l += 1 - self.global_model.score(client.X[r], client.y[r])
            # l /= client.round+1
            # emp_risk += l
            emp_risk += 1 - self.global_model.score(client.X, client.y)
        emp_risk /= self.n_clients
        
        return emp_risk, risk
        
    def _update_risks(self):
        # self.test_risks.append(self.compute_test_risk())
        # self.barnes_risks.append(self.compute_barnes_risk())
        # self.emp_risks.append(self.compute_emp_risk())
        pass

    def run_round(self):
        for client in self.distribution:
            client.round = self.round
            client.initial_fit(coef=self.global_coef_, intercept=self.global_intercept_)
            client.run_SGD_iterations()

        self._aggregate_models()
        self._update_risks()
        self.round += 1

            
        
def federated_learning(data, params, K=2, generator=None):
    """
    Runs the distributed learning setup. 
    Splits data into client datasets, creates clients and server instances and launch training. 
    Args:
        data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
        params (dict): parameters for training.
        K (int >= 2): number of clients, default = 2.
    """
    if K < 2:
        raise ValueError("K must be >= 2!")
    if params["rounds"] < 1:
        raise ValueError("params['rounds'] must be >= 1!")

    X_list = np.array_split(data["X"], K)
    y_list = np.array_split(data["y"], K)
        
    server = Server(data["X_test"], data["y_test"], params)
    for k in range(K):
        server.add_participant(Client(X_list[k], y_list[k], params))

    for _ in range(params["rounds"]):
        server.run_round()
        
    return server



def centralized_learning(data, params):
    """
    Runs the centralized learning setup. 
    Args:
        data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
        params (dict): parameters for training.
    """
    model = load_local_model(params)
    model.fit(data["X"], data["y"])
    
    loss_fn, pred = get_loss_pred_fn(params["loss"])
    pred_fn = getattr(model, pred) 
    outputs = pred_fn(data["X"])
    emp = loss_fn(data["y"], outputs)
    if params["task"] == "classification":
        emp_01 = 1 - model.score(data["X"], data["y"])
        print("Epoch 1 done. Emp risk: {0:.4f}".format(emp_01))
    else:
        print("Epoch 1 done. Emp risk: {0:.4f}".format(emp))
    # =============================================================
    stream = tnrange(2, params["epochs"]+1)
    for e in stream:
        idxs = np.random.permutation(len(data["y"])) # Shuffle at each epoch 
        model.partial_fit(data["X"][idxs], data["y"][idxs])
        outputs = pred_fn(data["X"][idxs])
        emp = loss_fn(data["y"][idxs], outputs)
        if params["task"] == "classification":
            emp_01 = 1 - model.score(data["X"][idxs], data["y"][idxs])
            stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp_01))
        else:
            stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp))
#         if emp_risk <= params["min_risk"]:
#             break
        
    outputs = pred_fn(data["X_test"])
    risk = loss_fn(data["y_test"], outputs)
    if params["task"] == "classification":
        risk_01 = 1 - model.score(data["X_test"], data["y_test"])
        return emp, risk, emp_01, risk_01

    return emp, risk



def compute_bound(n, K, B=1.0, theta=1.0):
    """
    This function computes the exact in-expectation bound for the distributed learning setup (Theorem 5ii).
    """
    if theta == 0:
        theta = 0.2
        
    m = 112*(B/(K*theta))**2*np.log(n*K*np.sqrt(K))
    c = np.sqrt(K**2*theta**2/(20*B**2) + 1)
    eps = 8*np.exp(-(m/7)*(K*theta/(4*B))**2) + m/((2*c)**m*np.sqrt(np.pi))*np.exp(-0.5*(m+1)*(K*theta/(2*c**2*B))**2)
    eps += 8*np.exp(-0.21*m*(c**2 - 1)) 
    
    return np.sqrt(2*m*np.log(2*c**2 + 1)/n) + eps 



# class Client:
#     """
#     Class simulating each client. 
#     """
#     def __init__(self, X, y, params, seed=1):
#         self.X = X
#         self.y = y
#         self.clf = SGDClassifier(
#             alpha=params["alpha"],
#             max_iter=1,
#             random_state=seed,
#             fit_intercept=False,
#             shuffle=True,
#             learning_rate='adaptive',
#             tol=1e-2,
#             n_iter_no_change=10,
#             eta0=params["lr"]
#         )
#         self.classes = np.unique(y)
#         self.epochs = params["client_epochs"]
#         self.theta = params["theta"]
#         self.min_risk = params["min_risk"]
        
#     def partial_fit(self):
#         self.clf.partial_fit(self.X, self.y)

#     def initial_fit(self, coef=None):
#         self.clf.fit(self.X, self.y, coef_init=coef)

#     def run_SGD_iterations(self):
#         for e in range(self.epochs-1):
#             self.partial_fit()
            
#             outputs = self.clf.decision_function(self.X)
#             if theta_risk(outputs, self.y, self.classes, self.theta) <= self.min_risk:
#                 break

                
# class Server:
#     """
#     Class simulating the central server. Instructs the clients to train their models, then aggregate their parameters 
#     and compute risks values.
#     """
#     def __init__(self, X_test, y_test, params):
#         self.X_test = X_test
#         self.y_test = y_test
#         self.X_placeholder = self.X_test[:10]
#         self.y_placeholder = self.y_test[:10]

#         self.global_coef_ = None
#         self.distribution = np.array([])
#         self.local_emp_risks = None
#         self.n_clients = 0

#         self.test_risks, self.emp_risks = [], []
#         self.global_clf = SGDClassifier(
#             alpha=0,
#             max_iter=1,
#             fit_intercept=False,
#             random_state=1,
#             shuffle=True,
#             learning_rate='constant',
#             eta0=1e-12
#         )
#         self.theta = params["theta"]

#     def add_participant(self, participant):
#         self.distribution = np.append(self.distribution, participant)
#         self.n_clients += 1

#     def _aggregate_models(self):
#         temp_coef = self.distribution[0].clf.coef_
#         for svm in self.distribution:
#             temp_coef += svm.clf.coef_
            
#         temp_coef /= len(self.distribution)

#         if self.global_coef_ is None:
#             self.global_coef_ = temp_coef
#         else:
#             self.global_coef_ = (self.global_coef_ + temp_coef)/2 

#     def _update_risks(self):
#         self.global_clf.fit( 
#             X=self.X_placeholder,
#             y=self.y_placeholder,
#             coef_init=self.global_coef_
#         )
#         self.test_risks.append(1 - self.global_clf.score(self.X_test, self.y_test))

#         emp_risk = 0
#         for svm in self.distribution:
#             outputs = self.global_clf.decision_function(svm.X)
#             emp_risk += theta_risk(outputs, svm.y, svm.classes, self.theta)
#         emp_risk /= self.n_clients
#         self.emp_risks.append(emp_risk)

#     def run_round(self):
#         local_emp_risks = np.array([])

#         for svm in tqdm_notebook(self.distribution):
#             svm.initial_fit(coef=self.global_coef_)
#             svm.run_SGD_iterations()
#             outputs = svm.clf.decision_function(svm.X)
#             local_emp_risks = np.append(local_emp_risks, theta_risk(outputs, svm.y, svm.classes, self.theta))

#         if self.local_emp_risks is None:
#             self.local_emp_risks = np.array([local_emp_risks])
#         else:
#             self.local_emp_risks = np.append(
#                 self.local_emp_risks, np.array([local_emp_risks]), axis=0)

#         self._aggregate_models()
#         self._update_risks()
        
        
# def distributed_learning(data, idxs, params, K=2, n_rounds=1, seed=1):
#     """
#     Runs the distributed learning setup. 
#     Splits data into client datasets, creates clients and server instances and launch training. 
#     Args:
#         data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
#         idxs (list-like): indexes of samples drawn randomly from data.
#         params (dict): parameters for training.
#         K (int >= 2): number of clients, default = 2.
#         n_rounds (int >= 1): number of distributed rounds, default = 1.
#         seed (int >= 0): seed for random experiments, default = 1.
#     """
#     if K < 2:
#         raise ValueError("K must be >= 2!")
#     if n_rounds < 1:
#         raise ValueError("n_rounds must be >= 1!")
        
#     X_list = np.split(data["X"][idxs], K)
#     y_list = np.split(data["y"][idxs], K)
    
#     server = Server(data["X_test"], data["y_test"], params)
#     for k in range(K):
#         server.add_participant(Client(X_list[k], y_list[k], params, seed))

#     for r in range(n_rounds):
#         server.run_round()

#     return server


# def centralized_learning(data, idxs, params, seed=1):
#     """
#     Runs the centralized learning setup. 
#     Args:
#         data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
#         idxs (list-like): indexes of samples drawn randomly from data.
#         params (dict): parameters for training.
#         seed (int >= 0): seed for random experiments, default = 1.
#     """
#     classes = np.unique(data["y"])
#     clf = SGDClassifier(
#         alpha=params["alpha"],
#         max_iter=1,
#         random_state=seed,
#         fit_intercept=False,
#         shuffle=True,
#         learning_rate='adaptive',
#         tol=1e-2,
#         n_iter_no_change=10,
#         eta0=params["lr"]
#     )
#     clf.fit(data["X"][idxs], data["y"][idxs])
    
#     stream = tnrange(2, params["epochs"]+1)
#     for e in stream:
#         clf.partial_fit(data["X"][idxs], data["y"][idxs])
#         outputs = clf.decision_function(data["X"][idxs])
#         emp_risk = theta_risk(outputs, data["y"][idxs], classes, params["theta"])
        
#         stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp_risk))
        
#         if emp_risk <= params["min_risk"]:
#             break
        
#     test_risk = 1 - clf.score(data["X_test"], data["y_test"])
    
#     return emp_risk, test_risk