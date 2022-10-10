import numpy as np
import random
import idx2numpy
from tqdm import tqdm_notebook, tnrange
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler



def theta_risk(outputs, targets, classes, theta=0):
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
        
    labels = targets.copy()
    labels[labels == classes[0]] = 0
    labels[labels == classes[1]] = 1
    
    return (2*(labels-0.5)*outputs.transpose() < theta).mean()


def compute_bound(n, K, B=1.0, theta=1.0):
    """
    This function computes the exact expectation bound for the distributed learning setup (Theorem 5ii).
    """
    if theta == 0:
        theta = 0.2
        
    m = 112*(B/(K*theta))**2*np.log(n*K*np.sqrt(K))
    c = np.sqrt(K**2*theta**2/(20*B**2) + 1)
    eps = 8*np.exp(-(m/7)*(K*theta/(4*B))**2) + m/((2*c)**m*np.sqrt(np.pi))*np.exp(-0.5*(m+1)*(K*theta/(2*c**2*B))**2)
    eps += 8*np.exp(-0.21*m*(c**2 - 1)) 
    
    return np.sqrt(2*m*np.log(2*c**2 + 1)/n) + eps 


def load_binary_data(class1, class2, path="", proj_dim=0, gamma=1.0):
    """
    Loads MNIST data (train/test sets), extracts two chosen classes, and project the data
    using a Gaussian kernel feature map.
    Args:
        class1, class2 (int): labels of the two chosen classes.
        path (str): path to the folder containing the MNIST files, default = "" (current directory).
        proj_dim (int >= 0): dimension of the kernel feature space, default = 0 and no projection is performed.
        gamma (float): parameter of the Gaussian kernel, default = 1.0.
    """
    
    X = idx2numpy.convert_from_file(path+'train-images.idx3-ubyte')
    y = idx2numpy.convert_from_file(path+'train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file(path+'t10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file(path+'t10k-labels.idx1-ubyte')

    y = np.squeeze(y)
    y_test = np.squeeze(y_test)
    
    idxs = ((y == class1) + (y == class2)).nonzero()
    idxs_test = ((y_test == class1) + (y_test == class2)).nonzero()
    X = X[idxs]
    y = y[idxs]
    X_test = X_test[idxs_test]
    y_test = y_test[idxs_test]

    X = (X/255. - 0.1307)/0.3081 
    X_test = (X_test/255. - 0.1307)/0.3081
    
    X = np.array([elm.ravel() for elm in X])
    X_test = np.array([elm.ravel() for elm in X_test])

    if proj_dim > 0:
        kernel = RBFSampler(gamma=gamma, n_components=proj_dim)
        X_proj = kernel.fit_transform(X)
        X_test_proj = kernel.transform(X_test)
        
        return {"X":X_proj, "y":y, "X_test":X_test_proj, "y_test":y_test}
    
    return {"X":X, "y":y, "X_test":X_test, "y_test":y_test}


class Client:
    """
    Class simulating each client. 
    """
    def __init__(self, X, y, params, seed=1):
        self.X = X
        self.y = y
        self.clf = SGDClassifier(
            alpha=params["alpha"],
            max_iter=1,
            random_state=seed,
            fit_intercept=False,
            shuffle=True,
            learning_rate='adaptive',
            tol=1e-2,
            n_iter_no_change=10,
            eta0=params["lr"]
        )
        self.classes = np.unique(y)
        self.epochs = params["client_epochs"]
        self.theta = params["theta"]
        self.min_risk = params["min_risk"]
        
    def partial_fit(self):
        self.clf.partial_fit(self.X, self.y)

    def initial_fit(self, coef=None):
        self.clf.fit(self.X, self.y, coef_init=coef)

    def run_SGD_iterations(self):
        for e in range(self.epochs-1):
            self.partial_fit()
            
            outputs = self.clf.decision_function(self.X)
            if theta_risk(outputs, self.y, self.classes, self.theta) <= self.min_risk:
                break

                
class Server:
    """
    Class simulating the central server. Instructs the clients to train their models, then aggregate their parameters 
    and compute risks values.
    """
    def __init__(self, X_test, y_test, params):
        self.X_test = X_test
        self.y_test = y_test
        self.X_placeholder = self.X_test[:10]
        self.y_placeholder = self.y_test[:10]

        self.global_coef_ = None
        self.distribution = np.array([])
        self.local_emp_risks = None
        self.n_clients = 0

        self.test_risks, self.emp_risks = [], []
        self.global_clf = SGDClassifier(
            alpha=0,
            max_iter=1,
            fit_intercept=False,
            random_state=1,
            shuffle=True,
            learning_rate='constant',
            eta0=1e-12
        )
        self.theta = params["theta"]

    def add_participant(self, participant):
        self.distribution = np.append(self.distribution, participant)
        self.n_clients += 1

    def _aggregate_models(self):
        temp_coef = self.distribution[0].clf.coef_
        for svm in self.distribution:
            temp_coef += svm.clf.coef_
            
        temp_coef /= len(self.distribution)

        if self.global_coef_ is None:
            self.global_coef_ = temp_coef
        else:
            self.global_coef_ = (self.global_coef_ + temp_coef)/2 

    def _update_risks(self):
        self.global_clf.fit( 
            X=self.X_placeholder,
            y=self.y_placeholder,
            coef_init=self.global_coef_
        )
        self.test_risks.append(1 - self.global_clf.score(self.X_test, self.y_test))

        emp_risk = 0
        for svm in self.distribution:
            outputs = self.global_clf.decision_function(svm.X)
            emp_risk += theta_risk(outputs, svm.y, svm.classes, self.theta)
        emp_risk /= self.n_clients
        self.emp_risks.append(emp_risk)

    def run_round(self):
        local_emp_risks = np.array([])

        for svm in tqdm_notebook(self.distribution):
            svm.initial_fit(coef=self.global_coef_)
            svm.run_SGD_iterations()
            outputs = svm.clf.decision_function(svm.X)
            local_emp_risks = np.append(local_emp_risks, theta_risk(outputs, svm.y, svm.classes, self.theta))

        if self.local_emp_risks is None:
            self.local_emp_risks = np.array([local_emp_risks])
        else:
            self.local_emp_risks = np.append(
                self.local_emp_risks, np.array([local_emp_risks]), axis=0)

        self._aggregate_models()
        self._update_risks()
        
        
def distributed_learning(data, idxs, params, K=2, n_rounds=1, seed=1):
    """
    Runs the distributed learning setup. 
    Splits data into client datasets, creates clients and server instances and launch training. 
    Args:
        data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
        idxs (list-like): indexes of samples drawn randomly from data.
        params (dict): parameters for training.
        K (int >= 2): number of clients, default = 2.
        n_rounds (int >= 1): number of distributed rounds, default = 1.
        seed (int >= 0): seed for random experiments, default = 1.
    """
    if K < 2:
        raise ValueError("K must be >= 2!")
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1!")
        
    X_list = np.split(data["X"][idxs], K)
    y_list = np.split(data["y"][idxs], K)
    
    server = Server(data["X_test"], data["y_test"], params)
    for k in range(K):
        server.add_participant(Client(X_list[k], y_list[k], params, seed))

    for r in range(n_rounds):
        server.run_round()

    return server


def centralized_learning(data, idxs, params, seed=1):
    """
    Runs the centralized learning setup. 
    Args:
        data (dict): dict containing train and test datasets, with keys "X", "y", "X_test", "y_test". 
        idxs (list-like): indexes of samples drawn randomly from data.
        params (dict): parameters for training.
        seed (int >= 0): seed for random experiments, default = 1.
    """
    classes = np.unique(data["y"])
    clf = SGDClassifier(
        alpha=params["alpha"],
        max_iter=1,
        random_state=seed,
        fit_intercept=False,
        shuffle=True,
        learning_rate='adaptive',
        tol=1e-2,
        n_iter_no_change=10,
        eta0=params["lr"]
    )
    clf.fit(data["X"][idxs], data["y"][idxs])
    
    stream = tnrange(2, params["epochs"]+1)
    for e in stream:
        clf.partial_fit(data["X"][idxs], data["y"][idxs])
        outputs = clf.decision_function(data["X"][idxs])
        emp_risk = theta_risk(outputs, data["y"][idxs], classes, params["theta"])
        
        stream.set_description("Epoch {0} done. Emp risk: {1:.4f}".format(e, emp_risk))
        
        if emp_risk <= params["min_risk"]:
            break
        
    test_risk = 1 - clf.score(data["X_test"], data["y_test"])
    
    return emp_risk, test_risk
            
        