
import numpy as np
import idx2numpy
from copy import deepcopy
from sklearn.datasets import (
    make_regression, make_friedman1, make_friedman2, fetch_california_housing
)
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split





def generate_ball_data(num_points, dim, center=0, radius=1):
    # Generate a point on the sphere
    random_directions = np.random.normal(size=(num_points, dim))
    random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1, 1)
    # Pick a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1/dim)
    # Scale to get points on the ball.
    X = radius * (random_radii.reshape(-1, 1) * random_directions) + center
    # y = (X[:, -1] < 0).astype(int)
    coeff = np.ones(dim)
    coeff[0] = -.2
    if center == 0:
        center = np.zeros(dim)
    y = (np.dot(X, coeff) + center[0]/5 > 0).astype(int)
    # print(y)

    return X, y



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



def make_classification_data(params, loaded_data=None):
    data = {}                                           
    if params.name == "mnist":
        if loaded_data is not None:
            data = deepcopy(loaded_data)
            
        if params.N <= data["X"].shape[0]: # Sampling
            data["X"], _, data["y"], _ = train_test_split(
                data["X"], 
                data["y"], 
                train_size=params.N, 
                random_state=params.seed,
                stratify=data["y"]
            )
        else:
            print("N is greater than the dataset size. The whole dataset is loaded.")

        if params.compare_hom_het:
            d = data["X"].shape[1]
            if params.iid:
                data["X"][:params.N//4, :] += np.random.normal(scale=params.noise_std, size=(params.N//4, d))
                data["X"][3*params.N//4:, :] += np.random.normal(scale=params.noise_std, size=(params.N//4, d))
            else:
                data["X"][:params.N//2, :] += np.random.normal(scale=params.noise_std, size=(params.N//2, d))
            N_test = data["X_test"].shape[0]
            idxs = np.random.permutation(N_test)[:N_test//2]
            data["X_test"][idxs, :] += np.random.normal(scale=params.noise_std, size=(N_test//2, d))

    elif params.name == "balls":    
        s = 1 
        if params.iid:
            centers = np.zeros((params.N, params.dim))
            centers[:, 0] = s*(2*np.random.binomial(n=1, p=0.5, size=params.N) - 1)
            data["X"], data["y"] = generate_ball_data(params.N, params.dim, radius=params.radius)
            data["X"] += centers
        else:
            centers = np.zeros((params.N, params.dim))
            centers[:params.N//2, 0] = -s
            centers[params.N//2:, 0] = s
            data["X"], data["y"] = generate_ball_data(params.N, params.dim, radius=params.radius)
            data["X"] += centers

        N_test = 500
        centers = np.zeros((N_test, params["dim"]))
        centers[:, 0] = s*(2*np.random.binomial(n=1, p=0.5, size=N_test) - 1)
        data["X_test"], data["y_test"] = generate_ball_data(N_test, params.dim, radius=params.radius)
        data["X_test"] += centers

    return data



def make_regression_data(params):
    N_samples = params.N+params["N_test"]
    if params.name == "default":
        X, y = make_regression(
            n_samples=N_samples, 
            n_features=params.dim,
            n_informative=params.dim,
            bias=params.bias,
            noise=params.noise_std,
            shuffle=True,
            random_state=params.seed
        )
    elif params.name == "friedman":
        X, y = make_friedman1(
            n_samples=N_samples,
            n_features=params.dim,
            noise=params.noise_std,
            random_state=params.seed
        )
    elif params.name == "gaussian":
        X = np.random.normal(params.noise_mean, params.noise_std, (N_samples, params.dim))
        W = np.random.normal(params.noise_mean, params.noise_std, params.dim)
        y = np.dot(X, W) 
        if params.noise_std > 0:
            y += params.noise_std*np.random.normal(0, 1.0, N_samples)
    elif params.name == "california":
        dataset = fetch_california_housing()
        if N_samples <= len(dataset.target):
            X, _, y, _ = train_test_split(
                dataset.data, 
                dataset.target, 
                train_size=N_samples, 
                random_state=params.seed
            )
#             idxs = np.random.randint(0, len(dataset.target), N_samples)
#             X, y = dataset.data[idxs, :], dataset.target[idxs]
        else:
            X, y = dataset.data, dataset.target
            print("N+N_test is greater than the dataset size. The whole dataset is loaded.")
    else:
        raise ValueError("params['name'] is not recognized!")
    
    X, X_test, y, y_test = train_test_split(
        X, 
        y, 
        test_size=params.N//10, 
        random_state=params.seed
    )
    
#     y_test += np.random.normal(0, 1.0, params["N_test"])
    
    if params.rescale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)
    
    return {"X":X, "y":y, "X_test":X_test, "y_test":y_test}



            
        