"""
Machine learning helper module
Author: Romain Chor

Packages:
	numpy as np
	pandas as pd
	matplotlib.pyplot as plt
	seaborn as sns

Functions:
	missing_val
    nan_filling
	rmse
    filling_season
    train_test_base
    train_test_random
    train_test_cv
    blending_cv (UNDER DEVELOPMENT)
    plot_confusion_matrix
"""

print(__doc__)

#------------------------------------------------------------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import itertools
import time
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, log_loss, average_precision_score
#------------------------------------------------------------------------


def missing_val(df):
    """
    Indicates which columns of df contain missing values,
    with count and percentage.
    df: a Pandas dataframe
     """
    miss = df.isnull() #null or na?
    total = miss.sum()
    perc = total*100/miss.count()
    tab = pd.concat([total, perc], axis=1, keys=['Total', 'Percentage'])

    print(tab[total>0])


def nan_filling(x):
    """
    Fills missing values in column x naively.
    x: a Pandas series (column of a Pandas dataframe)
    """
    if x.dtype == int:
        return x.fillna(x.mode().values[0])
    elif x.dtype == float:
        return x.fillna(x.median())
    elif x.dtype == object:
        return x.fillna('__NC__')


def rmse(targets, predictions):
    """Calculates RMSE between predicitions and ground truth targets"""
    return np.sqrt(np.mean((predictions - targets) ** 2))


def filling_season(data):
    """
    Adds a 'season' feature based on a 'month' feature to data.
    data: a Pandas dataframe
    """
    data.loc[data.month.isin([1, 2, 3]), 'season'] = 'winter'
    data.loc[data.month.isin([4, 5, 6]), 'season'] = 'spring'
    data.loc[data.month.isin([7, 8, 9]), 'season'] = 'summer'
    data.loc[data.month.isin([10, 11, 12]), 'season'] = 'autumn'
    print ('Filling season : done \n')


def train_test_base(X_tr, X_va, y_tr, y_va, models, metric, chrono=True):
    """
    Train and test given models, train and validation/test sets
    -X_tr, X_va, y_tr, y_va : training (resp. validation/test) dataset and labels
    -models: dict of models to train and test
    with keys = names to display and values = sklearn object
    -metric: sklearn score function with the shape score(y_true, y_pred)
    -chrono: whether to time fit duration or not, bool
    """
    df = pd.DataFrame(columns=list(models.keys()))

    scores = []
    times = []
    for name, model in models.items():
        if chrono:
            start = time.time()
            model.fit(X_tr, y_tr)
            end = time.time()
            times.append(end-start)
            # print("time("+name+"): {0:2.3f}".format(end-start))
        else:
            model.fit(X_tr, y_tr)

        if metric == roc_auc_score: scores.append(metric(y_va, model.predict_proba(X_va)[:, 1]))
        else: scores.append(metric(y_va, model.predict(X_va)))

    df.loc['Score'] = scores
    if chrono: df.loc['Training time'] = times

    return df


def train_test_random(X, y, models, metric, seed=None, test_size=0.2, chrono=True):
    """
    Train and test given models: splits datasets then for each model,
    times fit duration and calculates a score with a random validation set.
    -X, y: training dataset and labels
    -models: dict of models to train and test
    with keys = names to display and values = sklearn object
    -metric: score function with the shape score(y_true, y_pred)
    -chrono: whether to time fit duration or not, bool
    """
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=seed)
    df = pd.DataFrame(columns=list(models.keys()))

    scores = []
    times = []
    for name, model in models.items():
        if chrono:
            start = time.time()
            model.fit(X_tr, y_tr)
            end = time.time()
            times.append(end-start)
            # print("time("+name+"): {0:2.3f}".format(end-start))
        else:
            model.fit(X_tr, y_tr)

        if metric in [roc_auc_score, log_loss, average_precision_score]: scores.append(metric(y_va, model.predict_proba(X_va)[:, 1]))
        else: scores.append(metric(y_va, model.predict(X_va)))

    df.loc['Score'] = scores
    if chrono: df.loc['Training time'] = times

    return df


def train_test_cv(X, y, models, metric, cv=3, chrono=True):
    """
    Train and test given models using cross-validation: times fit duration and
    computes mean cross validation score for each model.
    -X, y: training dataset and labels
    -models: dict of models to train and test
    with keys = names to display and values = sklearn object
    -metric: str respecting sklearn metrics names (Cf. doc) (recommended)
	or sklearn score function with the shape score(y_true, y_pred)
    -cv: Cf. sklearn cross_val_score
    -chrono: whether to time fit duration or not, bool
    """
    warnings.warn("Be careful if you give a score function for the 'metric' parameter as cross_val_score from sklearn gives sometimes weird results...")
    warnings.warn("To avoid give a str name. Ex: metric='roc_auc'")
    df = pd.DataFrame(columns=list(models.keys()))

    means, stds = [], []
    times = []

    if type(metric) is str: scorer = metric
    else: scorer = make_scorer(metric)

    for name, model in models.items():
        if chrono:
            start = time.time()
            cv_score = cross_val_score(model, X, y,
                scoring=scorer, cv=cv, n_jobs=-1)
            end = time.time()
            times.append(end-start)
        else:
            cv_score = cross_val_score(model, X, y,
            	scoring=scorer, cv=cv, n_jobs=-1)

        means.append(np.mean(cv_score))
        stds.append(np.std(cv_score))
        df[name] = cv_score

    df.loc['Mean score'] = means
    df.loc['Std'] = stds
    if chrono: df.loc['Training time'] = times

    return df

###### UNDER DEVELOPMENT ######
# def blending_cv(models, X, y, cv=5):
#     """
#     Performs cross-validated models blending and returns scores on each fold.
#     models: dict with model name as key and Sklearn model instance as value
#     -X: pandas Dataframe
#     -y: numpy array
#     -cv: number of folds
#     """
#     kf = KFold(n_splits=cv)
#     scores = []
#     i = 0
#     for train_index, val_index in kf.split(X):
#         X_tr, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
#         y_tr, y_val = y[train_index], y[val_index]
#         y_pred = 0
#         for __, model in models.items():
#             model.fit(X_tr, y_tr)
#             y_pred += model.predict_proba(X_val)[:, 1]
#         y_pred /= cv
#         scores.append(roc_auc_score(y_val, y_pred))
#         i += 1
#     scores.append(np.mean(scores))
#     scores = pd.DataFrame(scores, columns=['Score']).rename(index={5:'Mean score'})
#
#     return scores


def plot_confusion_matrix(y_pred, y, classes=None, normalize=False):
    """
    Plots the confusion matrix (only for classification!)
    Normalization can be applied by setting `normalize=True`.
    y_pred, y: Numpy arrays of Pandas series, resp. predictions and ground truth labels
    """
    title='Confusion matrix'
    cmap=plt.cm.Blues

    cm = confusion_matrix(y, y_pred)

    if classes is None:
        classes = np.unique(y)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Unnormalized confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid()
