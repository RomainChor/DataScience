"""
Machine learning module 
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
    plot_confusion_matrix
"""

print(__doc__)


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import itertools
import time
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score  


def missing_val(df):
    miss = df.isnull() #null or na?
    total = miss.sum()
    perc = total*100/miss.count()
    tab = pd.concat([total, perc], axis=1, keys=['Total', 'Percentage'])

    print(tab[total>0])


def nan_filling(x):
    if x.dtype == int:
        return x.fillna(x.mode().values[0])
    elif x.dtype == float:
        return x.fillna(x.median())
    elif x.dtype == object:
        return x.fillna('__NC__')

def rmse(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())


def filling_season(data): #season
    data.loc[data.month.isin([1, 2, 3]), 'season'] = 'winter'
    data.loc[data.month.isin([4, 5, 6]), 'season'] = 'spring'
    data.loc[data.month.isin([7, 8, 9]), 'season'] = 'summer'
    data.loc[data.month.isin([10, 11, 12]), 'season'] = 'autumn'
    print ('Filling season : done \n')


def train_test_base(X_tr, X_va, y_tr, y_va, models, metric, chrono=True):
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
    -metric: sklearn score function with the shape score(y_true, y_pred)
    -chrono: whether to time fit duration or not
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

        if metric == roc_auc_score: scores.append(metric(y_va, model.predict_proba(X_va)[:, 1]))
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
    -metric: sklearn score function with the shape score(y_true, y_pred)
    -cv: Cf. sklearn cross_val_score
    -chrono: whether to time fit duration or not
    """	
    df = pd.DataFrame(columns=list(models.keys()))

    scores = []
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

        scores.append(np.mean(cv_score))
        df[name] = cv_score

    df.loc['Mean score'] = scores
    if chrono: df.loc['Training time'] = times

    return df



def plot_confusion_matrix(y_pred, y, classes=None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
