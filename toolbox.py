"""
Machine learning helper toolbox
Author: Romain Chor

Libraries:
	numpy
	pandas
	matplotlib.pyplot
	seaborn
	sklearn.model_selection, sklearn.metrics

Functions:
	missing_val
	nan_filling
	rmse
	mape
	filling_season
	train_test_base
	train_test_random
	train_test_cv
	plot_confusion_matrix
	display_side_by_side
	extract_json

Classes:
	Blender
"""

print(__doc__)

#------------------------------------------------------------------------
import os, json
import itertools
import time
from IPython.display import display_html
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, make_scorer, roc_auc_score, log_loss, average_precision_score
#------------------------------------------------------------------------


def missing_val(df):
	"""
	Indicates which columns of df contain missing values, with count and percentage.

	df: Pandas DataFrame
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
	return None


def rmse(targets, predictions):
	"""Calculates RMSE between predicitions and ground truth targets"""
	return np.sqrt(np.mean((predictions - targets) ** 2))


def mape(targets, predictions):
	"""Calculates MAPE between predicitions and ground truth targets"""
	return np.mean(np.abs((predictions - targets)/targets))*100


def filling_season(data):
	"""
	Adds a 'season' feature based on a 'month' feature to data.

	data: a Pandas DataFrame
	"""
	data.loc[data.month.isin([1, 2, 3]), 'season'] = 'winter'
	data.loc[data.month.isin([4, 5, 6]), 'season'] = 'spring'
	data.loc[data.month.isin([7, 8, 9]), 'season'] = 'summer'
	data.loc[data.month.isin([10, 11, 12]), 'season'] = 'autumn'
	print ('Filling season : done \n')


def train_test_base(X_tr, X_va, y_tr, y_va, models, metric, chrono=True):
	"""
	Train and test given models, train and validation/test sets.

	X_tr, X_va: training (resp. validation/test) features, Pandas DataFrames or Numpy arrays
	y_tr, y_va: trainig (resp. validation/test) targets, Numpy arrays
	models: dict with models names as keys and sklearn object as values
	metric: sklearn score function with the shape score(y_true, y_pred)
	chrono: whether to time fit duration or not, bool (default True)
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

	    if metric in [roc_auc_score, log_loss, average_precision_score]: scores.append(metric(y_va, model.predict_proba(X_va)[:, 1]))
	    else: scores.append(metric(y_va, model.predict(X_va)))

	df.loc['Score'] = scores
	if chrono: df.loc['Training time'] = times

	return df


def train_test_random(X, y, models, metric, seed=None, test_size=0.2, chrono=True):
	"""
	Train and test given models: splits datasets using train_test_split then for each model, times fit duration and evaluates on a validation set.

	X: training features, Pandas DataFrame or Numpy array
	y: training target, Numpy array
	models: dict with models names as keys and sklearn object as values
	metric: score function with the shape score(y_true, y_pred)
	chrono: whether to time fit duration or not, bool (default True)
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
	Train and test given models using cross-validation: times fit duration and computes cross validation scores for each model.

	X: training features, Pandas DataFrame or Numpy array
	y: training target, Numpy array
	models: dict with models names as keys and sklearn object as values
	metric: str respecting sklearn metrics names (Cf. doc) (recommended) or sklearn metric function with the shape metric(y_true, y_pred)
	cv: Cf. sklearn cross_val_score
	chrono: whether to time fit duration or not, bool (default True)
	"""
	warnings.warn("Be careful if you give a score function for the 'metric' parameter as cross_val_score from sklearn gives sometimes weird results...")
	warnings.warn("To avoid that give a str name. Ex: metric='roc_auc'")
	df = pd.DataFrame(columns=list(models.keys()))

	means, stds = [], []
	times = []

	if type(metric) is str: scorer = metric
	else: scorer = make_scorer(metric)

	for name, model in models.items():
	    if chrono:
	        start = time.time()
	        cv_score = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)
	        end = time.time()
	        times.append(end-start)
	    else:
	        cv_score = cross_val_score(model, X, y, scoring=scorer, cv=cv, n_jobs=-1)

	    means.append(np.mean(cv_score))
	    stds.append(np.std(cv_score))
	    df[name] = cv_score

	df.loc['Mean score'] = means
	df.loc['Std'] = stds
	if chrono: df.loc['Training time'] = times

	return df


class Blender():
	"""
	Class for blending aggregation method.
	"""
	def __init__(self, models, metric):
		"""
		models: dict with name of model as key and object instance with fit method
		metric: metric function with the shape metric(y_true, y_pred)
		"""
		self.models = models
		self.metric = metric

	def fit(self, X, y):
		"""
		X: training features, Pandas DataFrame or Numpy array
		y: training target, Numpy array
		"""
		for name, __ in self.models.items():
		    self.models[name].fit(X, y)

	def predict(self, X):
		"""
		X: test features, to predict associated targets, Pandas DataFrame or Numpy array
		"""
		y_pred = 0
		for __, model in self.models.items():
			if self.metric in [roc_auc_score, log_loss, average_precision_score]:
			    y_pred += model.predict_proba(X)[:, 1]
			else:
				y_pred += model.predict(X)
		y_pred /= len(self.models)

		return y_pred

	def score(self, X, y):
		"""
		X: test features, to predict associated targets, Pandas DataFrame or Numpy array
		y: ground truth test target, Numpy array
		"""
		y_pred = self.predict(X)
		return self.metric(y, y_pred)

	def cv_score(self, X, y, cv=3):
		"""
		Performs K-folds cross validation and return cross validation scores.
		X: training features, Pandas DataFrame or Numpy array
		y: training target, Numpy array
		cv: number of splits for K-folds, int (default 3)
		"""
		kf = KFold(n_splits=cv)
		scores = pd.DataFrame(columns=['Score'])
		i = 0
		for train_index, val_index in kf.split(X):
		    X_tr, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
		    y_tr, y_val = y[train_index], y[val_index]
		    y_pred = 0
		    for __, model in self.models.items():
		        model.fit(X_tr, y_tr)
		        if self.metric in [roc_auc_score, log_loss, average_precision_score]:
		            y_pred += model.predict_proba(X_val)[:, 1]
		        else:
		            y_pred += model.predict(X_val)
		    y_pred /= len(self.models)
		    scores.loc[str(i)] = self.metric(y_val, y_pred)
		    i += 1
		scores.loc['Mean'] = np.mean(scores['Score'])

		return scores

# def blending_cv(X, y, models, metric, cv=3):
# 	"""
# 	Performs cross-validated models blending and returns scores.
# 	models: dict with model name as key and Sklearn model instance as value
# 	-X: pandas Dataframe
# 	-y: numpy array
# 	-cv: number of folds
# 	"""
# 	warnings.warn("Be careful if you give a score function for the 'metric' parameter as cross_val_score from sklearn gives sometimes weird results...")
# 	warnings.warn("To avoid that give a str name. Ex: metric='roc_auc'")
#
# 	if type(metric) is str: scorer = metric
# 	else: scorer = make_scorer(metric)
#
# 	kf = KFold(n_splits=cv)
# 	scores = pd.DataFrame()
# 	i = 0
# 	for train_index, val_index in kf.split(X):
# 	    X_tr, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
# 	    y_tr, y_val = y[train_index], y[val_index]
# 	    y_pred = 0
# 	    for __, model in models.items():
# 	        model.fit(X_tr, y_tr)
# 	        if metric in [roc_auc_score, log_loss, average_precision_score]:
# 	            y_pred += model.predict_proba(X_val)[:, 1]
# 	        else:
# 	            y_pred += model.predict(X_val)
# 	    y_pred /= len(models)
# 	    scores.loc[str(i)] = scorer(y_val, y_pred)
# 	    i += 1
# 	scores.loc['Mean']
#
# 	return scores


def plot_confusion_matrix(y_pred, y, classes=None, normalize=False):
	"""
	Plots the confusion matrix (only for classification!)
	Normalization can be applied by setting `normalize=True`.

	y_pred, y: predictions and ground truth labels, Numpy arrays or Pandas series
	classes: classes for classification, list or Numpy array (default None)
	normalize: whether to normalize counts or not (default False)
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


def display_side_by_side(args):
	"""
	Displays Pandas DataFrames side by side (for Jupyter Notebook env).

	args: list/tuple of Pandas DataFrames
	"""
	html_str = ''
	for df in args:
	    html_str += df.to_html()
	display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)


def extract_json(base_dir, NB=500, verbose=True):
    """
    Extracts the NB first json files from base_dir to a list of dictionnaries.

	base_dir: directory containing files to extract, str
	NB: number of files to extract (default 500), int
    verbose: whether to print informations on extraction or not.
    """
    #Get all files in the directory
    i = 0
    data_list = []
    start = time.time()
    for file in os.listdir(base_dir):
        #If file is a json, construct it's full path and open it, append all json data to list
        if 'json' in file and i < NB:
            json_path = os.path.join(base_dir, file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            data_list.append(json_data)
            i += 1
    end = time.time()

    if verbose:
        print("Nb of files extracted: ", len(data_list))
        print("Extraction time: {0:.2f}s".format(end-start))

    return data_list
