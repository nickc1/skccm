"""
Metrics for scoring predictions from CCM
"""

import numpy as np
from scipy import stats as stats

def corrcoef(preds,actual):
	"""
	Correlation Coefficient of between predicted values and actual values

	Parameters
	----------

	preds : array shape (num samples,num targets)

	test : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
    """

	cc = np.corrcoef(preds,actual)[0,1]

	return cc


def varianceExplained(preds,actual):
	"""
	Explained variance between predicted values and actual values scaled
	to the most common prediction of the space

	Parameters
	----------

	preds : array shape (num samples,num targets)

	actual : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""


	cc = np.var(preds - actual) / np.var(actual)

	return cc


def score(preds,actual):
	"""
	The coefficient R^2 is defined as (1 - u/v), where u is the regression
	sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
	sum of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible
	score is 1.0, lower values are worse.

	Parameters
	----------

	preds : array shape (num samples,num targets)

	test : array of shape (num samples, num targets)
		actual values from the testing set

	Returns
	-------

	cc : float
		Returns the correlation coefficient
	"""

	u = np.square(actual - preds ).sum()
	v = np.square(actual - actual.mean()).sum()
	r2 = 1 - u/v

	return r2


def feature_scale(X):
	"""
	Scales the features between 0 and 1
	"""

	top = X - np.min(X)
	bot = np.max(X) - np.min(X)

	return top/bot

def train_test_split(x1,x2,percent=.75):
	"""
	Splits the embedded time series into a testing set and training set for
	use the ccm predict_casuation call. Returns x1tr, x1te, x2tr, x2te.

	Parameters
	----------
	x1 : embedded time series of shape (n_samps,embed_dim)
	x2 : embedded time series of shape (n_samps,embed_dim)
	percent : what percent to use for training set

	Returns
	-------
	x1tr:
	x1te:
	x2tr:
	x2te:
	"""

	if len(x1) != len(x2):
		print("X1 and X2 are different lengths!")

	split = int(len(x1)*percent)

	x1tr = x1[:split]
	x2tr = x2[:split]

	x1te = x1[split:]
	x2te = x2[split:]

	return x1tr, x1te, x2tr, x2te
