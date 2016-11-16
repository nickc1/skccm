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
def in_library_len(ind,dist,lib_len,keep=4):
	"""
	Returns the filtered indices and distances that are in that specific
	library length. This allows the distances to only be calculated once.
	ind : np.array, indices to be filtered
	dist : np.array, distances to be filtered
	val : what indices to keep
	"""


	# mask = ind < lib_len
	# filt_ind = ind[mask].reshape(-1,lib_len)
	# filt_dist = dist[mask].reshape(-1,lib_len)

	r,c = np.where(ind<lib_len)

	r = r.reshape(-1,lib_len)[:,:keep].ravel()
	c = c.reshape(-1,lib_len)[:,:keep].ravel()

	filt_ind = ind[r,c].reshape(-1,keep)
	filt_dist = dist[r,c].reshape(-1,keep)

	return filt_ind, filt_dist


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


def exp_weight(X, weights='exponential_paper'):
	"""
	Calculates the weights based on the distances.
	Parameters
	----------
	d1 : distances from X1_train to X1_test
	d2 : distances from X2_train to X2_test
	"""

	#add a small number so it stays defined
	norm = X[:,[0]] +.00001

	numer = np.exp(-X/norm)
	denom = np.sum(numer,axis=1,keepdims=True)

	W = numer/denom

	return W
