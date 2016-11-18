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

def in_library_len(ind,dist,lib_len):
	"""
	Returns the filtered indices and distances that are in that specific
	library length. This allows the distances to only be calculated once.
	ind : np.array, indices to be filtered
	dist : np.array, distances to be filtered
	val : what indices to keep

	"""


	mask = ind < lib_len
	filt_ind = ind[mask].reshape(-1,lib_len)
	filt_dist = dist[mask].reshape(-1,lib_len)

	# this was slower :(
	# r,c = np.where(ind<lib_len)
	#
	# r = r.reshape(-1,lib_len)[:,:keep].ravel()
	# c = c.reshape(-1,lib_len)[:,:keep].ravel()
	#
	# filt_ind = ind[r,c].reshape(-1,keep)
	# filt_dist = dist[r,c].reshape(-1,keep)

	return filt_ind, filt_dist

def in_library_len_keep(ind,dist,lib_len,keep):
	"""
	Returns the filtered indices and distances that are in that specific
	library length. Only returns the top n depending on the value of keep.
	This allows the distances to only be calculated once. This algorithm is
	slow for large matrices.
	ind : np.array, indices to be filtered
	dist : np.array, distances to be filtered
	keep : what indices to keep

	"""
	ind_store = []
	dist_store = []
	for i in range(len(ind)):
		mask = ind[i] < lib_len
		ind_store.append( ind[i][mask] )
		dist_store.append( dist[i][mask] )

	ind_store = [x[:keep] for x in ind_store]
	dist_store = [x[:keep] for x in dist_store]

	return np.vstack(ind_store), np.vstack(dist_store)




def throw_out_nn_indices(dist,ind, Xind):
	"""
	Throw out near neighbor indices that are used to embed the time series.

	dist : distances to all the near neighbors
	ind : near neighbor indices
	X : embedded X time series
		This is used to calculate the indices. It assumes a sequential
		embedding. For example: [[1,2],[3,4],...]
	Xind : list of lists.
		These are the conflicting indices that must be removed. These are
		calcuated using conflicting_indices().
	"""

	ind_store = []
	dist_store = []

	#iterate through each row
	for i in range(len(Xind)):

		xrow = Xind[i]
		indrow = ind[i]
		distrow = dist[i]
		mask = np.ones(len(indrow),dtype=bool)

		for val in xrow:
			mask[indrow == val] = False

		ind_store.append( indrow[mask] )
		dist_store.append(distrow[mask])

	#keep up to the shortest mask. This is so that we can vstack them
	ind_len = min( [len(m) for m in ind_store] )

	#make all lists the same size for concatenation
	ind_store = [m[:ind_len] for m in ind_store]
	dist_store = [m[:ind_len] for m in dist_store]


	ind_store = np.vstack(ind_store)
	dist_store = np.vstack(dist_store)

	return dist_store, ind_store

def conflicting_indices(X):
	"""
	Finds where the indices are in the rest of feature matrix. This assures
	that the correct indices are dropped.

	X : The embed indices. This is the same shape as the actual embedded time
		series.
	"""

	conf_ind = []
	for i in range(len(X)):

		inds = [] #where to store

		#iterate through all other rows
		for j in range(len(X)):

			#check where they intersect
			if len(set( X[i] ).intersection( X[j] ))>0:
				inds.append(j)

		conf_ind.append(inds)

	return conf_ind
