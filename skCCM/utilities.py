#
# Metrics for scoring predictions from CCM
#

import numpy as np
from scipy import stats as stats

def corrcoef(preds, actual):
    """Correlation Coefficient between predicted and actual values.

    Parameters
    ----------
    preds : 1d array
        Predicted values.
    actual : 1d array
        Actual values from the testing set.

    Returns
    -------
    cc : float
    	Returns the correlation coefficient between preds and actual.
    """

    cc = np.corrcoef(preds,actual)[0,1]

    return cc


def variance_explained(preds, actual):
    """Explained variance between predicted values and actual values.

    Parameters
    ----------
    preds : 1d array
        Predicted values.
    actual : 1d array
        Actual values from the testing set.

    Returns
    -------
    cc : float
        Returns the explained variance between preds and actual.
    """

    cc = np.var(preds - actual) / np.var(actual)

    return cc


def score(preds, actual):
    """The coefficient R^2 is defined as (1 - u/v), where u is the regression
    sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
    sum of squares ((y_true - y_true.mean()) ** 2).sum(). Best possible
    score is 1.0, lower values are worse.

    Parameters
    ----------
    preds : 1d array
        Predicted values.
    actual : 1d array
        Actual values from the testing set.

    Returns
    -------
    cc : float
        Returns the coefficient of determiniation between preds and actual.
    """

    u = np.square(actual - preds ).sum()
    v = np.square(actual - actual.mean()).sum()
    r2 = 1 - u/v

    return r2


def feature_scale(X):
    """Scales features between 0 and 1.

    Parameters
    ----------
    X : 1d array
        Time series values to be scaled.

    Returns
    -------
    scaled : 1d array
        Scaled array.
    """

    top = X - np.min(X)
    bot = np.max(X) - np.min(X)
    scaled = top/bot

    return scaled


def train_test_split(x1, x2, percent=.75):
    """Splits the embedded time series into a training set and testing set.

    Parameters
    ----------
    x1 : 2D array
        Embed time series.
    x2 : 2D array
        Embed time series.
    percent : float
        Percent to use for training set.

    Returns
    -------
    x1tr : 2D array
    x1te : 2D array
    x2tr : 2D array
    x2te : 2D array
    """

    if len(x1) != len(x2):
        print("X1 and X2 are different lengths!")

    split = int(len(x1)*percent)

    x1tr = x1[:split]
    x2tr = x2[:split]

    x1te = x1[split:]
    x2te = x2[split:]

    return x1tr, x1te, x2tr, x2te


def exp_weight(X):
    """Calculates the weights based on the distances.

    e^(-distances/min(distances,axis=1))
    Parameters
    ----------
    X : 2D array
        Distances from the training set to the testing set.

    Returns
    -------
    W : 2D array
        Exponentially weighted and normalized weights.
    """

    #add a small number so it stays defined
    norm = X[:,[0]] +.00001

    numer = np.exp(-X/norm)
    denom = np.sum(numer,axis=1,keepdims=True)

    W = numer/denom

    return W

def in_library_len(ind, dist, lib_len):
    """Returns the filtered indices and distances that are in that specific
    library length. This allows the distances to only be calculated once.

    This was created in an attempt to speed up the algorithm. It turns out the
    naive implementation was faster.

    Parameters
    ----------
    ind : 2d array
        Indices to be filtered.
    dist : 2d array
        Distances to be filtered.
    lib_len : int
        What indices to keep.

    Returns
    -------
    filt_ind : 2d array
        Filtered indices.
    filt_dist : 2d array
        Filtered distances.

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

def in_library_len_keep(ind, dist, lib_len, keep):
    """Returns the filtered indices and distances that are in that specific
    library length. Only returns the top n depending on the value of keep.

    This allows the distances to only be calculated once. This algorithm is
    slow for large matrices. The naive implementation of this algorithm is
    actually faster.

    Parameters
    ----------
    ind : 2d array
        Indices to be filtered.
    dist : 2d array
        Distances to be filtered.
    lib_len : int
        What indices to keep.
    keep : int
        How much of the matrix to keep.

    Returns
    -------
    filt_ind : 2d array
        Filtered indices.
    filt_dist : 2d array
        Filtered distances.

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




def throw_out_nn_indices(ind, dist, Xind):
    """Throw out near neighbor indices that are used to embed the time series.

    This is an attempt to get around the problem of autocorrelation.

    Parameters
    ----------
    ind : 2d array
        Indices to be filtered.
    dist : 2d array
        Distances to be filtered.
    Xind : int
        Indices to filter.

    Returns
    -------
    filt_ind : 2d array
        Filtered indices.
    filt_dist : 2d array
        Filtered distances.
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
    """Finds where the indices are in the rest of feature matrix. This assures
    that the correct indices are dropped.

    Parameters
    ----------
    X : 2D array
        The embed indices. This is the same shape as the actual embedded time
        series.

    Returns
    -------
    conf_ind : 1d array
        Conflicting indices to be dropped.
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
