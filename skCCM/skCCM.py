#
# Data for analyzing causality.
# By Nick Cortale
#
# Classes:
# 	ccm
# 	embed
#
# Paper:
# Detecting Causality in Complex Ecosystems
# George Sugihara et al. 2012
#
# Thanks to Kenneth Ells and Dylan McNamara
#
# Notes:
# Originally I thought this can be made way faster by only calculting the
# distances once and then chopping it to a specific library length. It turns out
# that calculating the distances is cheaper than filtering the indices.
#



import numpy as np
from sklearn import neighbors
from sklearn import metrics
from . import utilities
import pandas as pd
import time




class CCM:
    """Convergent cross mapping for two embedded time series.

    Parameters
    ----------
    weights : str
        Weighting scheme for predictions. Options:

        - 'exp' : exponential weighting
    score : str
        How to score the predictions. Options:

        - 'score'
        - 'corrcoef'
    verbose : bool
        Prints out calculation status.

    """

    def __init__(self, weights='exp', score_metric='corrcoef', verbose=False):

        self.weights = weights
        self.score_metric = score_metric
        self.verbose = verbose


    def fit(self, X1_train, X2_train):
        """Fit the training data for ccm. Can be thought of as reconstructing the
        shadow manifolds of each time series.

        Amount of near neighbors is set to be embedding dimension plus one.
        Creates seperate near neighbor regressors for X1 and X2 independently.

        Parameters
        ----------
        X1_train : 2d array
            Embed time series of shape (num_samps,embed_dim).
        X2_train : 2d array
            Embed time series of shape (num_samps,embed_dim).
        """

        # Save X1_train and X2_train for prediction later. Confusing,
        # but we need to make predictions about our testing set using these.
        self.X1_train = X1_train
        self.X2_train = X2_train

        #to sorround a point, there must be ndim + 1 points
        near_neighs = X1_train.shape[1] + 1

        self.knn1 = neighbors.KNeighborsRegressor(near_neighs)
        self.knn2 = neighbors.KNeighborsRegressor(near_neighs)

    def predict(self, X1_test, X2_test, lib_lengths):
        """Make a prediction.

        Parameters
        ----------
        X1_test : 2d array
            Embed time series of shape (num_samps,embed_dim).
        X2_test : 2d array
            Embed time series of shape (num_samps,embed_dim).
        lib_lengths : 1d array of ints
            Library lengths to test.

        Returns
        -------
        X1_pred : list of 2d arrays
            Predictions for each library length.
        X2_pred : list of 2d arrays
            Predictions for each library length.

        """

        #store X1_test and X2_test for use later
        self.X1_test = X1_test
        self.X2_test = X2_test

        X1_pred = []
        X2_pred = []


        for liblen in lib_lengths:

            x1_p = np.empty(X1_test.shape)
            x2_p = np.empty(X2_test.shape)

            #keep only the indices that are less than library length
            self.knn1.fit(self.X1_train[:liblen], self.X1_train[:liblen])
            self.knn2.fit(self.X2_train[:liblen], self.X2_train[:liblen])

            dist1,ind1 = self.knn1.kneighbors(X1_test)
            dist2,ind2 = self.knn2.kneighbors(X2_test)


            for j in range(self.X1_train.shape[1]):

                W1 = utilities.exp_weight(dist1)
                W2 = utilities.exp_weight(dist2)

                #flip the weights and indices
                x1_p[:, j] = np.sum(self.X1_train[ind2, j] * W2, axis=1)
                x2_p[:, j] = np.sum(self.X2_train[ind1, j] * W1, axis=1)

            X1_pred.append(x1_p)
            X2_pred.append(x2_p)

        self.X1_pred = X1_pred
        self.X2_pred = X2_pred

        return X1_pred, X2_pred


    def score(self, how='corrcoef'):
        """Evalulate the predictions.

        Parameters
        ----------
        how : string
            How to score the predictions. Options:
            - 'score'
            - 'corrcoef'

        Returns
        -------
        score_1 : 2d array
            Scores for the first time series using the weights from the second
            time series.
        score_2 : 2d array
            Scores for the second time series using the weights from the first
            time series.
        """

        num_preds = self.X1_train.shape[1]

        score_1 = []
        score_2 = []

        for x1_p, x2_p in zip(self.X1_pred, self.X2_pred):

            sc1 = np.empty(num_preds)
            sc2 = np.empty(num_preds)

            for ii in range(num_preds):

                p1 = x1_p[:,ii]
                p2 = x2_p[:,ii]

                if self.score_metric == 'score':
                    sc1[ii] = utilities.score(p1,self.X1_test[:,ii])
                    sc2[ii] = utilities.score(p2,self.X2_test[:,ii])

                if self.score_metric == 'corrcoef':
                    sc1[ii] = utilities.corrcoef(p1,self.X1_test[:,ii])
                    sc2[ii] = utilities.corrcoef(p2,self.X2_test[:,ii])

            score_1.append( np.mean(sc1) )
            score_2.append( np.mean(sc2) )

        return score_1, score_2


class Embed:
    """Embed a time series.

    Parameters
    ----------
    X : 1D array
        Time series to be embed.
    """

    def __init__(self,X):

        if type(X) is pd.pandas.core.frame.DataFrame:
            self.df = X
        else:
            self.X = X


    def df_mutual_information(self, max_lag):
        """Calculates the mutual information along each column of a dataframe.

        Ensure that the time series is continuous in time and sampled regularly.
        You can resample it hourly, daily, minutely etc. if needed.

        Parameters
        ----------
        max_lag : int
        	maximum amount to shift the time series

        Returns
        -------
        mi : dataframe
        	columns are the columns of the original dataframe with rows being
        	the mutual information. shape(max_lag,num_cols)
        """

        cols = self.df.columns
        mi = np.empty((max_lag, len(cols)))

        for i,col in enumerate(cols):

            self.X = self.df[col].values
            mi[:,i] = self.mutual_information(max_lag)

        mi = pd.DataFrame(mi,columns=cols)

        return mi

    def mutual_information(self, max_lag):
        """Calculates the mutual information between the an unshifted time
        series and a shifted time series.

        Utilizes scikit-learn's implementation of the mutual information found
        in sklearn.metrics.

        Parameters
        ----------
        max_lag : integer
            Maximum amount to shift the time series.

        Returns
        -------
        m_score : 1-D array
            Mutual information at between the unshifted time series and the
            shifted time series,
        """

        #number of bins - say ~ 20 pts / bin for joint distribution
        #and that at least 4 bins are required
        N = max(self.X.shape)
        num_bins = max(4.,np.floor(np.sqrt(N/20)))
        num_bins = int(num_bins)

        m_score = np.zeros((max_lag))

        for jj in range(max_lag):
            lag = jj+1

            ts = self.X[0:-lag]
            ts_shift = self.X[lag::]

            min_ts = np.min(self.X)
            max_ts = np.max(self.X)+.0001 #needed to bin them up

            bins = np.linspace(min_ts,max_ts,num_bins+1)

            bin_tracker = np.zeros_like(ts)
            bin_tracker_shift = np.zeros_like(ts_shift)

            for ii in range(num_bins):

                locs = np.logical_and( ts>=bins[ii], ts<bins[ii+1] )
                bin_tracker[locs] = ii

                locs_shift = np.logical_and( ts_shift>=bins[ii], ts_shift<bins[ii+1] )
                bin_tracker_shift[locs_shift]=ii

            m_score[jj] = metrics.mutual_info_score(bin_tracker,bin_tracker_shift)
        return m_score



    def embed_vectors_1d(self, lag, embed):
        """Embeds vectors from a one dimensional time series in m-dimensional
        space.

        Parameters
        ----------
        X : 1d array
            Training or testing set.
        lag : int
            Lag value as calculated from the first minimum of the mutual info.
        embed : int
            Embedding dimension. How many lag values to take.
        predict : int
            Distance to forecast (see example).

        Returns
        -------
        features : 2d array
            Contains all of the embedded vectors. Shape (num_vectors,embed).

        Example
        -------
        >>> X = [0,1,2,3,4,5,6,7,8,9,10]
        em = 3
        lag = 2
        predict=3

        >>> embed_vectors_1d
        features = [[0,2,4], [1,3,5], [2,4,6], [3,5,7]]
        """

        tsize = self.X.shape[0]
        t_iter = tsize-(lag*(embed-1))

        features = np.zeros((t_iter,embed))

        for ii in range(t_iter):

            end_val = ii+lag*(embed-1)+1

            part = self.X[ii : end_val]

            features[ii,:] = part[::lag]

        return features
