"""
Data for analyzing causality.
By Nick Cortale

Classes:
	ccm
	embed

Paper:
Detecting Causality in Complex Ecosystems
George Sugihara et al. 2012

Thanks to Kenneth Ells and Dylan McNamara

TODO: This can be made way faster (I think) by only calculting the distances once and then
Chopping it to a specific library length.
"""



import numpy as np
from sklearn import neighbors
from sklearn import metrics
import skCCM.utilities as ut
import pandas as pd




class ccm:
	"""
	Convergent cross mapping for two embedded time series
	"""

	def __init__(self, weights='exponential_paper', verbose=False,
				score_metric='corrcoef' ):
		"""
		Parameters
		----------
		weights : weighting scheme for predictions
		    - exponential_paper : weighting scheme from paper
		verbose : prints out calculation status
		score : how to score the predictions
			-'score'
			-'corrcoef'
		"""

		self.weights = weights
		self.verbose = verbose
		self.score_metric = score_metric

	def predict_causation(self,X1_train,X1_test,X2_train,X2_test,lib_lens):
		"""
		Wrapper for predicting causation as a function of library length.
		X1_train : embedded train series of shape (num_samps,embed_dim)
		X2_train : embedded train series of shape (num_samps,embed_dim)
		X1_test : embedded test series of shape (num_samps, embed_dim)
		X2_test : embedded test series of shape (num_samps, embed_dim)
		lib_lens : which library lengths to use for prediction
		near_neighs : how many near neighbors to use (int)
		how : how to score the predictions
			-'score'
			-'corrcoef'
		"""

		sc1_store = np.zeros(len(lib_lens))
		sc2_store = np.zeros(len(lib_lens))

		for ii,lib in enumerate(lib_lens):

			self.fit(X1_train[0:lib],X2_train[0:lib])
			self.predict(X1_test, X2_test)
			sc1, sc2 = self.score()

			sc1_store[ii] = sc1
			sc2_store[ii] = sc2

		return sc1_store, sc2_store

	def fit(self,X1_train,X2_train):
		"""
		Fit the training data for ccm. Amount of near neighbors is set to be
		embedding dimension plus one. Creates seperate near neighbor regressors
		for X1 and X2 independently. Also Calculates the distances to each
		sample.

		X1 : embedded time series of shape (num_samps,embed_dim)
		X2 : embedded time series of shape (num_samps,embed_dim)
		near_neighs : number of near neighbors to use
		"""

		# Save X1_train and X2_train for prediction later. Confusing,
		# but we need to make predictions about our testing set using these.
		self.X1_train = X1_train
		self.X2_train = X2_train

		near_neighs = X1_train.shape[1] + 1

		self.knn1 = neighbors.KNeighborsRegressor(near_neighs)
		self.knn2 = neighbors.KNeighborsRegressor(near_neighs)

		self.knn1.fit(X1_train, X1_train)
		self.knn2.fit(X2_train, X2_train)


	def dist_calc(self,X1_test,X2_test):
		"""
		Calculates the distance from X1_test to X1_train and X2_test to
		X2_train.

		Returns
		-------
		dist1 : distance from X1_train to X1_test
		ind1 : indices that correspond to the closest
		dist2 : distance from X2_train to X2_test
		ind2 : indices that correspond to the closest
		"""

		dist1,ind1 = self.knn1.kneighbors(X1_test)
		dist2,ind2 = self.knn2.kneighbors(X2_test)

		if self.verbose: print("distances calculated")

		return dist1, ind1, dist2, ind2


	def weight_calc(self,d1,d2):
		"""
		Calculates the weights based on the distances.
		Parameters
		----------
		d1 : distances from X1_train to X1_test
		d2 : distances from X2_train to X2_test
		"""

		if self.weights is 'linear':

			numer_w1 = 1./d1
			denom_w1 = np.sum(numer_w1, axis=1)

			numer_w2 = 1./d2
			denom_w2 = np.sum(numer_w2, axis=1)

		elif self.weights is 'uniform':

			numer_w1 = np.ones_like(d1)
			denom_w1 = np.sum(np.ones_like(d1),axis=1)

			numer_w2 = np.ones_like(d2)
			denom_w2 = np.sum(np.ones_like(d2),axis=1)

		elif self.weights is 'exponential_paper':

			#add a small number so it stays defined
			norm1 = d1[:,0].reshape(len(d1),1) +.00001

			numer_w1 = np.exp(-d1/norm1)
			denom_w1 = np.sum(numer_w1,axis=1)

			norm2 = d2[:,0].reshape(len(d2),1) +.00001

			numer_w2 = np.exp(-d2/norm2)
			denom_w2 = np.sum(numer_w2,axis=1)

		W1 = numer_w1/denom_w1[:,np.newaxis]
		W2 = numer_w2/denom_w2[:,np.newaxis]

		if self.verbose: print("weights calculated")

		return W1, W2

	def predict(self,X1_test,X2_test):
		"""
		Make a prediction

		Parameters
		----------
		X1 : test set
		X2 : test set

		"""
		#store X1_test and X2_test for use later
		self.X1_test = X1_test
		self.X2_test = X2_test

		#calculate the distances and weights
		dist1, ind1, dist2, ind2 = self.dist_calc(X1_test,X2_test)
		W1, W2 = self.weight_calc(dist1,dist2)

		X1_pred = np.empty(X1_test.shape, dtype=np.float)
		X2_pred = np.empty(X2_test.shape, dtype=np.float)

		for j in range(self.X1_train.shape[1]):

			#flip the weights and indices
			X1_pred[:, j] = np.sum(self.X1_train[ind2, j] * W2, axis=1)
			X2_pred[:, j] = np.sum(self.X2_train[ind1, j] * W1, axis=1)

		self.X1_pred = X1_pred
		self.X2_pred = X2_pred

		if self.verbose: print("predictions made")

	def score(self):
		"""
		Evalulate the predictions

		how : how to score the predictions
			-'score'
			-'corrcoef'
		"""

		num_preds = self.X1_pred.shape[1]

		sc1 = np.empty((1,num_preds))
		sc2 = np.empty((1,num_preds))

		for ii in range(num_preds):

			p1 = self.X1_pred[:,ii]
			p2 = self.X2_pred[:,ii]

			if self.score_metric == 'score':
				sc1[0,ii] = ut.score(p1,self.X1_test[:,ii])
				sc2[0,ii] = ut.score(p2,self.X2_test[:,ii])

			if self.score_metric == 'corrcoef':
				sc1[0,ii] = ut.corrcoef(p1,self.X1_test[:,ii])
				sc2[0,ii] = ut.corrcoef(p2,self.X2_test[:,ii])

		return np.mean(sc1,axis=1), np.mean(sc2,axis=1)


class embed:

	def __init__(self,X):
		"""
		Parameters
		----------
		X : series or dataframe,
		"""
		if type(X) is pd.pandas.core.frame.DataFrame:
			self.df = X
		else:
			self.X = X


	def df_mutual_information(self,max_lag):
		"""
		Calculates the mutual information along each row of a time series.
		Ensure that the time series is continuous in time and sampled regularly.
		You can resample it hourly, daily, minutely etc. if needed.

		Parameters
		----------
		max_lag : int
			maximum amount to shift the time series
		Returns
		-------
		mi : dataframe, shape(max_lag,num_cols)
			columns are the columns of the original dataframe with rows being
			the mutual information
		"""

		cols = self.df.columns
		mi = np.empty((max_lag, len(cols)))

		for i,col in enumerate(cols):

			self.X = self.df[col].values
			mi[:,i] = self.mutual_information(max_lag)

		mi = pd.DataFrame(mi,columns=cols)

		return mi

	def mutual_information(self,max_lag):
		"""
		Calculates the mutual information between the an unshifted time series
		and a shifted time series. Utilizes scikit-learn's implementation of
		the mutual information found in sklearn.metrics.

		Parameters
		----------
		max_lag : integer
		    maximum amount to shift the time series

		Returns
		-------
		m_score : 1-D array
		    mutual information at between the unshifted time series and the
		    shifted time series
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



	def embed_vectors_1d(self,lag,embed):
		"""
		Embeds vectors from a one dimensional time series in
		m-dimensional space.

		Parameters
		----------
		X : array
			A 1-D array representing the training or testing set.

		lag : int
			lag values as calculated from the first minimum of the mutual info.

		embed : int
			embedding dimension, how many lag values to take

		predict : int
			distance to forecast (see example)


		Returns
		-------
		features : array of shape [num_vectors,embed]
			A 2-D array containing all of the embedded vectors

		Example
		-------
		X = [0,1,2,3,4,5,6,7,8,9,10]

		em = 3
		lag = 2
		predict=3

		returns:
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
