

import numpy as np
from sklearn import neighbors
from sklearn import metrics
import metrics as mets



class ccm:
	"""
	Convergent cross mapping for two embedded time series
	"""

	def __init__(self, weights='exponential_paper',verbose=False):
		"""
		Parameters
		----------
		weights : weighting scheme for predictions
		    - exponential_paper : weighting scheme from paper
		verbose : prints out calculation status
		"""

		self.weights = weights
		self.verbose = verbose

	def predict_causation(self,X1,X2,near_neighs,how='score'):
		"""
		Wrapper to call all the other needed functions
		X1 : embedded time series of shape (num_samps,embed_dim)
		X2 : embedded time series of shape (num_samps,embed_dim)
		near_neighs : how many near neighbors to use (int)
		how : how to score the predictions
		-'score'
		-'corrcoef'
		"""

		self.fit(X1,X2,near_neighs)
		self.dist_calc_to_self()
		self.weight_calc()
		self.predict()

		sc1, sc2 = self.score(how)

		return sc1, sc2

	def predict_causation_lib_len(self,X1,X2,lib_lens,near_neighs,how='score'):
		"""
		Wrapper for predicting causation as a function of library length.
		X1 : embedded time series of shape (num_samps,embed_dim)
		X2 : embedded time series of shape (num_samps,embed_dim)
		lib_lens : which library lengths to use for prediction
		near_neighs : how many near neighbors to use (int)
		how : how to score the predictions
			-'score'
			-'corrcoef'
		"""
		s_shape = (len(lib_lens),)
		sc1_store = np.empty(s_shape)
		sc2_store = np.empty(s_shape)

		for ii,lib in enumerate(lib_lens):

			self.fit(X1[0:lib],X2[0:lib],near_neighs)
			self.dist_calc_to_self()
			self.weight_calc()
			self.predict()

			sc1, sc2 = self.score(how)
			sc1_store[ii] = sc1
			sc2_store[ii] = sc2

		return sc1_store, sc2_store

	def fit(self,X1,X2,near_neighs):
		"""
		Initialize the data for ccm.

		X1 : embedded time series of shape (num_samps,embed_dim)
		X2 : embedded time series of shape (num_samps,embed_dim)
		near_neighs : number of near neighbors to use
		"""
		self.X1 = X1
		self.X2 = X2

		self.y1 = X1
		self.y2 = X2

		self.knn1 = neighbors.KNeighborsRegressor(near_neighs)
		self.knn2 = neighbors.KNeighborsRegressor(near_neighs)

		self.knn1.fit(X1, X1)
		self.knn2.fit(X2, X2)


	def dist_calc_to_self(self):
		"""
		Calculates the distance from X1 to X1 and X2 to X2.

		Note: Scrapes off the closest point, since that is the
		distance to itself. For example d1[:,1:]

		Returns
		-------
		self.dist1 : distance from X1 to X1
		self.ind1 : indices that correspond to the closest
		self.dist2 : distance from X2 to X2
		self.ind2 : indices that correspond to the closest
		"""

		d1,i1 = self.knn1.kneighbors(self.X1)
		d2,i2 = self.knn2.kneighbors(self.X2)

		self.dist1 = d1[:,1:]
		self.ind1 = i1[:,1:]

		self.dist2 = d2[:,1:]
		self.ind2 = i2[:,1:]

		if self.verbose: print("distances calculated")


	def weight_calc(self):
		"""
		Calculates the weights based on the distances.
		"""
		d1 = self.dist1
		d2 = self.dist2

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


		self.W1 = numer_w1/denom_w1[:,np.newaxis]

		self.W2 = numer_w2/denom_w2[:,np.newaxis]


		if self.verbose: print("weights calculated")

	def predict(self):
		"""
		Make a prediction for a certain number of near neighbors
		"""

		neigh_ind1 = self.ind1
		neigh_ind2 = self.ind2

		y1_pred = np.empty((self.X1.shape[0], self.X1.shape[1]), dtype=np.float)
		y2_pred = np.empty((self.X2.shape[0], self.X2.shape[1]), dtype=np.float)

		for j in range(self.y1.shape[1]):

			#flip the weights and indices
			y1_pred[:, j] = np.sum(self.y1[neigh_ind2, j] * self.W2, axis=1)
			y2_pred[:, j] = np.sum(self.y2[neigh_ind1, j] * self.W1, axis=1)

		self.y1_pred = y1_pred
		self.y2_pred = y2_pred

		if self.verbose: print("predictions made")

	def score(self,how='score'):
		"""
		Evalulate the predictions

		how : how to score the predictions
			-'score'
			-'corrcoef'
		"""

		num_preds = self.y1.shape[1]

		sc1 = np.empty((1,num_preds))
		sc2 = np.empty((1,num_preds))

		for ii in range(num_preds):

			p1 = self.y1_pred[:,ii]
			p2 = self.y2_pred[:,ii]

			if how == 'score':
				sc1[0,ii] = mets.score(p1,self.y1[:,ii])
				sc2[0,ii] = mets.score(p2,self.y2[:,ii])

			if how == 'corrcoef':
				sc1[0,ii] = mets.corrcoef(p1,self.y1[:,ii])
				sc2[0,ii] = mets.corrcoef(p2,self.y2[:,ii])

		return np.mean(sc1,axis=1), np.mean(sc2,axis=1)


	class embed:

	def __init__(self,X):
		"""
		Parameters
		----------
		X : series, 2d array, or 3d array to be embedded
		"""

		self.X = X

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
