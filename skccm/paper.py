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
import skccm.utilities as ut
import pandas as pd
import time




class CCM:
	"""
	Convergent cross mapping for two embedded time series
	"""

	def __init__(self, weights='exp', score_metric='corrcoef', verbose=False):
		"""
		Parameters
		----------
		weights : weighting scheme for predictions
		    - exp : exponential weighting
		score : how to score the predictions
			-'score'
			-'corrcoef'
		verbose : prints out calculation status
		"""

		self.weights = weights
		self.score_metric = score_metric
		self.verbose = verbose


	def fit(self,X1,X2):
		"""
		Fit the training data for ccm. Creates seperate near neighbor regressors
		for X1 and X2 independently.

		X1 : embedded time series of shape (num_samps,embed_dim)
		X2 : embedded time series of shape (num_samps,embed_dim)
		near_neighs : string
			- 'sorround' : this is what the paper uses
			- 'all' : calculate the distance to all near neighbors
		"""

		# Save X1_train and X2_train for prediction later. Confusing,
		# but we need to make predictions about our testing set using these.
		self.X1 = X1
		self.X2 = X2

		#to sorround a point, there must be ndim + 1 points
		# we add two here because the closest neighbor is itself. so that is
		# going to be dropped.
		near_neighs = X1.shape[1] + 2

		self.knn1 = neighbors.KNeighborsRegressor(near_neighs)
		self.knn2 = neighbors.KNeighborsRegressor(near_neighs)

	def predict_no_drop(self,lib_lengths):
		"""
		Make a prediction

		Parameters
		----------
		X1_test : test set
		X2_test : test set
		lib_lengths : list of library lengths to test
		"""

		X1_pred = []
		X2_pred = []


		for liblen in lib_lengths:

			x1_p = np.empty(self.X1.shape)
			x2_p = np.empty(self.X2.shape)

			#keep only the indices that are less than library length
			self.knn1.fit(self.X1[:liblen], self.X1[:liblen])
			self.knn2.fit(self.X2[:liblen], self.X2[:liblen])

			dist1,ind1 = self.knn1.kneighbors(self.X1)
			dist2,ind2 = self.knn2.kneighbors(self.X2)

			#drop indices and distances to themselves
			dist1 = dist1[:,1:]
			dist2 = dist2[:,1:]
			ind1 = ind1[:,1:]
			ind2 = ind2[:,1:]

			for j in range(self.X1.shape[1]):

				W1 = ut.exp_weight(dist1)
				W2 = ut.exp_weight(dist2)

				#flip the weights and indices
				x1_p[:, j] = np.sum(self.X1[ind2, j] * W2, axis=1)
				x2_p[:, j] = np.sum(self.X2[ind1, j] * W1, axis=1)

			X1_pred.append(x1_p)
			X2_pred.append(x2_p)

		self.X1_pred = X1_pred
		self.X2_pred = X2_pred

		return X1_pred, X2_pred

	def predict_drop_in_list(self,lib_lengths,emb_ind1,emb_ind2):
		"""
		Make a prediction, but the same indices cant be matched with each other.
		Parameters
		----------
		lib_lengths : library lengths to Test
		e_ind1 : indices of the first embed time series.
		e_ind2 : indices of the second embed time series.
		"""

		X1_pred = []
		X2_pred = []

		#need to reset the class ot use all neighbors so that the appropriate
		# neighbors can be dropped for each class
		self.knn1 = neighbors.KNeighborsRegressor(len(self.X1))
		self.knn2 = neighbors.KNeighborsRegressor(len(self.X2))

		self.knn1.fit(self.X1, self.X1)
		self.knn2.fit(self.X2, self.X2)

		dist1,ind1 = self.knn1.kneighbors(self.X1)
		dist2,ind2 = self.knn2.kneighbors(self.X2)

		#find the conflicting indices
		conf1 = ut.conflicting_indices(emb_ind1)
		conf2 = ut.conflicting_indices(emb_ind2)


		#throw out the indices that are in the embedding
		dist1, ind1 = ut.throw_out_nn_indices(dist1,ind1,conf1)
		dist2, ind2 = ut.throw_out_nn_indices(dist2,ind2,conf2)

		n_sorround = self.X1.shape[1] + 1
		#flipping allows for a faster implentation as we can feed
		# ut.in_libary_len smaller and smaller arrays
		for liblen in lib_lengths:


			#keep only the indices that are less than library length
			#t0 = time.time()
			i_1, d_1 = ut.in_library_len_keep(ind1, dist1, liblen,n_sorround)
			i_2, d_2 = ut.in_library_len_keep(ind2, dist2, liblen,n_sorround)
			#t1 = time.time()

			#t0 = time.time()
			W1 = ut.exp_weight(d_1)
			W2 = ut.exp_weight(d_2)

			x1_p = np.empty(self.X1.shape)
			x2_p = np.empty(self.X2.shape)

			for j in range(self.X1.shape[1]):

				#flip the weights and indices
				x1_p[:, j] = np.sum(self.X1[i_2, j] * W2, axis=1)
				x2_p[:, j] = np.sum(self.X2[i_1, j] * W1, axis=1)
			#t1 = time.time()

			#print('second_loop:',np.around(t1-t0,4))

			X1_pred.append(x1_p)
			X2_pred.append(x2_p)

		self.X1_pred = X1_pred
		self.X2_pred = X2_pred

		if self.verbose: print("predictions made")

		return X1_pred, X2_pred


	def score(self,how='corrcoef'):
		"""
		Evalulate the predictions. Calculates the skill down each column
		and averages them together to get the total skill.

		how : how to score the predictions
			-'score'
			-'corrcoef'
		"""

		num_preds = self.X1.shape[1]

		score_1 = []
		score_2 = []

		for x1_p, x2_p in zip(self.X1_pred, self.X2_pred):

			sc1 = np.empty(num_preds)
			sc2 = np.empty(num_preds)

			for ii in range(num_preds):

				p1 = x1_p[:,ii]
				p2 = x2_p[:,ii]

				if self.score_metric == 'score':
					sc1[ii] = ut.score(p1,self.X1[:,ii])
					sc2[ii] = ut.score(p2,self.X2[:,ii])

				if self.score_metric == 'corrcoef':
					sc1[ii] = ut.corrcoef(p1,self.X1[:,ii])
					sc2[ii] = ut.corrcoef(p2,self.X2[:,ii])

			score_1.append( np.mean(sc1) )
			score_2.append( np.mean(sc2) )

		return score_1, score_2


class Embed:

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

	def embed_indices(self,lag,embed):
		"""
		Gets the indices of the embedded time series. This assumes that the
		time series is sequential. Non-sequential time series are currently
		not supported.

		Parameters
		----------
		lag : int
			lag values as calculated from the first minimum of the mutual info.

		embed : int
			embedding dimension, how many lag values to take
		"""

		tsize = self.X.shape[0]

		X = np.arange(0,tsize)

		t_iter = tsize-(lag*(embed-1))

		features = np.zeros((t_iter,embed))

		for ii in range(t_iter):

			end_val = ii+lag*(embed-1)+1

			part = X[ii : end_val]

			features[ii,:] = part[::lag]

		return features


	def embed_vectors_1d(self,lag,embed):
		"""
		Embeds vectors from a one dimensional time series in
		m-dimensional space.

		Parameters
		----------
		lag : int
			lag values as calculated from the first minimum of the mutual info.

		embed : int
			embedding dimension, how many lag values to take


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
