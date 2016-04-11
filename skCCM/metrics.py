"""
Metrics for scoring predictions from CCM
"""

import numpy as np
from scipy import stats as stats

def corrCoef(preds,actual):
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










