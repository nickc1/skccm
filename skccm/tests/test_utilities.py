import os.path as op
import numpy as np
import numpy.testing as npt
import skccm.utilities as ut



def test_weighted_mean():
    distances = np.array([ [.1,.2,.3,.4],
                     [.1,.2,.3,.4]])

    X = np.array([ [1,2,3,4],
                [1,2,3,4]])

    w_mean = ut.weighted_mean(X,distances)

    #by hand
    arr = np.array([1, 2, 3, 4])
    num = np.array([1/.1, 1/.2, 1/.3, 1/.4])
    denom = np.sum(num)
    W = num/denom
    mean = np.sum(W*np.array([1,2,3,4]))

    #only test to 4 decimal places since the array cant be zero, so the
    # function adds .00001 to the distances to avoid errors.
    np.testing.assert_array_almost_equal(np.array([mean,mean]),w_mean,decimal=4)

def test_mi_digitize():
    # this will result in 4 bins since we are going for a binomial distribution
    x = np.array([ 0. ,  0.6,  1.1,  1.7,  2.2,  2.8,  3.3,  3.9,  4.4,  5. ])
    targ = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 4])

    res = ut.mi_digitize(x)
    npt.assert_equal(targ,res)

def test_weighted_mode():

	x = [4, 1, 4, 2, 4, 2]
	weights = [1, 3, 0.5, 1.5, 1, 2]

	M,C = ut.weighted_mode(x, weights)

	npt.assert_equal(M,np.array([2]))
	npt.assert_equal(C,np.array([3.5]))


def test_quick_mode_axis1():

    X = np.array([[2, 1, 3, 1, 3, 3],
            [2, 0, 3, 1, 2, 2],
            [3, 1, 1, 0, 2, 2],
            [3, 1, 2, 3, 1, 1]],dtype=int)

    M = ut.quick_mode_axis1(X)
    npt.assert_equal(M, np.array([ 3.,  2.,  1.,  1.]))

def test_quick_mode_axis1_keep_nearest_neigh():

    X = np.array([[2, 1, 3, 1, 3, 3],
            [2, 3, 3, 3, 2, 2],
            [3, 1, 1, 0, 2, 2],
            [3, 1, 3, 3, 1, 1]],dtype=int)

    M = ut.quick_mode_axis1_keep_nearest_neigh(X)

    npt.assert_equal(M,np.array([3,2,1,3]))

def test_keep_diversity():

	X = np.array([[2, 1, 3, 1, 3, 3],
				[2, 2, 2, 2, 2, 2],
				[3, 1, 1, 0, 2, 2],
				[1, 1, 1, 1, 1, 1]],dtype=int)

	M = ut.keep_diversity(X)
	npt.assert_equal(M, np.array([ True, False,  True, False], dtype=bool) )
