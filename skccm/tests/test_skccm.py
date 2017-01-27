import os.path as op
import numpy as np
import numpy.testing as npt
import skccm as ccm

def test_regression_dist_calc():

    X = np.array([
        [ 0.3,  0.6],
        [ 0.2,  1.4],
        [ 1.2,  0.2]])
    y = X.sum(axis=1,keepdims=True)

    R = edm.Regression()
    R.fit(X,y)
    R.dist_calc(X)

    d = np.array([[ 0., 0.80622577, 0.98488578],
        [ 0., 0.80622577, 1.56204994],
        [ 0., 0.98488578, 1.56204994]])

    i = np.array([[0, 1, 2],
            [1, 0, 2],
            [2, 0, 1]])

    npt.assert_almost_equal(R.dist, d)
    npt.assert_equal(R.ind, i)

def test_uniform_regression():
    """
    Tests a full regression using a uniform weighting
    """

    X = np.array([
        [ 0.3,  0.6],
        [ 0.2,  1.4],
        [ 1.2,  0.2]])
    y = X.sum(axis=1,keepdims=True)

    R = edm.Regression()
    R.fit(X,y)
    p = R.predict(X,[1,2,3])

    #p[0] should just return y. The closest neighbor is itself
    npt.assert_array_almost_equal(p[0],y)

    #p[1] will be an average between itself and its closest neighbor
    p1_test = np.empty((3,1))
    p1_test[0] = (0.9 + 1.6)/2
    p1_test[1] = (1.6 + 0.9)/2
    p1_test[2] = (1.4 + 0.9)/2
    npt.assert_array_almost_equal(p1_test,p[1])

    #p[2] should be an average of them all
    p2_test = np.mean(y)
    p2_test = np.array([p2_test]*3).reshape(3,1) #convert it to the same shape
    npt.assert_array_almost_equal(p2_test,p[2])

def test_weighted_regression():
    """
    Tests a regression by weighting the neighbors
    """

    Xtr = np.array([
             [ 0.3,  0.6],
             [ 0.2,  1.4],
             [ 1.2,  0.2]])
    Xte = np.array([
             [ 0.7,  1.6],
             [ 1.3,  0.4],
             ])

    ytr = np.array([[ 0.9],
                    [ 1.6],
                    [ 1.4]])

    yte = np.array([[ 2.3],
                    [ 1.7]])

    R = edm.Regression(weights='distance')
    R.fit(Xtr,ytr)
    p = R.predict(Xte,[1,2,3])

    #calculated distances:
    dist = np.array([[ 0.53851648,  1.07703296,  1.48660687],
                    [ 0.2236068 ,  1.0198039 ,  1.48660687]])
    #near neighbor indices:
    ind = np.array([[1, 0, 2],
                    [2, 0, 1]])

    #p[0] should return the nearest neighbor
    npt.assert_array_almost_equal(p[0], np.array([[1.6],
                                    [1.4]]), decimal=4 )

    #p[1] should return a weighted average of the two nearest
    W = 1/dist[:,0:2]
    W/= np.sum(W,axis=1,keepdims=True)
    w_avg = np.sum(W*ytr[R.ind[:,0:2],0],axis=1).reshape(-1,1)

    npt.assert_array_almost_equal(p[1], w_avg,decimal=4)

    #p[2] should return a weighted average of them all
    W = 1/dist
    W/= np.sum(W,axis=1,keepdims=True) #normalize
    w_avg = np.sum(W*ytr[R.ind,0],axis=1).reshape(-1,1)

    npt.assert_array_almost_equal(p[2], w_avg,decimal=4)

def test_uniform_classification_():

    Xtr = np.array([
                [ 3, 6],
                [ 2, 1],
                [ 1, 3]])
    Xte = np.array([
                [ 3,  5],
                [ 2,  2]])

    ytr = np.array([[9],
                    [3],
                    [9]])

    R = edm.Classification()
    R.fit(Xtr,ytr)
    p = R.predict(Xte,[1,2,3])

    ind = np.array([[0, 1, 2],
                    [1, 0, 2]])

    #p[0] should just be the nearest neighbor
    yp = np.array([[9.],
                [3.]])
    npt.assert_equal(p[0], yp)

    # p[1] should be  the mode of the first two, but how its set up it will
    # always take the first one if there is no clear mean
    yp = np.array([[9.],
                [3.]])
    npt.assert_equal(p[1], yp)

    # p[2] should be the mode of all three,
    yp = np.array([[9.],
                [9.]])
    npt.assert_equal(p[2], yp)

def test_weighted_classification():
    Xtr = np.array([
         [ 3, 5],
         [ 2, 1],
         [ 1, 3]])
    Xte = np.array([
         [ 3,  5],
         [ 2,  2],
         ])
    ytr = np.array([[9],
               [3],
               [9]])

    R = edm.Classification(weights='distance')
    R.fit(Xtr,ytr)
    p = R.predict(Xte,[1,2,3])

    dist = np.array([[ 0. ,  1. ,  1. ],
                    [ 0.5,  1. ,  1. ]])
    W = 1/(dist+.00001) #make sure to not divide by zero

    #p[0] should just be the nearest neighbor
    yp = np.array([[9.],
                [3.]])
    npt.assert_equal(p[0], yp)

    # p[1] should be  the mode of the first two, since the first one is
    # closer, it will be chosen.
    yp = np.array([[9.],
                [3.]])
    npt.assert_equal(p[1], yp)

    # p[2] should be the weighted mode of all three. Due to rounding, W[1,1]
    # and W[1,2] is actually slightly larger than W[1,0]
    yp = np.array([[9.],
                [9.]])
    npt.assert_equal(p[2], yp)
