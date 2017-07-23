import os.path as op
import numpy as np
import numpy.testing as npt
import skccm.utilities as ut



def test_exp_weight():

    #ensure it sums to one
    X = np.array([ [0.1,0.2,.3,.4],
                 [.3,.3,.7,.7]])

    W = ut.exp_weight(distances)

    np.testing.assert_array_almost_equal(np.array([1.,1.]),W.sum(axis=1))
