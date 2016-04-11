"""
Data for CCM
By Nick Cortale

Thanks to Kenneth Ells and Dylan McNamara
"""


import numpy as np
from numpy import genfromtxt
from scipy import integrate


def coupled_logistic(rx1, rx2, b12, b21, ts_length,random_seed=None):
    """
    Coupled logistic map
    
    Parameters
    ----------
    rx1 : float
        parameter that determines chaotic behavior of the x1 series
    rx2 : float
        parameter that determines chatotic behavior of the x2 series
    b12 : float
        Influence of x1 on x2
    b21 : float
        Influence of x2 on x1
    length : int 
        Length of the calculated time series

    Returns
    -------
    x1 : array
        array of length (ts_length,) that stores the values of the x series
    x2 : array
        array of length (ts_length,) that stores the values of the y series
    """

    # Initial conditions after McCracken (2014)
    x1 = np.zeros(ts_length)
    x2 = np.zeros(ts_length)

    if random_seed:
        x1[0] = .15 + .1*np.random.rand()
        x2[0] = .35 + .1 *np.random.rand()
    else:
        x1[0] = 0.2
        x2[0] = 0.4

    for i in range(ts_length-1):
        x1[i+1] = x1[i] * (rx1 - rx1 * x1[i] - b21 * x2[i])
        x2[i+1] = x2[i] * (rx2 - rx2 * x2[i] - b12 * x1[i])
        
    return x1,x2

def driving_sin():

    rx = 3.7
    ry = 3.7
    bxy = 0.1
    byx = 0.2

    iterations=250
    x = np.zeros(iterations)
    y = np.zeros(iterations)

    x[0] = 0.2
    y[0] = 0.4

    for i in range(iterations-1):
        #x[i+1] = x[i] * (rx - rx * x[i] - bxy * y[i])
        x[i+1] = np.sin(i)*.5
        y[i+1] = y[i] * (ry - ry * y[i] - byx * x[i])
    return x,y












