#
# Data for analyzing causality.
# By Nick Cortale
#
# Paper:
# Detecting Causality in Complex Ecosystems
# George Sugihara et al. 2012
#
# Thanks to Kenneth Ells and Dylan McNamara
#

import numpy as np
from numpy import genfromtxt
from scipy import integrate


def coupled_logistic(rx1, rx2, b12, b21, ts_length,random_start=False):
    """Coupled logistic map.

    Parameters
    ----------
    rx1 : float
        Parameter that determines chaotic behavior of the x1 series.
    rx2 : float
        Parameter that determines chatotic behavior of the x2 series.
    b12 : float
        Influence of x1 on x2.
    b21 : float
        Influence of x2 on x1.
    ts_length : int
        Length of the calculated time series.
    random_start : bool
        Random initialization of starting conditions.

    Returns
    -------
    x1 : 1d array
        Array of length (ts_length,) that stores the values of the x series.
    x2 : 1d array
        Array of length (ts_length,) that stores the values of the y series.

    """

    # Initial conditions after McCracken (2014)
    x1 = np.zeros(ts_length)
    x2 = np.zeros(ts_length)

    if random_start:
        x1[0] = .15 + .1*np.random.rand()
        x2[0] = .35 + .1 *np.random.rand()
    else:
        x1[0] = 0.2
        x2[0] = 0.4

    for i in range(ts_length-1):

        x1[i+1] = x1[i] * (rx1 - rx1 * x1[i] - b21 * x2[i])
        x2[i+1] = x2[i] * (rx2 - rx2 * x2[i] - b12 * x1[i])

    return x1,x2


def driven_rand_logistic(rx2, b12, ts_length,random_start=False):
    """Logistic map with random forcing. x1 is the random array and x2 is the
    logistic map.

    Parameters
    ----------
    rx2 : float
        Parameter that determines chatotic behavior of the x2 series.
    b12 : float
        Influence of x1 on x2.
    ts_length : int
        Length of the calculated time series.
    random_start : Boolean
        Random initialization of starting conditions.

    Returns
    -------
    x1 : array
        Array of length (ts_length,)
    x2 : array
        Array of length (ts_length,)
    """

    x1 = np.random.rand(ts_length)*.4
    x2 = np.zeros(ts_length)

    if random_start:

        x2[0] = .35 + .1 *np.random.rand()
    else:

        x2[0] = 0.4

    for i in range(ts_length-1):

        x2[i+1] = x2[i] * (rx2 - rx2 * x2[i] - b12 * x1[i])

    return x1,x2



def driving_sin(rx2, b12, ts_length, random_start=False):
    """Sine wave driving a logistic map.

    Parameters
    ----------
    rx2 : float
        Parameter that determines chatotic behavior of the x2 series.
    b12 : float
        Influence of x1 on x2.
    ts_length : int
        Length of the calculated time series.
    random_start : Boolean
        Random initialization of starting conditions.

    Returns
    -------
    x1 : array
        Array of length (ts_length,) that stores the values of the x series.
    x2 : array
        Array of length (ts_length,) that stores the values of the y series.
    """

    x1 = np.sin(np.linspace(0,100*np.pi,ts_length))*.4
    x2 = np.zeros(ts_length)

    if random_start:
        x2[0] = .35 + .1 *np.random.rand()

    else:
        x2[0] = 0.4

    for i in range(ts_length-1):
        x2[i+1] = x2[i] * (rx2 - rx2 * x2[i] - b12 * x1[i])

    return x1,x2

def lagged_coupled_logistic(rx1, rx2, b12, b21, ts_length, random_start=False):
    """Coupled logistic map. x1 is driven by random lags of x2.

    Parameters
    ----------
    rx1 : float
        Parameter that determines chaotic behavior of the x1 series.
    rx2 : float
        Parameter that determines chatotic behavior of the x2 series.
    b12 : float
        Influence of x1 on x2.
    b21 : float
        Influence of x2 on x1.
    ts_length : int
        Length of the calculated time series.
    random_start : Boolean
        Random initialization of starting conditions.

    Returns
    -------
    x1 : array
        Array of length (ts_length,) that stores the values of the x series.
    x2 : array
        Array of length (ts_length,) that stores the values of the y series.
    """

    # Initial conditions after McCracken (2014)
    x1 = np.zeros(ts_length)
    x2 = np.zeros(ts_length)

    if random_start:
        x1[0] = .15 + .1*np.random.rand()
        x2[0] = .35 + .1 *np.random.rand()
    else:
        x1[0] = 0.2
        x2[0] = 0.4

    for i in range(ts_length-1):

        try:
            randi = np.random.randint(1,10)
            x1[i+1] = x1[i] * (rx1 - rx1 * x1[i] - b21 * x2[i-randi])
        except:
            x1[i+1] = x1[i] * (rx1 - rx1 * x1[i] - b21 * x2[i])

        x2[i+1] = x2[i] * (rx2 - rx2 * x2[i] - b12 * x1[i])

    return x1,x2

def lorenz(sz=10000, noise=0, max_t=100.):
    """Integrates the lorenz equation.

    Parameters
    ----------
    sz : int
        Length of the time series to be integrated.
    noise : float
        Amplitude of noise to be added to the lorenz equation.
    max_t : float
        Length of time to solve the lorenz equation over.

    Returns
    -------
    X : 2D array
        Solutions to the Lorenz equations. Columns are X,Y,Z.
    """

    def lorenz_deriv(xyz, t0, sigma=10., beta=8./3, rho=28.0):
        x,y,z = xyz
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, max_t, sz)  # one thousand time steps
    X = integrate.odeint(lorenz_deriv, x0, t) + noise*np.random.rand(sz,3)

    return X
