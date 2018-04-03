""" File contains function witch generate data to regressions """
import numpy as np

NUM_DIM_LIN = 13
def generate_linear_data(size, dist='unif'):
    """ Generation of data for fit linear regression.

    Parameters
    ----------
    size: int
    Length of data

    dist: {'unif', 'norm'}
    Sample distribution 'unif' or 'norm'. Default 'unif'

    Returns:
    ----------
    x: numpy array
    Uniformly or normally distributed array

    y: numpy array
    array with some random noize """
    if dist == 'unif':
        x = np.random.uniform(0, 2, size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)

    elif dist == 'norm':
        x = np.random.normal(size=size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)

    w = np.random.normal(loc=1., size=[NUM_DIM_LIN])
    error = np.random.normal(loc=0., scale=0.1, size=size)

    xmulw = np.dot(x, w)
    y_obs = xmulw + error

    return x, y_obs.reshape(-1, 1)

def generate_logistic_data(size, first_params, second_params):
    """ Generation of data for fit logistic regression.
    Parameters
    ----------
    size: int
    Length of data

    first_params: list
    List of lists with params of distribution to create first cloud of points

    second_params: list
    List of lists with params of distribution to create second cloud of points

    Returns:
    ----------
    x: numpy array
    Coordinates of points in two-dimensional space

    y: numpy array
    Labels of dots """
    first = np.random.multivariate_normal(first_params[0], first_params[1], size)
    second = np.random.multivariate_normal(second_params[0], second_params[1], size)

    x = np.vstack((first, second))
    y = np.hstack((np.zeros(size), np.ones(size)))
    shuffle = np.arange(len(x))
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]

    return x, y

def generate_poisson_data(lam, size=10):
    """ Generation of data for fit poisson regression
    Parameters
    ----------
    size: int
    Length of data

    lam:
    Poisson distribution parameter

        Returns:
    ----------
    x: numpy array
    Matrix with random numbers of uniform distribution
    y:
    Array of poisson distribution numbers """
    x = np.random.random(size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)
    b = np.random.random(1)

    y_obs = np.random.poisson(np.exp(np.dot(x, lam) + b))

    shuffle = np.arange(len(x))
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y_obs[shuffle]

    return x, y.reshape(-1, 1)
