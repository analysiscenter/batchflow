""" Contains utility function for metrics evaluation """
from numba import njit, prange, jit
import numpy as np
from scipy.ndimage import measurements
from time import time


@njit(nogil=True)
def binarize(inputs, threshold=.5):
    """ Create a binary mask from probabilities with a given threshold.

    Parameters
    ----------
    inputs : np.array
        input mask with probabilities
    threshold : float
        where probability is above the threshold, the output mask will have 1, otherwise 0.

    Returns
    -------
    np.array
        binary mask of the same shape as the input mask
    """
    return inputs >= threshold


@njit(nogil=True)
def sigmoid(arr):
    return 1. / (1. + np.exp(-arr))


@njit(nogil=True, parallel=True)
def _get_components(connected_array, num_components):
    components = np.zeros((num_components, connected_array.ndim, 2), dtype=np.int32)
    for i in prange(num_components):
        coords = np.where(connected_array == (i + 1))
        for d in range(components.shape[1]):
            components[i, d, 0] = np.min(coords[d])
            components[i, d, 1] = np.max(coords[d]) + 1
    return components

def get_components(inputs, batch=True):
    coords = []
    num_items = len(inputs) if batch else 1
    for i in range(num_items):
        connected_array, num_components = measurements.label(inputs[i], output=None)
        c = _get_components(connected_array, num_components)
        coords.append(c)
    return coords if batch else coords[0]
