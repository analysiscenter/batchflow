""" Contains utility function for metrics evaluation """
import numpy as np
import warnings

from numba import njit
from scipy.ndimage import measurements


@njit(nogil=True)
def binarize(inputs, threshold=.5):
    """ Create a binary mask from probabilities with a given threshold.

    Parameters
    ----------
    inputs : np.array
        input mask with probabilities
    threshold : float
        where probability is above the threshold, the output mask will have 1,
        otherwise 0.

    Returns
    -------
    np.array
        binary mask of the same shape as the input mask
    """
    return inputs >= threshold


@njit(nogil=True)
def sigmoid(arr):
    return 1. / (1. + np.exp(-arr))


def get_components(inputs, batch=True):
    """ Find connected components """
    coords = []
    num_items = len(inputs) if batch else 1
    for i in range(num_items):
        connected_array, num_components = measurements.label(inputs[i], output=None)
        comps = []
        for j in range(num_components):
            c = np.where(connected_array == (j + 1))
            comps.append(c)
        coords.append(comps)
    return coords if batch else coords[0]

def infmean(arr, axis):
    """ Compute the arithmetic mean along 0 axis ignoring infs, when there is
    at least one finite number along averaging axis. Done via np.nanmean()
    while temporarily replacing np.inf with np.nan.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    arr[np.isinf(arr)] = np.nan
    # Mean of empty slice is expected to be np.nan, so the warning is redundant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        arr = np.nanmean(arr, axis=axis)
    if np.isscalar(arr):
        return np.inf if np.isnan(arr) else arr
    arr[np.isnan(arr)] = np.inf
    return arr