""" Contains utility function for metrics evaluation """
import numpy as np
import numpy.ma as ma
try:
    from scipy.ndimage import measurements
except ImportError:
    pass
try:
    from numba import njit
except ImportError:
    from ...decorators import njit


@njit(nogil=True, parallel=True)
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


@njit(nogil=True, parallel=True)
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
    """ Compute the arithmetic mean along given axis ignoring infs,
    when there is at least one finite number along averaging axis.
    """
    masked = ma.masked_invalid(arr)
    masked = masked.mean(axis=axis)
    if np.isscalar(masked):
        return masked
    if isinstance(masked, ma.core.MaskedConstant):
        return np.inf
    masked[masked.mask] = np.inf
    return masked.data
