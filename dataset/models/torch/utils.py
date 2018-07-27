""" Auxiliary functions for Torch models """
import numpy as np
import torch


def get_num_channels(inputs, axis=1):
    """ Return a number of channels """
    return get_shape(inputs)[axis]

def get_num_dims(inputs, axis=1):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    if isinstance(inputs, np.ndarray):
        dim = inputs.ndim
    elif isinstance(inputs, torch.Tensor):
        dim = inputs.dim()
    elif isinstance(inputs, (torch.Size, tuple, list)):
        dim = len(inputs)
    else:
        raise TypeError('inputs can be array, tensor or tuple/list', inputs)
    return max(1, dim - 2)

def get_shape(inputs):
    """ Return inputs shape """
    if isinstance(inputs, np.ndarray):
        return inputs.shape
    elif isinstance(inputs, torch.Tensor):
        return tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        return tuple(inputs)
    raise TypeError('inputs can be array, tensor or tuple/list', inputs)

def get_output_shape(layer, shape=None):
    if hasattr(layer, 'output_shape'):
        return tuple(layer.output_shape)
    return shape
