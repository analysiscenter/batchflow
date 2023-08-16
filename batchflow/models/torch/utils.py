""" Utility functions. """
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def to_n_tuple(value, n):
    return value if isinstance(value, (tuple, list)) else (value,) * n


def safe_eval(expression, value, names=None):
    """ Safely evaluates expression given value and names.
    Supposed to be used to parse string parameters and allow dependencies between parameters (e.g. number of channels)
    in subsequent layers.

    Parameters
    ----------
    expression : str
        Valid Python expression. Each element of the `names` will be swapped with `value`.
    value : object
        Value to use instead of elements of `names`.
    names : sequence of str
        Names inside `expression` to be interpreted as `value`. Default names are `same`, `S`.

    Examples
    --------
    Add 5 to the value::
    safe_eval('same + 5', 10)

    Increase number of channels of tensor by the factor of two::
    new_channels = safe_eval('same * 2', old_channels)
    """
    #pylint: disable=eval-used
    names = names or ['S', 'same']
    return eval(expression, {}, {name: value for name in names})


def make_initialization_inputs(inputs, device=None):
    """ Take either tensor, shape tuple or list of them, and always return tensor or list of them. """
    if isinstance(inputs, torch.Tensor):
        pass
    elif isinstance(inputs, tuple):
        inputs = torch.rand(*inputs, device=device)
    elif isinstance(inputs, list):
        inputs = [make_initialization_inputs(item, device=device) for item in inputs]
    return inputs


def get_shape(inputs, default_shape=None):
    """ Compute shape of a tensor or shapes of list of tensors. """
    if inputs is None:
        shape = default_shape
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (tuple, torch.Size)):
        shape = inputs
    elif isinstance(inputs, list):
        shape = [get_shape(item) for item in inputs]
    else:
        raise TypeError(f'Inputs can be array, tensor, or sequence, got {type(inputs)} instead!')
    return shape

def get_num_channels(inputs):
    """ Get number of channels in one tensor. """
    return inputs.shape[1]

def get_num_dims(inputs):
    """ Get number of dimensions in one tensor, minus the batch and channel dimension. """
    dim = len(inputs.shape)
    return max(1, dim - 2)

def get_device(inputs):
    """ Get used device of a tensor or list of tensors. """
    if isinstance(inputs, torch.Tensor):
        device = inputs.device
    elif isinstance(inputs, (tuple, list)):
        device = inputs[0].device
    else:
        raise TypeError(f'Inputs can be a tensor or list of tensors, got {type(inputs)} instead!')
    return device

def make_shallow_dict(module):
    """ Create a dictionary from a module, where:
        - keys are valid attribute names for each children
        - values are submodules themselves, directly contained in torch.nn, e.g. nn.Conv2d, nn.Linear
    """
    #pylint: disable=protected-access
    if module.__class__ is getattr(nn, module.__class__.__name__, None):
        return {None : module}

    result = {}
    for key, value in module._modules.items():
        subdict = make_shallow_dict(value)
        for subkey, subvalue in subdict.items():
            store_key = f'{key}/{subkey}' if subkey is not None else key
            result[store_key] = subvalue
    return result

def pad(inputs, spatial_shape, target_shape):
    """ Pad spatial dimensions of a tensor to a target shape. """
    pad_values = [0] * len(spatial_shape) * 2
    pad_dims = np.nonzero(np.array(spatial_shape) < np.array(target_shape))[0]
    for dim in pad_dims:
        pad_values[dim * 2] = (target_shape[dim] - spatial_shape[dim]) // 2
        pad_values[dim * 2 + 1] = (target_shape[dim] - spatial_shape[dim] + 1) // 2
    padded_inputs = F.pad(inputs, pad_values[::-1])
    return padded_inputs

def center_crop(inputs, target_shape, dims):
    """ Crop last dims of a tensor at the center. """
    inputs_shape = np.array(get_shape(inputs))
    to_crop_shape = inputs_shape[-dims:]
    no_crop_shape = inputs_shape[:-dims]
    crop_lefts = (to_crop_shape - target_shape) // 2
    crops_ = [slice(crop_lefts[i], crop_lefts[i] + target_shape[i]) for i in range(dims)]
    crops = [slice(dim) for dim in no_crop_shape] + crops_
    return inputs[crops]
