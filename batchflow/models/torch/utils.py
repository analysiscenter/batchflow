""" Utility functions. """
import numpy as np
import torch



def unpack_fn_from_config(param, config=None):
    """ Return params from config """
    value = config.get(param)

    if value is None:
        return None, {}

    value = value if isinstance(value, list) else [value]
    res = []

    for item in value:
        if isinstance(item, (tuple, list)):
            if len(item) == 0:
                name, args = None, None
            elif len(item) == 1:
                name, args = item[0], {}
            elif len(item) == 2:
                name, args = item
            else:
                name, args = item[0], item[1:]
        elif isinstance(item, dict):
            item = item.copy()
            name, args = item.pop('name', None), item
        else:
            name, args = item, {}
        res.append((name, args))

    res = res[0] if len(res) == 1 else res
    return res


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
    """ !!. """
    if isinstance(inputs, torch.Tensor):
        pass
    elif isinstance(inputs, tuple):
        inputs = torch.ones(*inputs, device=device)
    elif isinstance(inputs, (tuple, list)):
        inputs = [make_initialization_inputs(item, device=device) for item in inputs]
    return inputs


def get_shape(inputs, default_shape=None):
    """ !!. """
    if inputs is None:
        shape = default_shape
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (tuple, list)):
        shape = [get_shape(item) for item in inputs]
    else:
        raise TypeError(f'Inputs can be array, tensor, or sequence, got {type(inputs)} instead!')
    return shape

def get_num_channels(inputs, axis=1):
    """ !!. """
    return get_shape(inputs)[axis]

def get_num_dims(inputs):
    """ !!. """
    shape = get_shape(inputs)
    dim = len(shape)
    return max(1, dim - 2)

def get_device(inputs):
    """ !!. """
    if isinstance(inputs, torch.Tensor):
        device = inputs.device
    elif isinstance(inputs, (tuple, list)):
        device = inputs[0].device
    else:
        raise TypeError(f'Inputs can be a tensor, shape, or sequence of them, got {type(inputs)} instead!')
    return device
