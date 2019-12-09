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


def get_shape(inputs, default_shape=None):
    """ Return inputs shape """
    if inputs is None:
        shape = default_shape
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        shape = tuple(inputs)
    else:
        raise TypeError('inputs can be array, tensor, or sequence', type(inputs))
    return shape

def get_num_channels(inputs, axis=1):
    """ Return a number of channels """
    return get_shape(inputs)[axis]

def get_num_dims(inputs):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    shape = get_shape(inputs)
    dim = len(shape)
    return max(1, dim - 2)


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

    Increase number of filters of tensor by the factor of two::
    new_filters = safe_eval('same * 2', old_filters)
    """
    #pylint: disable=eval-used
    names = names or ['S', 'same']
    return eval(expression, {}, {name: value for name in names})


def calc_padding(inputs, padding=0, kernel_size=None, dilation=1, transposed=False, stride=1, **kwargs):
    """ Get padding values for various convolutions. """
    _ = kwargs

    dims = get_num_dims(inputs)
    shape = get_shape(inputs)

    if isinstance(padding, str):
        if padding == 'valid':
            result = 0
        elif padding == 'same':
            if transposed:
                result = 0
            else:
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size,) * dims
                if isinstance(dilation, int):
                    dilation = (dilation,) * dims
                if isinstance(stride, (int, np.int64)):
                    stride = (stride,) * dims
                result = tuple(_get_padding(kernel_size[i], shape[i+2], dilation[i], stride[i]) for i in range(dims))
        else:
            raise ValueError("padding can be 'same' or 'valid'")
    elif isinstance(padding, (int, tuple)):
        result = padding
    else:
        raise ValueError("padding can be 'same' or 'valid' or int or tuple of int")
    return result

def _get_padding(kernel_size=None, input_shape=None, dilation=1, stride=1):
    kernel_size = dilation * (kernel_size - 1) + 1
    if stride >= input_shape:
        padding = max(0, kernel_size - input_shape)
    else:
        if input_shape % stride == 0:
            padding = kernel_size - stride
        else:
            padding = kernel_size - input_shape % stride
    padding = (padding // 2, padding - padding // 2)
    return padding
