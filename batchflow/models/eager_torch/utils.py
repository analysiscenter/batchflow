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
        if isinstance(item, tuple):
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


def get_shape(inputs, shape=None):
    """ Return inputs shape """
    if inputs is None:
        pass
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        shape = tuple(inputs)
    else:
        raise TypeError('inputs can be array, tensor, tuple/list or layer', type(inputs))
    return shape

def get_num_channels(inputs, axis=1):
    """ Return a number of channels """
    return get_shape(inputs)[axis]

def get_num_dims(inputs):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    shape = get_shape(inputs)
    dim = len(shape)
    return max(1, dim - 2)


def calc_padding(inputs, padding=0, kernel_size=None, dilation=1, transposed=False, stride=1, **kwargs):
    """ Get padding values for various convolutions. """
    _ = kwargs

    dims = get_num_dims(inputs)
    shape = get_shape(inputs)

    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            if transposed:
                padding = 0
            else:
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size,) * dims
                if isinstance(dilation, int):
                    dilation = (dilation,) * dims
                if isinstance(stride, int):
                    stride = (stride,) * dims
                padding = tuple(_get_padding(kernel_size[i], shape[i+2], dilation[i], stride[i]) for i in range(dims))
        else:
            raise ValueError("padding can be 'same' or 'valid'")
    elif isinstance(padding, int):
        pass
    elif isinstance(padding, tuple):
        pass
    else:
        raise ValueError("padding can be 'same' or 'valid' or int or tuple of int")
    return padding

def _get_padding(kernel_size=None, width=None, dilation=1, stride=1):
    kernel_size = dilation * (kernel_size - 1) + 1
    if stride >= width:
        p = max(0, kernel_size - width)
    else:
        if width % stride == 0:
            p = kernel_size - stride
        else:
            p = kernel_size - width % stride
    p = (p // 2, p - p // 2)
    return p
