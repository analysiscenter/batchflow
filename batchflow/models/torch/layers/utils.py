""" !!. """
import numpy as np

from ..utils import get_shape, get_num_dims


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

                if all(item == 0 for item in sum(result, ())):
                    result = 0
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
