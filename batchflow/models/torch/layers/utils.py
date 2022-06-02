""" Utils for individual layers. """
from math import floor, ceil

from ..utils import to_n_tuple


def compute_padding(padding, shape, kernel_size, dilation, stride, transposed=False):
    """ Compute padding.
    If supplied padding not `same`, just return the same value.

    If `padding` is `same`, then compute amount of padding, taking `stride` into account:
        - if not `transposed` (regular conv or pooling), then compute padding
        required for making output shape equal to `ceil(shape / stride)`.
        - if `transposed` (transposed conv), then compute padding and output padding
        required for making output shape equal to `shape * stride`.

    Under the hood, iterates over possible values of `padding` and `output_padding` parameters,
    until condition (difference between desired size and computed output size) is met.
    """
    if padding == 'valid':
        return {'padding': 0}
    if padding != 'same':
        return {'padding': padding}

    n = len(shape)
    kernel_size = to_n_tuple(kernel_size, n)
    dilation = to_n_tuple(dilation, n)
    stride = to_n_tuple(stride, n)

    result = {'padding': []}
    if transposed:
        result['output_padding'] = []

    for size, k, d, s in zip(shape, kernel_size, dilation, stride):
        if not transposed:
            padding = _compute_same_padding(size=size, kernel_size=k, dilation=d, stride=s)
            result['padding'].append(padding)
        else:
            padding, output_padding = _compute_same_padding_transposed(kernel_size=k, dilation=d, stride=s)
            result['padding'].append(padding)
            result['output_padding'].append(output_padding)
    return result


def _compute_same_padding(size, kernel_size, dilation, stride):
    # Pre-compute some variables
    desired_size = ceil(size / stride)
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    underestimated_size = (size - effective_kernel_size) / stride + 1

    # Search for correct padding value
    for padding in range(0, effective_kernel_size):
        size_difference = desired_size - floor(underestimated_size + 2 * padding / stride)
        if size_difference == 0:
            return padding
    raise ValueError('Never raised')

def _compute_same_padding_transposed(kernel_size, dilation, stride):
    # Pre-compute some variables
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    kernel_residual = effective_kernel_size - stride

    for output_padding in range(stride)[::-1]:
        two_padding = kernel_residual + output_padding
        if two_padding % 2 == 0:
            return two_padding // 2, output_padding
    raise ValueError('Never raised')
