""" Contains blocks for various deep network architectures."""
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
from .utils import get_num_channels, safe_eval

CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']

class DenseBlock(nn.Module):
    """ DenseBlock module. """
    def __init__(self, inputs=None, layout='nacd', filters=None, kernel_size=3, strides=1, dropout_rate=0.2,
                 num_layers=4, growth_rate=12, skip=True, bottleneck=False, **kwargs):
        super().__init__()
        self.skip = skip
        self.input_num_channels = get_num_channels(inputs)

        if filters is not None:
            if isinstance(filters, str):
                filters = eval(filters, {}, {key: get_num_channels(inputs) for key in ['S', 'same']})
            growth_rate = (filters - self.input_num_channels) // num_layers
        filters = growth_rate

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout
            kernel_size = [1, kernel_size]
            strides = [1, strides]
            filters = [growth_rate * bottleneck, filters]

        layout = 'R' + layout + '.'
        self.block = ConvBlock(layout=layout, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate,
                               filters=filters, n_repeats=num_layers, inputs=inputs, **kwargs)

    def forward(self, x):
        output = self.block(x)
        return output if self.skip else output[:, self.input_num_channels:]

class ResBlock(nn.Module):
    """ ResNet Module: pass tensor through one or multiple (`n_reps`) blocks, each of which is a
    configurable residual layer, potentially including downsampling, bottleneck, squeeze-and-excitation and groups.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation.
        See more in :class:`~.layers.conv_block.BaseConvBlock` documentation. Default is 'cna cna'.
    filters : int, str, list of int, list of str
        If `str`, then number of filters is calculated by its evaluation. ``'S'`` and ``'same'`` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor. Default value is 'same'.
    kernel_size : int, list of int
        Convolution kernel size. Default is 3.
    strides : int, list of int
        Convolution stride. Default is 1.
    downsample : int, bool
        If int, in first repetition of block downsampling with a factor `downsample`.
        If True, in first repetition of block downsampling with a factor 2.
        If False, without downsampling. Default is False.
    bottleneck : bool, int
        If True, then construct a canonical bottleneck block from the given layout.
        If False, then bottleneck is not used. Default is False.
    se : bool
        If True, then construct a SE-ResNet block from the given layout.
        If False, then squeeze and excitation is not used. Default is False.
    groups : int
        Use `groups` convolution side by side, each  seeing 1 / `groups` the input channels,
        and producing 1 / `groups` the output channels, and both subsequently concatenated.
        Number of `inputs` channels must be divisible by `groups`. Default is 1.
    op : str or callable
        Operation for combination shortcut and residual.
        See more :class:`~.layers.Combine` documentation. Default is '+'.
    n_reps : int
        Number of times to repeat the whole block. Default is 1.
    kwargs : dict
        Other named arguments for the :class:`~.layers.ConvBlock`
    """
    def __init__(self, inputs=None, layout='cnacna', filters='same', kernel_size=3, strides=1,
                 downsample=False, bottleneck=False, se=False, groups=1, op='+', n_reps=1, **kwargs):
        super().__init__()

        num_convs = sum([letter in CONV_LETTERS for letter in layout])

        filters = [filters] * num_convs if isinstance(filters, (int, str)) else filters
        filters = [safe_eval(str(item), get_num_channels(inputs)) for item in filters]

        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        strides = [strides] * num_convs if isinstance(strides, int) else strides
        strides_d = list(strides)
        groups = [groups] * num_convs
        side_branch_stride = np.prod(strides)
        side_branch_stride_d = int(side_branch_stride)

        if downsample:
            downsample = 2 if downsample is True else downsample
            strides_d[0] *= downsample
            side_branch_stride_d *= downsample
        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout + 'cna'
            kernel_size = [1] + kernel_size + [1]
            strides = [1] + strides + [1]
            strides_d = [1] + strides_d + [1]
            groups = [1] + groups + [1]
            filters = [filters[0]] + filters + [filters[0] * bottleneck]
        if se:
            layout += 'S*'
        layout = 'B' + layout

        layer_params = [{'strides': strides_d, 'side_branch/strides': side_branch_stride_d}] + [{}]*(n_reps-1)
        self.block = ConvBlock(*layer_params, inputs=inputs, layout=layout, filters=filters,
                               kernel_size=kernel_size, strides=strides, groups=groups,
                               side_branch={'layout': 'c', 'filters': filters[-1], 'strides': side_branch_stride},
                               op=op, **kwargs)

    def forward(self, x):
        return self.block(x)
