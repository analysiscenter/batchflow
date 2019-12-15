""" Contains blocks for various deep network architectures."""
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
from .utils import get_num_channels, safe_eval



CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']



class DefaultBlock(nn.Module):
    """ Block with default parameters.
    Allows creation of modules with predefined arguments for :class:`~.ConvBlock`.
    """
    LAYOUT = 'cna'
    FILTERS = 'same'

    def __init__(self, **kwargs):
        super().__init__()
        attrs = {name.lower(): value for name, value in vars(type(self)).items()
                 if name.isupper()}
        kwargs = {**attrs, **kwargs}

        self.layer = ConvBlock(**kwargs)

    def forward(self, x):
        return self.layer(x)



class XceptionBlock(DefaultBlock):
    """ Xception building block.
    François Chollet. "`Xception: Deep Learning with Depthwise Separable Convolutions
    <https://arxiv.org/abs/1610.02357>`_"
    """
    LAYOUT = 'R' + 'wnacna'*3 + '&'
    FILTERS = 'same'
    STRIDES = [1]*6
    RESIDUAL_END = {'strides': 1}



class VGGBlock(nn.Module):
    """ Convenient VGG block.

    Parameters
    ----------
    depth3 : int
        Number of 3x3 convolutions.
    depth1 : int
        Number of 1x1 convolutions.
    """
    def __init__(self, inputs=None, layout='cna', filters=None, depth3=1, depth1=0, **kwargs):
        super().__init__()

        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        layout = layout * (depth3 + depth1)
        kernels = [3]*depth3 + [1]*depth1
        self.layer = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=kernels, **kwargs)

    def forward(self, x):
        return self.layer(x)



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
        If int, the first repetition of block will use downsampling with that factor.
        If True, the first repetition of block will use downsampling with a factor of 2.
        If False, then no downsampling. Default is False.
    bottleneck : bool, int
        If True, then add a canonical bottleneck (1x1 conv-batchnorm-activation) with that factor of filters increase.
        If False, then bottleneck is not used. Default is False.
    se : bool
        If True, then add a squeeze-and-excitation block.
        If False, then nothing is added. Default is False.
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

        num_convs = sum(letter in CONV_LETTERS for letter in layout)

        filters = [filters] * num_convs if isinstance(filters, (int, str)) else filters
        filters = [safe_eval(item, get_num_channels(inputs)) if isinstance(item, str) else item
                   for item in filters]

        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        strides = [strides] * num_convs if isinstance(strides, int) else strides
        groups = [groups] * num_convs
        branch_stride = np.prod(strides)

        # Used in the first repetition of the block.
        # Different from strides and branch_stride in other blocks if `downsample` is not ``False``.
        strides_downsample = list(strides)
        branch_stride_downsample = int(branch_stride)

        if downsample:
            downsample = 2 if downsample is True else downsample
            strides_downsample[0] *= downsample
            branch_stride_downsample *= downsample
        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout + 'cna'
            kernel_size = [1] + kernel_size + [1]
            strides = [1] + strides + [1]
            strides_downsample = [1] + strides_downsample + [1]
            groups = [1] + groups + [1]
            filters = [filters[0]] + filters + [filters[0] * bottleneck]
        if se:
            layout += 'S*'
        layout = 'B' + layout + op

        layer_params = [{'strides': strides_downsample, 'branch/strides': branch_stride_downsample}]
        layer_params += [{}]*(n_reps-1)

        self.layer = ConvBlock(*layer_params, inputs=inputs, layout=layout, filters=filters,
                               kernel_size=kernel_size, strides=strides, groups=groups,
                               branch={'layout': 'c', 'filters': filters[-1], 'strides': branch_stride},
                               **kwargs)

    def forward(self, x):
        return self.layer(x)



class DenseBlock(nn.Module):
    """ DenseBlock module.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation.
        See more in :class:`~.layers.conv_block.BaseConvBlock` documentation. Default is 'nacd'.
    num_layers : int
        Number of consecutive layers to make. Each layer is made upon all of the previous tensors concatted together.
    growth_rate : int
        Amount of filters added after each layer.
    skip : bool
        Whether to concatenate inputs to the output result.
    bottleneck : bool, int
        If True, then add a canonical bottleneck (1x1 conv-batchnorm-activation) with that factor of filters increase.
        If False, then bottleneck is not used. Default is False.
    filters : int or None
        If int and is bigger than number of channels in the input tensor, then `growth_rate` is adjusted so
        that the number of output features is that number.
        If int and is smaller or equal to the number of channels in the input tensor, then `growth_rate` is adjusted so
        that the number of added output features is that number.
        If None, then not used.
    """
    def __init__(self, inputs=None, layout='nacd', filters=None, kernel_size=3, strides=1, dropout_rate=0.2,
                 num_layers=4, growth_rate=12, skip=True, bottleneck=False, **kwargs):
        super().__init__()
        self.skip = skip
        self.input_num_channels = get_num_channels(inputs)

        if filters is not None:
            if isinstance(filters, str):
                filters = safe_eval(filters, get_num_channels(inputs))

            if filters > self.input_num_channels:
                growth_rate = (filters - self.input_num_channels) // num_layers
            else:
                growth_rate = filters // num_layers
        filters = growth_rate

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout
            kernel_size = [1, kernel_size]
            strides = [1, strides]
            filters = [growth_rate * bottleneck, filters]

        layout = 'R' + layout + '.'
        self.layer = ConvBlock(layout=layout, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate,
                               filters=filters, n_repeats=num_layers, inputs=inputs, **kwargs)

    def forward(self, x):
        x = self.layer(x)
        return x if self.skip else x[:, self.input_num_channels:]
