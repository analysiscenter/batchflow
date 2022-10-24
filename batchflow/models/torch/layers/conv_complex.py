""" Complex layers, based on convolutions. """
from math import ceil, sqrt, prod

import torch
from torch import nn

from .conv import Conv
from .combine import Combine
from .utils import compute_padding
from ..utils import get_num_channels, get_num_dims, safe_eval, to_n_tuple




class MultiKernelConv(nn.ModuleList):
    """ Multi-kernel convolution: apply convolutions with different kernel sizes
    to the same inputs and concatenate their outputs. Somewhat similar to Inception blocks.

    Parameters
    ----------
    kernel_size : sequence of ints
        Kernel sizes to include.
    channels : int
        Number of channels in the layer output.
    channels_ratio : sequence of floats, optional
        If not provided, then `channels` are split evenly for each `kernel_size`.
        If provided, then ratio of `channels` for each corresponding `kernel_size`.
        Number of channels for the first convolution is adjusted so that total `channels` is the same, as required.
    """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    def __init__(self, inputs=None, channels=None, channels_ratio=None, kernel_size=(3, 5, 7),
                 stride=1, dilation=1, groups=1, bias=False):
        super().__init__()

        # Parse inputs
        in_channels = get_num_channels(inputs)
        constructor = self.LAYERS[get_num_dims(inputs)]

        # Parse channels
        if isinstance(channels, str):
            channels = safe_eval(channels, in_channels)

        if isinstance(channels, int):
            if channels_ratio is None:
                channels_ = [channels // len(kernel_size)] * len(kernel_size)
            else:
                channels_ = [max(int(channels * ratio), 1) for ratio in channels_ratio]

            channels_[0] = channels - sum(channels_[1:])
            channels = channels_

        # Parse other parameters
        n = len(kernel_size)
        dilation = to_n_tuple(dilation, n)
        groups = to_n_tuple(groups, n)

        # Create layers
        for channels_, kernel_size_, groups_, dilation_ in zip(channels, kernel_size, groups, dilation):
            padding_ = compute_padding(padding='same', shape=inputs.shape[2:], kernel_size=kernel_size_,
                                       dilation=dilation_, stride=stride)['padding']

            layer = constructor(in_channels=in_channels, out_channels=channels_, kernel_size=kernel_size_,
                                padding=padding_, stride=stride, groups=groups_, dilation=dilation_, bias=bias)
            self.append(layer)


    def forward(self, x):
        tensors = [layer(x) for layer in self]
        x = torch.cat(tensors, dim=1)
        return x


class SharedKernelConv(nn.Module):
    """ Kernel-sharing convolution: apply the same convolution weight and bias with different dilations
    to the same inputs and concatenate their outputs.

    Parameters
    ----------
    dilation : sequence of ints
        Dilation levels to apply convolutions on.
    channels : int
        Number of channels in the output.
        Each dilation level takes roughly `channels / num_dilations` channels, rounded up.
        If the number of channels is not divisible by the number of dilations, the output has more channels.
    """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }
    FUNCTIONS = {
        1: nn.functional.conv1d,
        2: nn.functional.conv2d,
        3: nn.functional.conv3d,
    }

    def __init__(self, inputs=None, channels=None,
                 kernel_size=3, stride=1, dilation=(1, 3, 6), groups=1, bias=False):
        super().__init__()

        # Parse inputs
        in_channels = get_num_channels(inputs)
        constructor = self.LAYERS[get_num_dims(inputs)]
        self.conv_function = self.FUNCTIONS[get_num_dims(inputs)]

        # Parse parameters
        self.dilation = dilation
        channels = ceil(channels / len(dilation))

        self.layer = constructor(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size,
                                 dilation=1, stride=stride, groups=groups, bias=bias)

    def forward(self, x):
        tensors = []
        for dilation in self.dilation:
            tensor = self.conv_function(x, weight=self.layer.weight, bias=self.layer.bias,
                                        stride=self.layer.stride, dilation=dilation, padding=dilation)
            tensors.append(tensor)

        x = torch.cat(tensors, dim=1)
        return x

    def extra_repr(self):
        return f'dilation={self.dilation}'



class AvgPoolConvInit:
    """ Common mixin for convolutions, initialized with average pooling kernels. """
    def reset_parameters(self):
        """Reset the weight and bias."""
        #pylint: disable=protected-access
        nn.init.constant_(self.weight, 0)
        denominator = prod(self.weight.shape[2:])

        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[j, i] = 1 / denominator

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        # Initialize the rest of the filters, if `out_channels` > `in_channels`
        if self.out_channels > self.in_channels:
            nn.init.kaiming_uniform_(self.weight[self.in_channels:], a=sqrt(5))

            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[self.in_channels:])
                bound = 1 / sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)


class AvgPoolConv1d(AvgPoolConvInit, nn.Conv1d):
    N_DIMS = 1

class AvgPoolConv2d(AvgPoolConvInit, nn.Conv2d):
    N_DIMS = 2

class AvgPoolConv3d(AvgPoolConvInit, nn.Conv3d):
    N_DIMS = 3


class AvgPoolConv(Conv):
    """ Convolution, initialized to average pooling. """
    LAYERS = {
        1: AvgPoolConv1d,
        2: AvgPoolConv2d,
        3: AvgPoolConv3d,
    }

    def __init__(self, inputs=None, channels=None, factor=2, kernel_size=None, stride=None, dilation=1, groups=1,
                 padding='same', bias=False):
        kernel_size = kernel_size if kernel_size is not None else 2 * factor - 1
        stride = stride if stride is not None else factor

        super().__init__(inputs=inputs, channels=channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         groups=groups, padding=padding, bias=bias)



class BilinearConvTransposeInit:
    """ Common mixin for convolutions, initialized with bilinear upsampling kernels. """
    def reset_parameters(self):
        """ Set the weight of the first filters to be identical to bilinear upsampling operation. """
        #pylint: disable=protected-access
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size, self.stride, self.N_DIMS)
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

        # Initialize the rest of the filters, if `out_channels` > `in_channels`
        if self.out_channels > self.in_channels:
            nn.init.kaiming_uniform_(self.weight[:, self.in_channels:], a=sqrt(5))

            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[:, self.in_channels:])
                bound = 1 / sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def bilinear_kernel(kernel_size, stride, n_dims):
        """ Make a bilinear upsampling kernel. """
        kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size,) * n_dims
        stride = stride if isinstance(stride, (tuple, list)) else (stride,) * n_dims

        # The bilinear kernel is separable in its spatial dimensions
        bilinear_kernel = torch.ones(*(1,) * n_dims)
        for channel in range(n_dims):
            channel_kernel_size = kernel_size[channel]
            channel_stride = stride[channel]
            left  = channel_kernel_size // 2
            right = channel_kernel_size - left

            if channel_kernel_size != channel_stride:
                denominator = channel_kernel_size - channel_stride + 1
            else:
                denominator = channel_kernel_size

            # e.g. with stride=4, kernel_size=7
            # delta = [-3, -2, -1, 0, 1, 2, 3]
            # channel_filter = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
            delta = torch.arange(-left, right)
            channel_filter = (1 - torch.abs(delta / denominator))

            # Apply the channel filter to the current channel
            shape = [1] * n_dims
            shape[channel] = channel_kernel_size
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        return bilinear_kernel

class BilinearConvTranspose1d(BilinearConvTransposeInit, nn.ConvTranspose1d):
    N_DIMS = 1

class BilinearConvTranspose2d(BilinearConvTransposeInit, nn.ConvTranspose2d):
    N_DIMS = 2

class BilinearConvTranspose3d(BilinearConvTransposeInit, nn.ConvTranspose3d):
    N_DIMS = 3


class BilinearConvTranspose(nn.Module):
    """ Transposed convolution, initialized to bilinear upsampling."""
    LAYERS = {
        1: BilinearConvTranspose1d,
        2: BilinearConvTranspose2d,
        3: BilinearConvTranspose3d,
    }

    def __init__(self, inputs=None, channels=None, factor=2, kernel_size=None, stride=None, padding=None,
                 groups=1, bias=False):
        super().__init__()
        n_dims = get_num_dims(inputs)
        in_channels = get_num_channels(inputs)
        if isinstance(channels, str):
            channels = safe_eval(channels, in_channels)
        constructor = self.LAYERS[n_dims]

        if groups not in [1, channels]:
            raise ValueError

        kernel_size = kernel_size if kernel_size is not None else (2 * factor - 1,) * n_dims
        stride = stride if stride is not None else (factor,) * n_dims
        padding = padding if padding is not None else (0,) * n_dims

        self.layer = constructor(in_channels=in_channels, out_channels=channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 groups=groups, bias=bias)

    def forward(self, x):
        return self.layer(x)



class MultiScaleConv(nn.ModuleList):
    """ Multi scale convolution: apply convolutions with different kernel sizes in parallel,
    combine their results into one tensor with desired (default sum) operation. Somewhat similar to Inception blocks.
    For better efficiency, KxK kernels are separated into sequence of Kx1 and 1xK kernels.

    This layers preserves the number of channels, and by default applied depth-wise convolutions.

    Parameters
    ----------
    kernel_size : sequence of ints
        Kernel sizes to include.
    """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

    def __init__(self, inputs=None, kernel_size=(3, 5, 7), stride=1, dilation=1, groups=None, bias=False, combine='+'):
        super().__init__()
        self.combine = combine

        # Parse inputs
        ndims = get_num_dims(inputs)
        if ndims != 2:
            raise NotImplementedError
        constructor = self.LAYERS[ndims]

        in_channels = get_num_channels(inputs)
        groups = groups or in_channels

        # Parse other parameters
        n = len(kernel_size)
        dilation = to_n_tuple(dilation, n)
        groups = to_n_tuple(groups, n)

        # Create layers
        for kernel_size_, groups_, dilation_ in zip(kernel_size, groups, dilation):
            sequential = []
            for kernel_size__ in [[kernel_size_, 1], [1, kernel_size_]]:
                padding_ = compute_padding(padding='same', shape=inputs.shape[2:], kernel_size=kernel_size__,
                                           dilation=dilation_, stride=stride)['padding']

                layer = constructor(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size__,
                                    padding=padding_, stride=stride, groups=groups_, dilation=dilation_, bias=bias)
                sequential.append(layer)

            self.append(nn.Sequential(*sequential))


    def forward(self, x):
        return Combine(op=self.combine)([layer(x) for layer in self])
