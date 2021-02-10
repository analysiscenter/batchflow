""" Convolutional layers. """
#pylint: disable=not-callable
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_num_channels, get_num_dims, safe_eval, calc_padding


class BaseConv(nn.Module):
    """ An universal module for plain and transposed convolutions. """
    LAYERS = {}
    TRANSPOSED = False

    def __init__(self, filters, kernel_size=3, strides=1, padding='same',
                 dilation_rate=1, groups=1, bias=False, inputs=None):
        super().__init__()

        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        args = {
            'in_channels': get_num_channels(inputs),
            'out_channels': filters,
            'groups': groups,
            'kernel_size': kernel_size,
            'dilation': dilation_rate,
            'stride': strides,
            'bias': bias,
        }

        padding = calc_padding(inputs, padding=padding, transposed=self.TRANSPOSED, **args)

        if isinstance(padding, tuple) and isinstance(padding[0], tuple):
            args['padding'] = 0
            self.padding = sum(padding, ())
        else:
            args['padding'] = padding
            self.padding = 0

        self.layer = self.LAYERS[get_num_dims(inputs)](**args)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)


class Conv(BaseConv):
    """ Multi-dimensional convolutional layer. """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }
    TRANSPOSED = False


class ConvTranspose(BaseConv):
    """ Multi-dimensional transposed convolutional layer. """
    LAYERS = {
        1: nn.ConvTranspose1d,
        2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d,
    }
    TRANSPOSED = True



class BaseDepthwiseConv(nn.Module):
    """ An universal module for plain and transposed depthwise convolutions. """
    LAYER = None

    def __init__(self, kernel_size=3, strides=1, padding='same',
                 dilation_rate=1, bias=False, depth_multiplier=1, inputs=None):
        super().__init__()

        args = {
            'filters': get_num_channels(inputs) * depth_multiplier,
            'kernel_size': kernel_size,
            'groups': get_num_channels(inputs),
            'strides': strides,
            'padding': padding,
            'dilation_rate': dilation_rate,
            'bias': bias,
        }

        self.layer = self.LAYER(inputs=inputs, **args)

    def forward(self, x):
        return self.layer(x)


class DepthwiseConv(BaseDepthwiseConv):
    """ Multi-dimensional depthwise convolutional layer. """
    LAYER = Conv


class DepthwiseConvTranspose(BaseDepthwiseConv):
    """ Multi-dimensional transposed depthwise convolutional layer. """
    LAYER = ConvTranspose



class BaseSeparableConv(nn.Module):
    """ An universal module for plain and transposed separable convolutions. """
    LAYER = None

    def __init__(self, filters, kernel_size=3, strides=1, padding='same',
                 dilation_rate=1, bias=False, depth_multiplier=1, inputs=None):
        super().__init__()

        self.layer = nn.Sequential(
            self.LAYER(kernel_size, strides, padding, dilation_rate, bias, depth_multiplier, inputs),
            Conv(filters, kernel_size=1, strides=1, padding=padding, dilation_rate=1, bias=bias, inputs=inputs)
            )

    def forward(self, x):
        return self.layer(x)


class SeparableConv(BaseSeparableConv):
    """ Multi-dimensional separable convolutional layer. """
    LAYER = DepthwiseConv


class SeparableConvTranspose(BaseSeparableConv):
    """ Multi-dimensional separable depthwise convolutional layer. """
    LAYER = DepthwiseConvTranspose
