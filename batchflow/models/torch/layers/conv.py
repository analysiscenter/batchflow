""" Convolutional layers. """
#pylint: disable=not-callable
from torch import nn
import torch.nn.functional as F

from .utils import calc_padding
from ..utils import get_num_channels, get_num_dims, safe_eval



class BaseConv(nn.Module):
    """ An universal module for plain and transposed convolutions. """
    LAYERS = {}
    TRANSPOSED = False

    def __init__(self, inputs=None, channels=None, kernel_size=3, stride=1, dilation=1, groups=1,
                 padding='same', custom_padding=False, bias=False):
        super().__init__()

        if isinstance(channels, str):
            channels = safe_eval(channels, get_num_channels(inputs))

        args = {
            'in_channels': get_num_channels(inputs),
            'out_channels': channels,
            'groups': groups,
            'kernel_size': kernel_size,
            'dilation': dilation,
            'stride': stride,
            'bias': bias,
        }

        calculated_padding = calc_padding(inputs, padding=padding, transposed=self.TRANSPOSED, **args)
        nested_padding = isinstance(calculated_padding, tuple) and isinstance(calculated_padding[0], tuple)

        self.padding = None
        if calculated_padding == 0:
            args['padding'] = 0
        elif (custom_padding or self.TRANSPOSED) and nested_padding:
            args['padding'] = 0
            self.padding = sum(calculated_padding, ())
        elif padding == 'same' and nested_padding:
            args['padding'] = tuple(max(0, max(item)) for item in calculated_padding)
        else:
            args['padding'] = padding

        self.layer = self.LAYERS[get_num_dims(inputs)](**args)

    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)

    def extra_repr(self):
        return f'custom_padding={self.padding is not None}'

    def __repr__(self):
        msg = super().__repr__()
        if getattr(self, 'collapsible', True):
            msg = msg.replace('(\n    (layer): ', ':').replace('\n    ', '\n  ').replace('\n  )\n)', '\n)')
        return msg

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

    def __init__(self, kernel_size=3, stride=1, padding='same',
                 dilation=1, bias=False, depth_multiplier=1, inputs=None):
        super().__init__()

        args = {
            'channels': get_num_channels(inputs) * depth_multiplier,
            'kernel_size': kernel_size,
            'groups': get_num_channels(inputs),
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
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

    def __init__(self, channels, kernel_size=3, stride=1, padding='same',
                 dilation=1, bias=False, depth_multiplier=1, inputs=None):
        super().__init__()

        self.layer = nn.Sequential(
            self.LAYER(inputs=inputs, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, bias=bias, depth_multiplier=depth_multiplier),
            Conv(inputs=inputs, channels=channels, kernel_size=1, stride=1,
                 padding=padding, dilation=1, bias=bias, )
            )

    def forward(self, x):
        return self.layer(x)


class SeparableConv(BaseSeparableConv):
    """ Multi-dimensional separable convolutional layer. """
    LAYER = DepthwiseConv


class SeparableConvTranspose(BaseSeparableConv):
    """ Multi-dimensional separable depthwise convolutional layer. """
    LAYER = DepthwiseConvTranspose
