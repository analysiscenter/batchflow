""" Convolutional layers. """
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_channels, get_num_dims, calc_padding


class BaseConv(nn.Module):
    LAYERS = {}
    TRANSPOSED = False

    def __init__(self, filters, kernel_size, stride=1, strides=None, padding='same',
                dilation=1, dilation_rate=None, groups=1, bias=True, inputs=None):
        super().__init__()

        args = {
            'in_channels': get_num_channels(inputs),
            'out_channels': filters,
            'groups': groups,
            'kernel_size': kernel_size,
            'dilation': dilation or dilation_rate,
            'stride': stride or strides,
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
    """ Multi-dimensional convolutional layer """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }
    TRANSPOSED = False


class ConvTranspose(BaseConv):
    """ Multi-dimensional transposed convolutional layer """
    LAYERS = {
        1: nn.ConvTranspose1d,
        2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d,
    }
    TRANSPOSED = True



class BaseDepthwiseConv(nn.Module):
    LAYER = None

    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__()

        in_channels = get_num_channels(inputs)
        out_channels = in_channels * depth_multiplier
        self.layer = self.LAYER(out_channels, kernel_size, strides=strides, padding=padding,
                                dilation_rate=dilation_rate, groups=in_channels, bias=bias, inputs=inputs)

    def forward(self, x):
        return self.layer(x)


class DepthwiseConv(BaseDepthwiseConv):
    LAYER = Conv


class DepthwiseConvTranspose(BaseDepthwiseConv):
    LAYER = ConvTranspose



class BaseSeparableConv(nn.Module):
    LAYER = None

    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__()

        self.layer = nn.Sequential(
            self.LAYER(filters, kernel_size, stride, strides, padding,
                       dilation, dilation_rate, bias, depth_multiplier, inputs),
            Conv(filters, kernel_size=1, strides=1, padding=padding, dilation_rate=1, bias=bias, inputs=inputs)
            )

    def forward(self, x):
        return self.layer(x)


class SeparableConv(BaseSeparableConv):
    LAYER = DepthwiseConv


class SeparableConvTranspose(BaseSeparableConv):
    LAYER = DepthwiseConvTranspose
