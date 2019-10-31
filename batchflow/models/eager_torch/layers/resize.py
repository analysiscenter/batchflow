""" Resizing Torch layers. """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape



class Interpolate(nn.Module):
    """ Upsample inputs with a given factor

    Notes
    -----
    This is just a wrapper around ``F.interpolate``.

    For brevity ``mode`` can be specified with the first letter only: 'n', 'l', 'b', 't'.

    All the parameters should the specified as keyword arguments (i.e. with names and values).
    """
    MODES = {
        'n': 'nearest',
        'l': 'linear',
        'b': 'bilinear',
        't': 'trilinear',
    }

    def __init__(self, inputs=None, mode='b', size=None, scale_factor=None, **kwargs):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

        if mode in self.MODES:
            mode = self.MODES[mode]
        self.mode = mode
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, mode=self.mode, size=self.size, scale_factor=self.scale_factor,
                             align_corners=True, **self.kwargs)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class PixelShuffle(nn.PixelShuffle):
    """ Resize input tensor with depth to space operation """
    def __init__(self, upscale_factor=None, inputs=None):
        super().__init__(upscale_factor)


class SubPixelConv(PixelShuffle):
    """ An alias for PixelShuffle """
    pass



class Crop(nn.Module):
    """ Crop tensor to desired shape.

    Parameters
    ----------
    inputs
        Input tensor.
    resize_to
        Tensor or shape to resize input tensor to.
    """
    def __init__(self, inputs, resize_to):
        super().__init__()

        self.resize_to = resize_to


    def forward(self, inputs, resize_to):
        i_shape = get_shape(inputs)
        r_shape = get_shape(resize_to)
        if (i_shape[2] > r_shape[2]) or (i_shape[3] > r_shape[3]):
            # Decrease input tensor's shape by slicing desired shape out of it
            shape = [slice(None, c) for c in resize_to.size()[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            output = inputs[shape]
        elif (i_shape[2] < r_shape[2]) or (i_shape[3] < r_shape[3]):
            # Increase input tensor's shape by zero padding
            output = torch.zeros(*i_shape[:2], *r_shape[2:])
            output[:, :, :i_shape[2], :i_shape[3]] = inputs
        else:
            output = inputs
        return output
