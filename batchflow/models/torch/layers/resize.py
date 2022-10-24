""" Resizing Torch layers. """
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop

from ..utils import get_shape



class IncreaseDim(nn.Module):
    """ Increase dimensionality of passed tensor by one. """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = get_shape(x)
        dim = len(get_shape(x)) - 2 + self.dim
        ones = [1] * dim
        return x.view(*shape, *ones)


class Reshape(nn.Module):
    """ Enforce desired shape of tensor. """
    def __init__(self, reshape_to=None):
        super().__init__()
        self.reshape_to = reshape_to

    def forward(self, x):
        return x.view(x.size(0), *self.reshape_to)




class Crop(nn.Module):
    """ Crop tensor to desired shape.

    Parameters
    ----------
    resize_to : tuple or torch.Tensor
        Tensor or shape to resize input tensor to.
    """
    def __init__(self, resize_to):
        super().__init__()
        self.output_shape = get_shape(resize_to)[2:]

    def forward(self, inputs):
        output = center_crop(inputs, self.output_shape)
        return output



class Interpolate(nn.Module):
    """ Upsample inputs with a given factor.

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

    def __init__(self, mode='b', shape=None, scale_factor=None, align_corners=False, **kwargs):
        super().__init__()
        self.shape, self.scale_factor = shape, scale_factor

        if mode in self.MODES:
            mode = self.MODES[mode]
        self.mode = mode
        self.align_corners = align_corners
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, mode=self.mode, size=self.shape, scale_factor=self.scale_factor,
                             align_corners=self.align_corners, **self.kwargs)

    def extra_repr(self):
        """ Report interpolation mode and factor for a repr. """
        if self.scale_factor is not None:
            info = f'scale_factor={self.scale_factor}'
        else:
            info = f'size={self.shape}'
        info += f', mode={self.mode}, align_corners={self.align_corners}'
        return info
