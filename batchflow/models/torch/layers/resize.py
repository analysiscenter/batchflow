""" Resizing Torch layers. """
import torch
from torch import nn
import torch.nn.functional as F

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
    TODO: add `central_crop`

    Parameters
    ----------
    inputs
        Input tensor.
    resize_to : tuple or torch.Tensor
        Tensor or shape to resize input tensor to.
    """
    def __init__(self, resize_to):
        super().__init__()
        self.resize_to = resize_to

    def forward(self, inputs):
        i_shape = get_shape(inputs)
        r_shape = get_shape(self.resize_to)
        output = inputs
        for i, (i_shape_, r_shape_) in enumerate(zip(i_shape[2:], r_shape[2:])):
            if i_shape_ > r_shape_:
                # Decrease input tensor's shape by slicing desired shape out of it
                shape = [slice(None, None)] * len(i_shape)
                shape[i + 2] = slice(None, r_shape_)
                output = output[shape]
            elif i_shape_ < r_shape_:
                # Increase input tensor's shape by zero padding
                zeros_shape = list(i_shape)
                zeros_shape[i + 2] = r_shape_
                zeros = torch.zeros(zeros_shape, device=inputs.device)

                shape = [slice(None, None)] * len(i_shape)
                shape[i + 2] = slice(None, i_shape_)
                zeros[shape] = output
                output = zeros
            else:
                pass
            i_shape = get_shape(output)
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


class PixelShuffle(nn.PixelShuffle):
    """ Resize input tensor with depth to space operation. """
