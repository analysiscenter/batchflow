""" Pooling layers. """
#pylint: disable=not-callable, invalid-name
import torch
from torch import nn
import torch.nn.functional as F

from .utils import calc_padding
from ..utils import get_num_dims


MAX_ALIASES = ['max', 'p', 'P']
AVG_ALIASES = ['avg', 'mean', 'v', 'V']
SUM_ALIASES = ['sum', 'plus', '+']


class BasePool(nn.Module):
    """ Base class for that can select appropriate torch.nn layer, depending on input's number of dimensions,
    and apply proper padding.
    """
    LAYERS = {}

    def __init__(self, inputs=None, pool_size=2, pool_stride=2, padding='same'):
        super().__init__()

        if padding is not None:
            padding = calc_padding(inputs=inputs, padding=padding, kernel_size=pool_size, stride=pool_stride)

            if isinstance(padding, tuple) and isinstance(padding[0], tuple):
                self.padding = sum(padding, ())
            elif isinstance(padding, int):
                self.padding = (padding, ) * (2 * get_num_dims(inputs))
            else:
                raise ValueError('Incorrect padding!')

        layer = self.LAYERS[get_num_dims(inputs)]
        self.layer = layer(kernel_size=pool_size, stride=pool_stride)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)

class AvgPool(BasePool):
    """ Multi-dimensional average pooling layer. """
    LAYERS = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }

class MaxPool(BasePool):
    """ Multi-dimensional max pooling layer. """
    LAYERS = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }




class BaseGlobalPool(nn.Module):

    def __init__(self, inputs=None):
        super().__init__()
        ndims = get_num_dims(inputs)
        output_shape = [1] * ndims

        layer = self.LAYERS[ndims]
        self.layer = layer(output_size=output_shape)

    def forward(self, x):
        return self.layer(x).view(x.shape[0], -1)

class GlobalAvgPool(BaseGlobalPool):
    """ Multi-dimensional adaptive average pooling layer. """
    LAYERS = {
        1: nn.AdaptiveAvgPool1d,
        2: nn.AdaptiveAvgPool2d,
        3: nn.AdaptiveAvgPool3d,
    }

class GlobalMaxPool(BaseGlobalPool):
    """ Multi-dimensional adaptive max pooling layer. """
    LAYERS = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }




class ChannelPool(nn.Module):
    """ Channel pooling layer. """
    OP_SELECTOR = {**{op:  torch.max for op in MAX_ALIASES},
                   **{op:  torch.sum for op in SUM_ALIASES},
                   **{op:  torch.mean for op in AVG_ALIASES}}

    def __init__(self, op='mean'):
        super().__init__()
        self.op = self.OP_SELECTOR[op]

    def forward(self, x):
        result = self.op(x, dim=1, keepdim=True)
        return result[0] if isinstance(result, tuple) else result
