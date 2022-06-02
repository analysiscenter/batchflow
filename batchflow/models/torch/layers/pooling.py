""" Pooling layers. """
#pylint: disable=not-callable, invalid-name
import torch
from torch import nn

from .utils import compute_padding
from ..utils import get_num_dims


class BasePool(nn.Module):
    """ Base class for pooling.
    Selects appropriate layer, depending on input's number of dimensions, and applies proper padding.
    """
    LAYERS = {}

    def __init__(self, inputs=None, pool_size=2, pool_stride=2, padding='same'):
        super().__init__()

        padding = compute_padding(padding=padding, shape=inputs.shape[2:], kernel_size=pool_size,
                                  dilation=1, stride=pool_stride, transposed=False)['padding']

        layer = self.LAYERS[get_num_dims(inputs)]
        self.layer = layer(kernel_size=pool_size, stride=pool_stride, padding=padding)

    def forward(self, x):
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
    """ Base class for global pooling.
    Selects appropriate layer, depending on input's number of dimensions.
    """
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



MAX_ALIASES = ['max', 'p', 'P']
AVG_ALIASES = ['avg', 'mean', 'v', 'V']
SUM_ALIASES = ['sum', 'plus', '+']

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
