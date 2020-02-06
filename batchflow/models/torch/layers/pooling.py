""" Pooling layers. """
#pylint: disable=not-callable, invalid-name
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_dims, calc_padding


MAX_ALIASES = ['max', 'p', 'P']
AVG_ALIASES = ['avg', 'mean', 'v', 'V']
SUM_ALIASES = ['sum', 'plus', '+']


class BasePool(nn.Module):
    """ Base class that can select appropriate torch.nn layer depending on input's number of dimensions. """
    LAYERS = {}

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        layer = self.LAYERS[get_num_dims(inputs)]
        self.layer = layer(**kwargs)

    def forward(self, x):
        return self.layer(x)


class AdaptiveMaxPool(BasePool):
    """ Multi-dimensional adaptive max pooling layer. """
    LAYERS = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }


class AdaptiveAvgPool(BasePool):
    """ Multi-dimensional adaptive average pooling layer. """
    LAYERS = {
        1: nn.AdaptiveAvgPool1d,
        2: nn.AdaptiveAvgPool2d,
        3: nn.AdaptiveAvgPool3d,
    }


class BasePoolWithPadding(BasePool):
    """
    Base class that selects torch.nn layer depending on input's number of dimensions
    AND makes proper padding
    """

    def __init__(self, inputs=None, pool_size=2, pool_strides=2, padding='same'):
        self.padding = None

        if padding is not None:
            padding = calc_padding(inputs=inputs, padding=padding, kernel_size=pool_size, stride=pool_strides)

            if isinstance(padding, tuple) and isinstance(padding[0], tuple):
                self.padding = sum(padding, ())
            elif isinstance(padding, int):
                self.padding = (padding, ) * (2 * get_num_dims(inputs))
            else:
                raise ValueError('Incorrect padding!')

        super().__init__(inputs, kernel_size=pool_size, stride=pool_strides)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return super().forward(x)


class MaxPool(BasePoolWithPadding):
    """ Multi-dimensional max pooling layer. """
    LAYERS = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }


class AvgPool(BasePoolWithPadding):
    """ Multi-dimensional average pooling layer. """
    LAYERS = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }


class Pool(nn.Module):
    """ Multi-dimensional pooling layer that selects aggregation strategy (avg or max) """
    OP_SELECTOR = {**{op:  MaxPool for op in MAX_ALIASES},
                   **{op:  AvgPool for op in AVG_ALIASES}}

    def __init__(self, inputs=None, op='max', pool_size=2, pool_strides=2, padding='same'):
        super().__init__()
        self.pool = self.OP_SELECTOR[op](inputs=inputs,
                                         pool_size=pool_size, pool_strides=pool_strides, padding=padding)

    def forward(self, x):
        x = self.pool(x)
        return x


class AdaptivePool(nn.Module):
    """ Multi-dimensional adaptive pooling layer, that selects aggregation strategy (avg or max) """
    OP_SELECTOR = {**{op:  AdaptiveMaxPool for op in MAX_ALIASES},
                   **{op:  AdaptiveAvgPool for op in AVG_ALIASES}}

    def __init__(self, inputs=None, op='max', output_size=None):
        super().__init__()
        self.pool = self.OP_SELECTOR[op](output_size=output_size, inputs=inputs)

    def forward(self, x):
        return self.pool(x)


class GlobalPool(AdaptivePool):
    """ Multi-dimensional global pooling layer. """
    def __init__(self, inputs=None, op='max'):
        shape = get_shape(inputs)
        pool_shape = [1] * len(shape[2:])
        super().__init__(inputs=inputs, op=op, output_size=pool_shape)

    def forward(self, x):
        x = super().forward(x)
        return x.view(x.size(0), -1)


class GlobalMaxPool(GlobalPool):
    """ Multi-dimensional global max pooling layer. """
    def __init__(self, inputs=None):
        super().__init__(inputs=inputs, op='max')


class GlobalAvgPool(GlobalPool):
    """ Multi-dimensional global avg pooling layer. """
    def __init__(self, inputs=None):
        super().__init__(inputs=inputs, op='avg')


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
