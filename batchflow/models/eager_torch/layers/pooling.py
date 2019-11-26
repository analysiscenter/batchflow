""" Pooling layers. """
#pylint: disable=not-callable, invalid-name
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_dims, calc_padding



class BasePool(nn.Module):
    """ A universal pooling layer. """
    LAYER = None
    LAYERS = {}

    def __init__(self, inputs=None, pool_size=2, pool_strides=2, padding='same', **kwargs):
        super().__init__()
        self.padding = None

        if self.LAYER is not None:
            self.layer = self.LAYER(inputs=inputs, padding=padding, **kwargs)
        else:
            if padding is not None:
                args = {
                    'kernel_size': pool_size,
                    'stride': pool_strides,
                }

                padding = calc_padding(inputs=inputs, padding=padding, **{**kwargs, **args})
                if isinstance(padding, tuple) and isinstance(padding[0], tuple):
                    self.padding = sum(padding, ())
                else:
                    kwargs['padding'] = padding

            layer = self.LAYERS[get_num_dims(inputs)]
            if 'Adaptive' not in layer.__name__:
                kwargs['kernel_size'] = pool_size
            self.layer = layer(**kwargs)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)


class MaxPool(BasePool):
    """ Multi-dimensional max pooling layer. """
    LAYERS = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }


class AvgPool(BasePool):
    """ Multi-dimensional average pooling layer. """
    LAYERS = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }


class Pool(BasePool):
    """ Multi-dimensional pooling layer. """
    def __init__(self, inputs=None, op='max', **kwargs):
        if op in ['max', 'p', 'P']:
            self.LAYER = MaxPool
        elif op in ['avg', 'mean', 'v', 'V']:
            self.LAYER = AvgPool
        super().__init__(inputs=inputs, **kwargs)



class AdaptiveMaxPool(BasePool):
    """ Multi-dimensional adaptive max pooling layer. """
    LAYERS = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }

    def __init__(self, inputs=None, output_size=None, **kwargs):
        kwargs.pop('padding')
        super().__init__(inputs=inputs, output_size=output_size, padding=None, **kwargs)


class AdaptiveAvgPool(BasePool):
    """ Multi-dimensional adaptive average pooling layer. """
    LAYERS = {
        1: nn.AdaptiveAvgPool1d,
        2: nn.AdaptiveAvgPool2d,
        3: nn.AdaptiveAvgPool3d,
    }
    def __init__(self, inputs=None, output_size=None, **kwargs):
        kwargs.pop('padding', None)
        super().__init__(inputs=inputs, output_size=output_size, padding=None, **kwargs)


class AdaptivePool(BasePool):
    """ Multi-dimensional adaptive pooling layer. """
    def __init__(self, op='max', inputs=None, **kwargs):
        if op in ['max', 'p', 'P']:
            self.LAYER = AdaptiveMaxPool
        elif op in ['avg', 'mean', 'v', 'V']:
            self.LAYER = AdaptiveAvgPool
        super().__init__(inputs=inputs, padding=None, **kwargs)



class GlobalPool(nn.Module):
    """ Multi-dimensional global pooling layer. """
    def __init__(self, inputs=None, op='max', **kwargs):
        super().__init__()
        shape = get_shape(inputs)
        pool_shape = [1] * len(shape[2:])
        self.pool = AdaptivePool(op=op, output_size=pool_shape, inputs=inputs, **kwargs)

    def forward(self, x):
        x = self.pool(x)
        return x.view(x.size(0), -1)

class GlobalMaxPool(GlobalPool):
    """ Multi-dimensional global max pooling layer. """
    def __init__(self, inputs=None, **kwargs):
        super().__init__(inputs=inputs, op='max', **kwargs)


class GlobalAvgPool(GlobalPool):
    """ Multi-dimensional global avg pooling layer. """
    def __init__(self, inputs=None, **kwargs):
        super().__init__(inputs=inputs, op='avg', **kwargs)
