""" Basic Torch layers. """
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_channels, get_num_dims, calc_padding



class Identity(nn.Module):
    """ Module which just returns its inputs

    Notes
    -----
    It slows training and inference so you should have a very good reason to use it.
    For instance, this could be a good option to replace some other module when debugging.
    """
    def forward(self, x):
        return x



class Flatten(nn.Module):
    """ A module which reshapes inputs into 2-dimension (batch_items, features) """
    def forward(self, x):
        return x.view(x.size(0), -1)



class Dense(nn.Module):
    """ A dense layer """
    def __init__(self, units=None, out_features=None, bias=True, inputs=None):
        super().__init__()

        in_units = np.prod(get_shape(inputs)[1:])
        units = units or out_features
        self.linear = nn.Linear(in_units, units, bias)

    def forward(self, x):
        """ Make forward pass """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)



class Activation(nn.Module):
    ACTIVATIONS = {f.lower(): f for f in dir(nn)}

    def __init__(self, activation, *args, **kwargs):
        super().__init__()

        if 'inplace' not in kwargs:
            kwargs['inplace'] = True

        self.args = tuple()
        self.kwargs = {}

        if activation is None:
            self.activation = None
        if isinstance(activation, str):
            a = activation.lower()
            if a in self.ACTIVATIONS:
                _activation = getattr(nn, self.ACTIVATIONS[a])
                # check does activation has `in_place` parameter
                has_inplace = 'inplace' in inspect.getfullargspec(_activation).args
                if not has_inplace:
                    kwargs.pop('inplace', None)
                self.activation = _activation(*args, **kwargs)
            else:
                raise ValueError('Unknown activation', activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif issubclass(activation, nn.Module):
            self.activation = activation(*args, **kwargs)
        elif callable(activation):
            self.activation = activation
            self.args = args
            self.kwargs = kwargs
        else:
            raise ValueError("Activation can be str, nn.Module or a callable, but given", activation)

    def forward(self, x):
        """ Make forward pass """
        if self.activation:
            return self.activation(x, *self.args, **self.kwargs)
        return x



BATCH_NORM = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

class BatchNorm(nn.Module):
    """ Multi-dimensional batch normalization layer """
    LAYERS = {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    }

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        num_features = get_num_channels(inputs)
        self.layer = self.LAYERS[get_num_dims(inputs)](num_features=num_features, **kwargs)

    def forward(self, x):
        return self.layer(x)



class Dropout(nn.Module):
    """ Multi-dimensional dropout layer """
    LAYERS = {
        1: nn.Dropout,
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }

    def __init__(self, inputs=None, dropout_rate=0.0, **kwargs):
        super().__init__()
        self.layer = self.LAYERS[get_num_dims(inputs)](p=dropout_rate, **kwargs)

    def forward(self, x):
        return self.layer(x)

class AlphaDropout(Dropout):
    LAYERS = {
        1: nn.AlphaDropout,
        2: nn.AlphaDropout,
        3: nn.AlphaDropout,
    }


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
