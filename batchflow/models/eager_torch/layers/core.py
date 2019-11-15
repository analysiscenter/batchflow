""" Basic Torch layers. """
import inspect

import numpy as np
import torch
import torch.nn as nn

from .resize import Combine
from ..utils import get_shape, get_num_channels, get_num_dims



class Identity(nn.Module):
    """ Module which just returns its inputs.

    Notes
    -----
    It slows training and inference so you should have a very good reason to use it.
    For instance, this could be a good option to replace some other module when debugging.
    """
    def forward(self, x):
        return x



class Flatten(nn.Module):
    """ A module which reshapes inputs into 2-dimension (batch_items, features). """
    def forward(self, x):
        return x.view(x.size(0), -1)



class Dense(nn.Module):
    """ Dense layer. """
    def __init__(self, units=None, out_features=None, bias=True, inputs=None):
        super().__init__()

        in_units = np.prod(get_shape(inputs)[1:])
        units = units or out_features
        self.linear = nn.Linear(in_units, units, bias)

    def forward(self, x):
        if x.dim() > 2:
            x = Flatten()(x)
        return self.linear(x)



class Activation(nn.Module):
    """ Proxy activation module.

    Parameters
    ----------
    activation : str, nn.Module, callable or None
        If None, then identity function `f(x) = x`.
        If str, then name from `torch.nn`
        Also can be an instance of activation module (e.g. `torch.nn.ReLU()` or `torch.nn.ELU(alpha=2.0)`),
        or a class of activation module (e.g. `torch.nn.ReLU` or `torch.nn.ELU`),
        or a callable (e.g. `F.relu` or your custom function).
    args
        Positional arguments passed to either class initializer or callable.
    kwargs
        Additional named arguments passed to either class initializer or callable.
    """
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
        if self.activation:
            return self.activation(x, *self.args, **self.kwargs)
        return x



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
    """ Multi-dimensional dropout layer. """
    LAYERS = {
        1: nn.Dropout,
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }

    def __init__(self, inputs=None, dropout_rate=0.5, multisample=False, **kwargs):
        super().__init__()
        self.multisample = multisample
        self.sum = Combine(op='sum')
        print(dropout_rate, multisample)
        self.layer = self.LAYERS[get_num_dims(inputs)](p=dropout_rate)

    def forward(self, x):
        if self.multisample is not False:
            if self.multisample is True:
                self.multisample = 2
            elif isinstance(self.multisample, float):
                self.multisample = [self.multisample, 1 - self.multisample]

            if isinstance(self.multisample, int): # dropout to the whole batch, then average
                dropped = [self.layer(x) for _ in range(self.multisample)]
                output = self.sum(dropped) / self.multisample
            else: # split batch into separate-dropout branches
                if isinstance(self.multisample, (tuple, list)):
                    if all([isinstance(item, int) for item in self.multisample]):
                        sizes = self.multisample
                    elif all([isinstance(item, float) for item in self.multisample]):
                        batch_size = x.shape[0]
                        sizes = torch.Tensor([batch_size*item for item in self.multisample[:-1]])
                        sizes = torch.round(sizes)
                        residual = (x.shape[0] - torch.mean(sizes))
                        residual = torch.reshape(residual, shape=(1,))
                        sizes = torch.cat([sizes, residual], axis=0).to(int)
                else: # case of Tensor
                    sizes = self.multisample

                splitted = torch.split(x, tuple(sizes))
                dropped = [self.layer(branch) for branch in splitted]
                output = torch.cat(dropped, axis=0)
        else:
            output = self.layer(x)
        return output


class AlphaDropout(Dropout):
    """ Multi-dimensional alpha-dropout layer. """
    LAYERS = {
        1: nn.AlphaDropout,
        2: nn.AlphaDropout,
        3: nn.AlphaDropout,
    }
