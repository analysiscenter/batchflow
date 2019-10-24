""" All the layers. """

import os
import re
import threading
import inspect
from functools import partial

import numpy as np
import torch
import torch.nn as nn



def get_shape(inputs, shape=None):
    """ Return inputs shape """
    if inputs is None:
        pass
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        shape = tuple(inputs)
    elif isinstance(inputs, torch.nn.Module):
        shape = get_output_shape(inputs, shape)
    else:
        raise TypeError('inputs can be array, tensor, tuple/list or layer', type(inputs))
    return shape



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


class Conv(nn.Module):
    def __init__(self, filters, kernel_size=3, inputs=None):
        super().__init__()

        in_filters = get_shape(inputs)[1]
        self.conv = nn.Conv2d(in_filters, filters, kernel_size)

    def forward(self, inputs):
        return self.conv(inputs)


ACTIVATIONS = {f.lower(): f for f in dir(nn)}

class Activation(nn.Module):
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
            if a in ACTIVATIONS:
                _activation = getattr(nn, ACTIVATIONS[a])
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



class ConvBlock(nn.Module):
    def __init__(self, inputs, layout='', filters=None, units=None, **kwargs):
        super().__init__()

        self.layout = layout
        self.filters = filters
        print(layout)
        layers = []
        c_counter, f_counter = 0, 0
        for letter in layout:
            if letter == 'c':
                block = Conv(filters=filters[c_counter], inputs=inputs)
                c_counter += 1
            elif letter == 'f':
                block = Dense(units=units[f_counter], inputs=inputs)
                f_counter += 1
            elif letter == 'a':
                block = Activation('relu')

            inputs = block(inputs)
            layers.append(block)

        self.block = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.block(inputs)
