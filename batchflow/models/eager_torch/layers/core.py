""" All the layers. """
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



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
    else:
        raise TypeError('inputs can be array, tensor, tuple/list or layer', type(inputs))
    return shape

def get_num_channels(inputs, axis=1):
    """ Return a number of channels """
    return get_shape(inputs)[axis]

def get_num_dims(inputs):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    shape = get_shape(inputs)
    dim = len(shape)
    return max(1, dim - 2)


def get_padding(kernel_size=None, width=None, dilation=1, stride=1):
    kernel_size = dilation * (kernel_size - 1) + 1
    if stride >= width:
        p = max(0, kernel_size - width)
    else:
        if width % stride == 0:
            p = kernel_size - stride
        else:
            p = kernel_size - width % stride
    p = (p // 2, p - p // 2)
    return p

def calc_padding(inputs, padding=0, kernel_size=None, dilation=1, transposed=False, stride=1, **kwargs):
    _ = kwargs
    print(kernel_size)
    dims = get_num_dims(inputs)
    shape = get_shape(inputs)

    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            if transposed:
                padding = 0
            else:
                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size,) * dims
                if isinstance(dilation, int):
                    dilation = (dilation,) * dims
                if isinstance(stride, int):
                    stride = (stride,) * dims
                padding = tuple(get_padding(kernel_size[i], shape[i+2], dilation[i], stride[i]) for i in range(dims))
        else:
            raise ValueError("padding can be 'same' or 'valid'")
    elif isinstance(padding, int):
        pass
    elif isinstance(padding, tuple):
        pass
    else:
        raise ValueError("padding can be 'same' or 'valid' or int or tuple of int")
    return padding

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



class BaseConv(nn.Module):
    LAYERS = {}
    TRANSPOSED = False

    def __init__(self, filters, kernel_size, stride=1, strides=None, padding='same',
                dilation=1, dilation_rate=None, groups=1, bias=True, inputs=None):
        super().__init__()

        args = {
            'in_channels': get_num_channels(inputs),
            'out_channels': filters,
            'groups': groups,
            'kernel_size': kernel_size,
            'dilation': dilation or dilation_rate, # keep only `dilation_rate`?
            'stride': stride or strides, # keep only `strides`?
            'bias': bias,
        }


        padding = calc_padding(inputs, padding=padding, transposed=self.TRANSPOSED, **args)
        if isinstance(padding, tuple) and isinstance(padding[0], tuple):
            args['padding'] = 0
            self.padding = sum(padding, ())
        else:
            args['padding'] = padding
            self.padding = 0

        self.layer = self.LAYERS[get_num_dims(inputs)](**args)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)


class Conv(BaseConv):
    """ Multi-dimensional convolutional layer """
    LAYERS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }
    TRANSPOSED = False


class ConvTranspose(BaseConv):
    """ Multi-dimensional transposed convolutional layer """
    LAYERS = {
        1: nn.ConvTranspose1d,
        2: nn.ConvTranspose2d,
        3: nn.ConvTranspose3d,
    }
    TRANSPOSED = True



class BaseDepthwiseConv(nn.Module):
    LAYER = None

    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__()

        in_channels = get_num_channels(inputs)
        out_channels = in_channels * depth_multiplier
        self.layer = self.LAYER(out_channels, kernel_size, strides=strides, padding=padding,
                                dilation_rate=dilation_rate, groups=in_channels, bias=bias, inputs=inputs)

    def forward(self, x):
        return self.layer(x)


class DepthwiseConv(BaseDepthwiseConv):
    LAYER = Conv


class DepthwiseConvTranspose(BaseDepthwiseConv):
    LAYER = ConvTranspose



class BaseSeparableConv(nn.Module):
    LAYER = None

    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, inputs=None):
        super().__init__()

        self.layer = nn.Sequenital(
            self.LAYER(filters, kernel_size, stride, strides, padding,
                       dilation, dilation_rate, bias, depth_multiplier, inputs),
            Conv(filters, kernel_size=1, strides=1, padding=padding, dilation_rate=1, bias=bias, inputs=inputs)
            )

    def forward(self, x):
        return self.layer(x)


class SeparableConv(BaseSeparableConv):
    LAYER = DepthwiseConv


class SeparableConvTranspose(BaseSeparableConv):
    LAYER = DepthwiseConvTranspose





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

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        self.layer = self.LAYERS[get_num_dims(inputs)](**kwargs)

    def forward(self, x):
        return self.layer(x)



class BasePool(nn.Module):
    """ A universal pooling layer """
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
                    'strides': pool_strides,
                }

                padding = calc_padding(inputs=inputs, padding=padding, **{**kwargs, **args})

                print('P', padding)
                if isinstance(padding, tuple) and isinstance(padding[0], tuple):
                    self.padding = sum(padding, ())
                else:
                    kwargs['padding'] = padding

            print('WTF', self.LAYERS, get_num_dims(inputs))
            self.layer = self.LAYERS[get_num_dims(inputs)](kernel_size=pool_size, **kwargs)

    def forward(self, x):
        if self.padding:
            x = F.pad(x, self.padding[::-1])
        return self.layer(x)


class MaxPool(BasePool):
    """ Multi-dimensional max pooling layer """
    LAYERS = {
        1: nn.MaxPool1d,
        2: nn.MaxPool2d,
        3: nn.MaxPool3d,
    }


class AvgPool(BasePool):
    """ Multi-dimensional average pooling layer """
    LAYERS = {
        1: nn.AvgPool1d,
        2: nn.AvgPool2d,
        3: nn.AvgPool3d,
    }


class Pool(BasePool):
    """ Multi-dimensional pooling layer """
    def __init__(self, inputs=None, op='max', **kwargs):
        if op == 'max':
            self.LAYER = MaxPool
        elif op in ['avg', 'mean']:
            self.LAYER = AvgPool
        super().__init__(inputs=inputs, **kwargs)



class AdaptiveMaxPool(BasePool):
    """ Multi-dimensional adaptive max pooling layer """
    LAYERS = {
        1: nn.AdaptiveMaxPool1d,
        2: nn.AdaptiveMaxPool2d,
        3: nn.AdaptiveMaxPool3d,
    }

    def __init__(self, inputs=None, output_size=None, **kwargs):
        super().__init__(inputs=inputs, output_size=output_size, padding=None, **kwargs)



class AdaptiveAvgPool(BasePool):
    """ Multi-dimensional adaptive average pooling layer """
    LAYERS = {
        1: nn.AdaptiveAvgPool1d,
        2: nn.AdaptiveAvgPool2d,
        3: nn.AdaptiveAvgPool3d,
    }
    def __init__(self, inputs=None, output_size=None, **kwargs):
        kwargs.pop('padding', None)
        super().__init__(inputs=inputs, output_size=output_size, padding=None, **kwargs)



class AdaptivePool(BasePool):
    """ Multi-dimensional adaptive pooling layer """
    def __init__(self, op='max', inputs=None, **kwargs):
        if op == 'max':
            self.LAYER = AdaptiveMaxPool
        elif op in ['avg', 'mean']:
            self.LAYER = AdaptiveAvgPool
        super().__init__(inputs=inputs, padding=None, **kwargs)


class GlobalPool(nn.Module):
    """ Multi-dimensional global pooling layer """
    def __init__(self, inputs=None, op='max', **kwargs):
        super().__init__()
        shape = get_shape(inputs)
        pool_shape = [1] * len(shape[2:])
        self.pool = AdaptivePool(op=op, output_size=pool_shape, inputs=inputs, **kwargs)

    def forward(self, x):
        x = self.pool(x)
        return x.view(x.size(0), -1)




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

    def __init__(self, *args, inputs=None, **kwargs):
        super().__init__()
        _ = args

        if kwargs.get('mode') in self.MODES:
            kwargs['mode'] = self.MODES[mode]
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, **self.kwargs)



class PixelShuffle(nn.PixelShuffle):
    """ Resize input tensor with depth to space operation """
    def __init__(self, upscale_factor=None, inputs=None):
        super().__init__(upscale_factor)


class SubPixelConv(PixelShuffle):
    """ An alias for PixelShuffle """
    pass



# class ConvBlock(nn.Module):
#     def __init__(self, inputs, layout='', filters=None, units=None, **kwargs):
#         super().__init__()

#         self.layout = layout
#         self.filters = filters

#         print('ConvBlock layout', layout)
#         layers = []
#         c_counter, f_counter = 0, 0
#         for letter in layout:
#             if letter == 'c':
#                 block = Conv(filters=filters[c_counter], inputs=inputs)
#                 c_counter += 1
#             elif letter == 'f':
#                 block = Dense(units=units[f_counter], inputs=inputs)
#                 f_counter += 1
#             elif letter == 'a':
#                 block = Activation('relu')

#             inputs = block(inputs)
#             layers.append(block)

#         self.block = nn.Sequential(*layers)

#     def forward(self, inputs):
#         return self.block(inputs)
