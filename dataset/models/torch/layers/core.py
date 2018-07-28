""" Contains common layers """
import torch.nn as nn

from ..utils import get_num_dims, get_num_channels


ACTIVATIONS = {f.lower(): f for f in dir(nn)}


class Identity(nn.Module):
    """ Module which just returns its inputs

    Notes
    -----
    It slows training and inference so you should have a very good reason to use it.
    For instance, this could be a good option to replace some other module when debugging.
    """
    def __init__(self, shape=None):
        super().__init__()
        self.output_shape = tuple([*shape])

    def forward(self, x):
        return x


class Flatten(nn.Module):
    """ A module which reshapes inputs into 2-dimension (batch_items, features) """
    def forward(self, x):
        return x.view(x.size(0), -1)


class Dense(nn.Module):
    """ A dense layer """
    def __init__(self, units=None, out_features=None, bias=True, shape=None):
        super().__init__()

        units = units or out_features

        self.output_shape = [*shape]
        self.output_shape[-1] = units
        self.output_shape = tuple(self.output_shape)

        self.linear = nn.Linear(shape[-1], units, bias)

    def forward(self, x):
        return self.linear(x)


class Activation(nn.Module):
    """ A proxy activation module

    Parameters
    ----------
    activation : str, nn.Module, callable or None
        an activation function, can be

        - None - for identity function `f(x) = x`
        - str - a name from `torch.nn`
        - an instance of activation module (e.g. `torch.nn.ReLU()` or `torch.nn.ELU(alpha=2.0)`)
        - a class of activation module (e.g. `torch.nn.ReLU` or `torch.nn.ELU`)
        - a callable (e.g. `F.relu` or your custom function)

    args
        custom positional arguments passed to

        - a module class when creating a function
        - a callable during forward pass

    kwargs
        custom named arguments
    """
    def __init__(self, activation, *args, shape=None, **kwargs):
        super().__init__()

        self.args = tuple()
        self.kwargs = {}
        self.output_shape = shape

        if activation is None:
            self.activation = None
        if isinstance(activation, str):
            a = activation.lower()
            if a in ACTIVATIONS:
                self.activation = getattr(nn, ACTIVATIONS[a])(*args, **kwargs)
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


def _get_padding(padding=0, kernel_size=None, dilation=1, **kwargs):
    _ = kwargs
    if isinstance(padding, str):
        if padding == 'valid':
            padding = 0
        elif padding == 'same':
            padding = dilation * (kernel_size - 1) // 2
        else:
            raise ValueError("padding can be 'same' or 'valid' or int")
    return padding


class _Conv(nn.Module):
    """ An universal module for plain and transposed convolutions """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, shape=None, _fn=None):

        super().__init__()

        self.input_shape = shape
        self.output_shape = [*shape]
        self.output_shape[1] = filters
        self.output_shape = tuple(self.output_shape)

        args = {}

        args['in_channels'] = get_num_channels(shape)

        args['out_channels'] = filters

        args['groups'] = groups or 1

        args['kernel_size'] = kernel_size

        args['dilation'] = dilation or dilation_rate or 1

        args['stride'] = stride or strides or 1

        args['bias'] = bias

        args['padding'] = _get_padding(padding, args['kernel_size'], args['dilation'])

        self.conv = _fn[get_num_dims(shape)](**args)

    def forward(self, x):
        return self.conv(x)


CONV = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

class Conv(_Conv):
    """ Multi-dimensional convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, shape=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, groups, bias, shape, CONV)

class _SeparableConv(nn.Module):
    """ A universal multi-dimensional separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, shape=None, _fn=None):
        super().__init__()

        in_channels = get_num_channels(shape)
        out_channels = in_channels * depth_multiplier
        self.conv = _fn(out_channels, kernel_size, stride, strides, padding,
                        dilation, dilation_rate, in_channels, bias, shape)

        if filters != out_channels:
            self.conv = nn.Sequential(
                self.conv,
                Conv(filters, 1, 1, 1, padding, 1, 1, 1, bias, self.conv.output_shape)
            )

    def forward(self, x):
        return self.conv(x)

class SeparableConv(_SeparableConv):
    """ Multi-dimensional separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, shape=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, bias, depth_multiplier, shape, Conv)

CONV_TR = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

class ConvTranspose(_Conv):
    """ Multi-dimensional transposed convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, groups=None, bias=True, shape=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, groups, bias, shape, CONV_TR)


class SeparableConvTranspose(_SeparableConv):
    """ Multi-dimensional transposed separable convolutional layer """
    def __init__(self, filters, kernel_size, stride=None, strides=None, padding='same',
                 dilation=None, dilation_rate=None, bias=True, depth_multiplier=1, shape=None):
        super().__init__(filters, kernel_size, stride, strides, padding,
                         dilation, dilation_rate, bias, depth_multiplier, shape, ConvTranspose)

BATCH_NORM = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}

class BatchNorm(nn.Module):
    """ Multi-dimensional batch normalization layer """
    def __init__(self, shape=None, **kwargs):
        super().__init__()
        num_features = get_num_channels(shape)
        self.norm = BATCH_NORM[get_num_dims(shape)](num_features=num_features, **kwargs)

    def forward(self, x):
        return self.norm(x)


DROPOUT = {
    1: nn.Dropout,
    2: nn.Dropout2d,
    3: nn.Dropout3d,
}

class Dropout(nn.Module):
    """ Multi-dimensional dropout layer """
    def __init__(self, shape=None, **kwargs):
        super().__init__()
        self.dropout = DROPOUT[get_num_dims(shape)](**kwargs)

    def forward(self, x):
        return self.dropout(x)


class _Pool(nn.Module):
    """ A universal pooling layer """
    def __init__(self, shape=None, _fn=None, **kwargs):
        super().__init__()
        if isinstance(_fn, dict):
            self.pool = _fn[get_num_dims(shape)](**kwargs)
        else:
            self.pool = _fn(shape=shape, **kwargs)

    def forward(self, x):
        return self.pool(x)


MAXPOOL = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}

class MaxPool(_Pool):
    """ Multi-dimensional max pooling layer """
    def __init__(self, **kwargs):
        kwargs['padding'] = _get_padding(kwargs['padding'], kwargs['kernel_size'], kwargs['dilation'])
        super().__init__(_fn=MAXPOOL, **kwargs)


AVGPOOL = {
    1: nn.AvgPool1d,
    2: nn.AvgPool2d,
    3: nn.AvgPool3d,
}

class AvgPool(_Pool):
    """ Multi-dimensional average pooling layer """
    def __init__(self, **kwargs):
        kwargs['padding'] = _get_padding(kwargs['padding'], kwargs['kernel_size'], kwargs['dilation'])
        super().__init__(_fn=AVGPOOL, **kwargs)

class Pool(_Pool):
    """ Multi-dimensional pooling layer """
    def __init__(self, shape=None, op='max', **kwargs):
        if op == 'max':
            _fn = MaxPool
        elif op in ['avg', 'mean']:
            _fn = AvgPool
        super().__init__(_fn=_fn, shape=shape, **kwargs)

    def forward(self, x):
        return self.pool(x)


ADAPTIVE_MAXPOOL = {
    1: nn.AdaptiveMaxPool1d,
    2: nn.AdaptiveMaxPool2d,
    3: nn.AdaptiveMaxPool3d,
}

class AdaptiveMaxPool(_Pool):
    """ Multi-dimensional adaptive max pooling layer """
    def __init__(self, shape=None, output_size=None, **kwargs):
        super().__init__(_fn=ADAPTIVE_MAXPOOL, shape=shape, output_size=output_size, **kwargs)
        self.output_shape = tuple(shape[:2]) + tuple(output_size)


ADAPTIVE_AVGPOOL = {
    1: nn.AdaptiveAvgPool1d,
    2: nn.AdaptiveAvgPool2d,
    3: nn.AdaptiveAvgPool3d,
}

class AdaptiveAvgPool(_Pool):
    """ Multi-dimensional adaptive average pooling layer """
    def __init__(self, shape=None, output_size=None, **kwargs):
        super().__init__(_fn=ADAPTIVE_AVGPOOL, shape=shape, output_size=output_size, **kwargs)
        self.output_shape = tuple(shape[:2]) + tuple(output_size)


class AdaptivePool(_Pool):
    """ Multi-dimensional adaptive pooling layer """
    def __init__(self, op='max', shape=None, **kwargs):
        if op == 'max':
            _fn = AdaptiveMaxPool
        elif op in ['avg', 'mean']:
            _fn = AdaptiveAvgPool
        super().__init__(_fn=_fn, shape=shape, **kwargs)
        self.output_shape = self.pool.output_shape


class GlobalPool(nn.Module):
    """ Multi-dimensional global pooling layer """
    def __init__(self, shape=None, op='max', **kwargs):
        super().__init__()
        pool_shape = [1] * len(shape[2:])
        self.output_shape = tuple(shape[:2])
        self.pool = nn.Sequential(
            AdaptivePool(op=op, output_size=pool_shape, shape=shape, **kwargs),
            Flatten()
        )

    def forward(self, x):
        return self.pool(x)
