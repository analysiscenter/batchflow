""" Resizing Torch layers. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape



class IncreaseDim(nn.Module):
    """ Increase dimensionality of passed tensor by one. """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = len(get_shape(x)) - 2 + self.dim
        ones = [1] * dim
        return x.view(x.size(0), -1, *ones)


class ReduceDim(nn.Module):
    """ Reduce dimensionality of passed tensor by one. """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dim = max(len(get_shape(x)) - 2 - self.dim, 0)
        ones = [1] * dim
        return x.view(x.size(0), -1, *ones)


class Reshape(nn.Module):
    """ Enforce desired shape of tensor. """
    def __init__(self, shape=None, reshape_to=None):
        super().__init__()
        self.shape = shape or reshape_to

    def forward(self, x):
        return x.view(x.size(0), *self.shape)



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

    def __init__(self, mode='b', size=None, shape=None, scale_factor=None, **kwargs):
        super().__init__()
        self.size, self.scale_factor = size or shape, scale_factor

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
    """ Resize input tensor with depth to space operation. """
    def __init__(self, upscale_factor=None):
        super().__init__(upscale_factor)


class SubPixelConv(PixelShuffle):
    """ An alias for PixelShuffle. """
    pass



class Crop(nn.Module):
    """ Crop tensor to desired shape.

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
        if (i_shape[2] > r_shape[2]) or (i_shape[3] > r_shape[3]):
            # Decrease input tensor's shape by slicing desired shape out of it
            shape = [slice(None, c) for c in r_shape[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            output = inputs[shape]
        elif (i_shape[2] < r_shape[2]) or (i_shape[3] < r_shape[3]):
            # Increase input tensor's shape by zero padding
            output = torch.zeros(*i_shape[:2], *r_shape[2:])
            output[:, :, :i_shape[2], :i_shape[3]] = inputs
        else:
            output = inputs
        return output



class Upsample(nn.Module):
    """ Upsample inputs with a given factor.

    Parameters
    ----------
    inputs
        Input tensor.
    factor : int
        Upsamping scale.
    shape : tuple of int
        Shape to upsample to (used by bilinear and NN resize).
    layout : str
        Resizing technique, a sequence of:

        - b - bilinear resize
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :class:`~.torch.ConvBlock` layers are also allowed.


    Examples
    --------
    A simple bilinear upsampling::

        x = Upsample(layout='b', shape=(256, 256), inputs=inputs)

    Upsampling with non-linear normalized transposed convolution::

        x = Upsample(layout='nat', factor=2, kernel_size=3, inputs=inputs)

    Subpixel convolution::

        x = Upsample(layout='X', factor=2, inputs=inputs)
    """
    def __init__(self, factor=2, shape=None, layout='b', inputs=None, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()

        if 't' in layout or 'T' in layout:
            kwargs['kernel_size'] = kwargs.get('kernel_size') or factor
            kwargs['strides'] = kwargs.get('strides') or factor
            kwargs['filters'] = kwargs.get('filters') or 'same'

        self.layer = ConvBlock(inputs=inputs, layout=layout, factor=factor, shape=shape, **kwargs)

    def forward(self, x):
        return self.layer(x)



class SEBlock(nn.Module):
    """ Squeeze and excitation block.

    Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of filters.
    squeeze_layout : str
        Operations of tensor processing.
    squeeze_units : int or sequence of ints
        Sizes of dense layers.
    squeeze_activations : str or sequence of str
        Activations of dense layers.
    """
    def __init__(self, inputs=None, ratio=4, squeeze_layout='Vfafa', squeeze_units=None, squeeze_activations=None):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        in_units = get_shape(inputs)[1]
        units = squeeze_units or [in_units // ratio, in_units]
        activations = squeeze_activations or ['relu', 'sigmoid']

        self.layer = ConvBlock(layout=squeeze_layout, units=units, activations=activations, inputs=inputs)


    def forward(self, x):
        x = self.layer(x)
        return x.view(x.size(0), -1, 1, 1)



class SideBlock(nn.Module):
    """ Add side branch to a :class:`~.layers.ConvBlock`. """
    def __init__(self, inputs=None, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        self.layer = ConvBlock(inputs=inputs, **kwargs)

    def forward(self, x):
        return self.layer(x)



class Combine(nn.Module):
    """ Combine list of tensor into one.

    Parameters
    ----------
    inputs : sequence of torch.Tensors
        Tensors to combine.

    op : str or callable
        If callable, then operation to be applied to the list of inputs.
        If 'concat', 'cat', '.', then inputs are concated along channels axis.
        If 'sum', '+', then inputs are summed.
        If 'mul', '*', then inputs are multiplied.
        If 'avg', then inputs are averaged.
        If 'softsum', '&', then every tensor is passed through 1x1 convolution in order to have
        the same number of channels as the first tensor, and then summed.
    """
    @staticmethod
    def concat(inputs):
        return torch.cat(inputs, dim=1)

    @staticmethod
    def sum(inputs):
        return torch.stack(inputs, dim=0).sum(dim=0)

    @staticmethod
    def mul(inputs):
        """ Multiplication. """
        result = 1
        for item in inputs:
            result = result * item
        return result

    @staticmethod
    def mean(inputs):
        return torch.mean(inputs)

    @staticmethod
    def softsum(self, inputs):
        """ Softsum. """
        #pylint: disable=bad-staticmethod-argument
        inputs = [conv(tensor) for conv, tensor in zip(self.conv, inputs)]
        return torch.stack(inputs, dim=0).sum(dim=0)

    OPS = {
        concat.__func__: ['concat', 'cat', '.'],
        sum.__func__: ['sum', 'plus', '+'],
        mul.__func__: ['multi', 'mul', '*'],
        mean.__func__: ['average', 'avg', 'mean'],
        softsum.__func__: ['softsum', '&'],
    }

    OPS = {alias: method for method, aliases in OPS.items() for alias in aliases}
    ALL_OPS = list(OPS.keys())

    def __init__(self, inputs=None, op='concat'):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()

        self.op = op
        if op in self.OPS and self.OPS[op].__name__ == 'softsum':
            args = dict(layout='c', filters=get_shape(inputs[0])[1],
                        kernel_size=1)
            conv = [ConvBlock(inputs=tensor, **args) for tensor in inputs]
            self.conv = nn.ModuleList(conv)
            self.op = lambda inputs: self.OPS[op](self, inputs)


    def spatial_resize(self, inputs):
        """ Force the same shapes of the inputs, if needed. """
        shape_ = get_shape(inputs[0])
        spatial_shape_ = shape_[len(shape_)-2:]

        resized = []
        for item in inputs:
            shape = get_shape(item)
            dim = len(shape) - 2
            spatial_shape = shape[dim:]
            if dim > 0 and spatial_shape_ != tuple([1]*dim) and spatial_shape != spatial_shape_:
                item = Crop(inputs[0])(item)
            resized.append(item)
        return resized


    def forward(self, inputs):
        if callable(self.op):
            return self.op(inputs)
        inputs = self.spatial_resize(inputs)
        if self.op in self.OPS:
            return self.OPS[self.op](inputs)
        raise ValueError('Combine operation must be a callable or \
                          one from {}, instead got {}.'.format(self.ALL_OPS, self.op))

    def extra_repr(self):
        if isinstance(self.op, str):
            return 'op=' + self.op
        return 'op=' + 'callable ' + self.op.__name__
