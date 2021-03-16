""" Resizing Torch layers. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_dims, get_num_channels



class IncreaseDim(nn.Module):
    """ Increase dimensionality of passed tensor by one. """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = get_shape(x)
        dim = len(get_shape(x)) - 2 + self.dim
        ones = [1] * dim
        return x.view(*shape, *ones)


class Reshape(nn.Module):
    """ Enforce desired shape of tensor. """
    def __init__(self, reshape_to=None):
        super().__init__()
        self.reshape_to = reshape_to

    def forward(self, x):
        return x.view(x.size(0), *self.reshape_to)



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
        output = inputs
        for i, (i_shape_, r_shape_) in enumerate(zip(i_shape[2:], r_shape[2:])):
            if i_shape_ > r_shape_:
                # Decrease input tensor's shape by slicing desired shape out of it
                shape = [slice(None, None)] * len(i_shape)
                shape[i + 2] = slice(None, r_shape_)
                output = output[shape]
            elif i_shape_ < r_shape_:
                # Increase input tensor's shape by zero padding
                zeros_shape = list(i_shape)
                zeros_shape[i + 2] = r_shape_
                zeros = torch.zeros(zeros_shape, device=inputs.device)

                shape = [slice(None, None)] * len(i_shape)
                shape[i + 2] = slice(None, i_shape_)
                zeros[shape] = output
                output = zeros
            else:
                pass
            i_shape = get_shape(output)
        return output



class Combine(nn.Module):
    """ Combine list of tensor into one.

    Parameters
    ----------
    inputs : sequence of torch.Tensors
        Tensors to combine.

    op : str or callable
        If callable, then operation to be applied to the list of inputs.
        If 'concat', 'cat', '|', then inputs are concated along channels axis.
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
        """ Addition with broadcasting. """
        result = 0
        for item in inputs:
            result = result + item
        return result

    @staticmethod
    def mul(inputs):
        """ Multiplication with broadcasting. """
        result = 1
        for item in inputs:
            result = result * item
        return result

    @staticmethod
    def mean(inputs):
        return torch.mean(inputs)

    @staticmethod
    def softsum(inputs, **kwargs):
        """ Softsum. """
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports

        args = {'layout': 'c', 'filters': get_shape(inputs[0])[1], 'kernel_size': 1,
                **kwargs}
        conv = [ConvBlock(inputs=tensor, **args) for tensor in inputs[1:]]
        conv = nn.ModuleList(conv)
        inputs = [conv(tensor) for conv, tensor in zip(conv, inputs[1:])]
        return Combine.sum(inputs)

    @staticmethod
    def attention(inputs, **kwargs):
        """ Global Attention Upsample module.
        Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang. Pyramid Attention Network
        for Semantic Segmentation <https://arxiv.org/abs/1805.10180>'_"
        """
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        x, skip = inputs[0], inputs[1]
        num_channels = get_num_channels(skip)
        num_dims = get_num_dims(skip)
        conv1 = ConvBlock(inputs=x, layout='cna', kernel_size=3, filters=num_channels, **kwargs)(x)
        conv2 = ConvBlock(inputs=skip, layout='V > cna', kernel_size=1, filters='same', dim=num_dims, **kwargs)(skip)
        weighted = Combine.mul([conv1, conv2])
        return Combine.sum([weighted, skip])

    OPS = {
        concat: ['concat', 'cat', '|'],
        sum: ['sum', 'plus', '+'],
        mul: ['multi', 'mul', '*'],
        mean: ['average', 'avg', 'mean'],
        softsum: ['softsum', '&'],
        attention: ['attention'],
    }
    OPS = {alias: getattr(method, '__func__') for method, aliases in OPS.items() for alias in aliases}

    def __init__(self, inputs=None, op='concat', force_resize=None, leading_index=0, **kwargs):
        super().__init__()
        self.name = op
        self.idx = leading_index

        if self.idx != 0:
            inputs[0], inputs[self.idx] = inputs[self.idx], inputs[0]

        self.input_shapes, self.resized_shapes, self.output_shape = None, None, None

        if op in self.OPS:
            op = self.OPS[op]
            if op.__name__ in ['softsum', 'attention']:
                self.op = lambda inputs: op(inputs, **kwargs)
                self.force_resize = force_resize if force_resize is not None else False
            else:
                self.op = op
                self.force_resize = force_resize if force_resize is not None else True
        elif callable(op):
            self.op = op
            self.force_resize = force_resize if force_resize is not None else False
        else:
            raise ValueError('Combine operation must be a callable or \
                              one from {}, instead got {}.'.format(list(self.OPS.keys()), op))

    def forward(self, inputs):
        if self.idx != 0:
            inputs[0], inputs[self.idx] = inputs[self.idx], inputs[0]

        self.input_shapes = [get_shape(item) for item in inputs]
        if self.force_resize:
            inputs = self.spatial_resize(inputs)
            self.resized_shapes = [get_shape(item) for item in inputs]
        output = self.op(inputs)
        self.output_shape = get_shape(output)
        return output

    def extra_repr(self):
        """ Report shapes before and after combination to a repr. """
        if isinstance(self.name, str):
            res = 'op={}'.format(self.name)
        else:
            res = 'op=callable {}'.format(self.name.__name__)
        res += ',\nleading_idx={}'.format(self.idx)

        res += ',\ninput_shapes=[{}]'.format(self.input_shapes)
        if self.force_resize:
            res += ',\nresized_shapes=[{}]'.format(self.resized_shapes)
        res += ',\noutput_shape={}'.format(self.output_shape)
        return res


    def spatial_resize(self, inputs):
        """ Force the same shapes of the inputs, if needed. """
        shape_ = get_shape(inputs[0])
        dim_ = get_num_dims(inputs[0])
        spatial_shape_ = shape_[-dim_:]

        resized = []
        for item in inputs:
            shape = get_shape(item)
            dim = get_num_dims(item)
            spatial_shape = shape[-dim:]
            if dim > 0 and spatial_shape != tuple([1]*dim) and spatial_shape != spatial_shape_:
                item = Crop(inputs[0])(item)
            resized.append(item)
        return resized



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

    def __init__(self, mode='b', shape=None, scale_factor=None, **kwargs):
        super().__init__()
        self.shape, self.scale_factor = shape, scale_factor

        if mode in self.MODES:
            mode = self.MODES[mode]
        self.mode = mode
        self.align_corners = True if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, mode=self.mode, size=self.shape, scale_factor=self.scale_factor,
                             align_corners=self.align_corners, **self.kwargs)

    def extra_repr(self):
        """ Report interpolation mode and factor for a repr. """
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.shape)
        info += ', mode=' + self.mode
        return info


class PixelShuffle(nn.PixelShuffle):
    """ Resize input tensor with depth to space operation. """

class SubPixelConv(PixelShuffle):
    """ An alias for PixelShuffle. """
    pass



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
