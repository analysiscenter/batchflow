""" Contains upsampling and resize layers """
import numpy as np
import tensorflow.compat.v1 as tf

from .layer import Layer, add_as_function
from .conv import ConvTranspose
from .core import Xip
from ..utils import get_shape, get_num_channels, get_spatial_shape



class IncreaseDim(Layer):
    """ Increase dimensionality of passed tensor by desired amount.
    Used for `>` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, dim=1, insert=True, name='increase_dim', **kwargs):
        self.dim, self.insert = dim, insert
        self.name, self.kwargs = name, kwargs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            shape = get_shape(inputs)
            ones = [1] * self.dim
            if self.insert:
                return tf.reshape(inputs, (-1, *ones, *shape))
            return tf.reshape(inputs, (-1, *shape, *ones))


class Reshape(Layer):
    """ Enforce desired shape of tensor.
    Used for `r` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, reshape_to=None, name='reshape', **kwargs):
        self.reshape_to = reshape_to
        self.name, self.kwargs = name, kwargs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            return tf.reshape(inputs, (-1, *self.reshape_to))


class Crop:
    """ Crop input tensor to a shape of a given image.
    If resize_to does not have a fully defined shape (resize_to.get_shape() has at least one None),
    the returned tf.Tensor will be of unknown shape except the number of channels.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor.
    resize_to : tf.Tensor
        Tensor which shape the inputs should be resized to.
    data_format : str {'channels_last', 'channels_first'}
        Data format.
    """
    def __init__(self, resize_to, data_format='channels_last', name='crop'):
        self.resize_to = resize_to
        self.data_format, self.name = data_format, name

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            static_shape = get_spatial_shape(self.resize_to, self.data_format, False)
            dynamic_shape = get_spatial_shape(self.resize_to, self.data_format, True)

            if None in get_shape(inputs) + static_shape:
                return self._dynamic_crop(inputs, static_shape, dynamic_shape, self.data_format)
            return self._static_crop(inputs, static_shape, self.data_format)

    def _static_crop(self, inputs, shape, data_format='channels_last'):
        input_shape = np.array(get_spatial_shape(inputs, data_format))

        if np.abs(input_shape - shape).sum() > 0:
            begin = [0] * inputs.shape.ndims
            if data_format == "channels_last":
                size = [-1] + shape + [-1]
            else:
                size = [-1, -1] + shape
            x = tf.slice(inputs, begin=begin, size=size)
        else:
            x = inputs
        return x

    def _dynamic_crop(self, inputs, static_shape, dynamic_shape, data_format='channels_last'):
        input_shape = get_spatial_shape(inputs, data_format, True)
        n_channels = get_num_channels(inputs, data_format)
        if data_format == 'channels_last':
            slice_size = [(-1,), dynamic_shape, (n_channels,)]
            output_shape = [None] * (len(static_shape) + 1) + [n_channels]
        else:
            slice_size = [(-1, n_channels), dynamic_shape]
            output_shape = [None, n_channels] + [None] * len(static_shape)

        begin = [0] * len(inputs.get_shape().as_list())
        size = tf.concat(slice_size, axis=0)
        cond = tf.reduce_sum(tf.abs(input_shape - dynamic_shape)) > 0
        x = tf.cond(cond, lambda: tf.slice(inputs, begin=begin, size=size), lambda: inputs)
        x.set_shape(output_shape)
        return x



def _calc_size(inputs, factor, data_format):
    shape = inputs.get_shape().as_list()
    channels = shape[-1] if data_format == 'channels_last' else shape[1]
    if None in shape[1:]:
        shape = _dynamic_calc_shape(inputs, factor, data_format)
    else:
        shape = _static_calc_shape(inputs, factor, data_format)
    return shape, channels

def _static_calc_shape(inputs, factor, data_format):
    shape = inputs.get_shape().as_list()
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = np.asarray(shape) * np.asarray(factor)
    shape = list(np.ceil(shape).astype(np.int32))
    return shape

def _dynamic_calc_shape(inputs, factor, data_format):
    shape = tf.cast(tf.shape(inputs), dtype=tf.float32)
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = shape * np.asarray(factor)
    shape = tf.cast(tf.ceil(shape), dtype=tf.int32)
    return shape



class DepthToSpace(Layer):
    """ 1d, 2d and 3d depth_to_space transformation.

    Parameters
    ----------
    block_size : int
        An int that is >= 2. The size of the spatial block.
    name : str
        Scope name.
    data_format : {'channels_last', 'channels_first'}
        Position of the channels dimension.

    See also
    --------
    `tf.depth_to_space <https://www.tensorflow.org/api_docs/python/tf/depth_to_space>`_
    """
    def __init__(self, block_size, data_format='channels_last', **kwargs):
        self.block_size, self.data_format = block_size, data_format
        self.kwargs = kwargs

    def __call__(self, inputs):
        return depth_to_space(inputs, **self.params_dict, **self.kwargs)


def depth_to_space(inputs, block_size, data_format='channels_last', name='d2s'):
    """ 1d, 2d and 3d depth_to_space transformation. """
    dim = inputs.shape.ndims - 2
    if dim == 2:
        dafo = 'NHWC' if data_format == 'channels_last' else 'NCHW'
        return tf.depth_to_space(inputs, block_size, name, data_format=dafo)

    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0] + list(range(2, dim+2)) + [1])
    x = _depth_to_space(inputs, block_size, name)
    if data_format == 'channels_first':
        x = tf.transpose(x, [0, dim+1] + list(range(1, dim+1)))
    return x

def _depth_to_space(inputs, block_size, name='d2s'):
    dim = inputs.shape.ndims - 2

    with tf.variable_scope(name):
        shape = inputs.get_shape().as_list()[1:]
        channels = shape[-1]
        if channels % (block_size ** dim) != 0:
            raise ValueError('channels of the inputs must be divisible by block_size ** {}'.format(dim))
        output_shape = tf.concat([(tf.shape(inputs)[0],), tf.shape(inputs)[1:-1]*block_size,
                                  (tf.shape(inputs)[-1], )], axis=-1)
        slices = [np.arange(0, channels // (block_size ** dim)) + i
                  for i in range(0, channels, channels // (block_size ** dim))]
        tensors = []
        for i in range(block_size ** dim):
            zero_filter = np.zeros(block_size ** dim)
            selective_filter = np.zeros(block_size ** dim)
            selective_filter[i] = 1
            zero_filter = zero_filter.reshape([block_size] * dim)
            selective_filter = selective_filter.reshape([block_size] * dim)
            fltr = []
            for j in range(channels):
                _filter = [zero_filter] * channels
                _filter[j] = selective_filter
                _filter = np.stack(_filter, axis=-1)
                fltr.append(_filter)
            fltr = np.stack(fltr, axis=-1)
            fltr = np.transpose(fltr, axes=list(range(dim))+[dim, dim+1])
            fltr = tf.constant(fltr, tf.float32)
            x = ConvTranspose(fltr, output_shape, [1] + [block_size] * dim + [1])(inputs)
            if None in shape[:-1]:
                resized_shape = shape[:-1]
            else:
                resized_shape = list(np.array(shape[:-1]) * block_size)
            x.set_shape([None] + resized_shape + [channels/(block_size ** dim)])
            x = tf.gather(x, slices[i], axis=-1)
            tensors.append(x)
        x = tf.add_n(tensors)
    return x



class UpsamplingLayer(Layer):
    """ Parent for all the upsampling layers with the same parameters. """
    def __init__(self, factor=2, shape=None, data_format='channels_last', name='upsampling', **kwargs):
        self.factor, self.shape = factor, shape
        self.data_format = data_format
        self.name, self.kwargs = name, kwargs


class SubpixelConv(UpsamplingLayer):
    """ Resize input tensor with subpixel convolution (depth to space operation).
    Used for `X` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    factor : int
        Upsampling factor.
    layout : str
        Layers applied before depth-to-space transform.
    name : str
        Scope name.
    data_format : {'channels_last', 'channels_first'}
        Position of the channels dimension.
    """
    def __call__(self, inputs):
        return subpixel_conv(inputs, **self.params_dict, **self.kwargs)


def subpixel_conv(inputs, factor=2, name='subpixel', data_format='channels_last', **kwargs):
    """ Resize input tensor with subpixel convolution (depth to space operation. """
    dim = inputs.shape.ndims - 2

    _, channels = _calc_size(inputs, factor, data_format)
    layout = kwargs.pop('layout', 'cna')
    kwargs['filters'] = channels*factor**dim

    x = inputs
    with tf.variable_scope(name):
        if layout:
            from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
            x = ConvBlock(layout=layout, kernel_size=1, name='conv', data_format=data_format, **kwargs)(inputs)
        x = depth_to_space(x, block_size=factor, name='d2s', data_format=data_format)
    return x


class ResizeBilinearAdditive(UpsamplingLayer):
    """ Resize input tensor with bilinear additive technique.
    Used for `A` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    factor : int
        Upsampling factor.
    layout : str
        Layers applied between bilinear resize and xip.
    name : str
        Scope name.
    data_format : {'channels_last', 'channels_first'}
        Position of the channels dimension.
    """
    def __call__(self, inputs):
        return resize_bilinear_additive(inputs, **self.params_dict, **self.kwargs)

def resize_bilinear_additive(inputs, factor=2, name='bilinear_additive', data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear additive technique. """
    dim = inputs.shape.ndims - 2
    _, channels = _calc_size(inputs, factor, data_format)
    layout = kwargs.pop('layout', 'cna')
    with tf.variable_scope(name):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        x = resize_bilinear(inputs, factor, name=name, data_format=data_format, **kwargs)
        x = ConvBlock(layout=layout, filters=channels*factor**dim, kernel_size=1, name='conv', **kwargs)(x)
        x = Xip(depth=factor**dim, reduction='sum', name='addition')(x)
    return x

def resize_bilinear_1d(inputs, size, name='resize', **kwargs):
    """ Resize 1D input tensor with bilinear method.

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize.
    size : tf.Tensor or list
        size of the output image.
    name : str
        scope name.

    Returns
    -------
    tf.Tensor
    """
    x = tf.expand_dims(inputs, axis=1)
    size = tf.concat([[1], size], axis=-1)
    x = tf.image.resize_bilinear(x, size=size, name=name, **kwargs)
    x = tf.squeeze(x, [1])
    return x

def resize_bilinear_3d(tensor, size, name='resize', **kwargs):
    """ Resize 3D input tensor with bilinear method.

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize.
    size : tf.Tensor or list
        size of the output image.
    name : str
        scope name.

    Returns
    -------
    tf.Tensor
    """
    with tf.variable_scope(name):
        tensor = _resize_along_axis(tensor, size, 2, **kwargs)
        tensor = _resize_except_axis(tensor, size, 2, **kwargs)
    return tensor

def _resize_along_axis(inputs, size, axis, **kwargs):
    """ Resize 3D input tensor to size along just one axis. """
    except_axis = (axis + 1) % 3
    size, _ = _calc_size_after_resize(inputs, size, axis)
    output = _resize_except_axis(inputs, size, except_axis, **kwargs)
    return output

def _resize_except_axis(inputs, size, axis, **kwargs):
    """ Resize 3D input tensor to size except just one axis. """
    perm = np.arange(5)
    reverse_perm = np.arange(5)

    if axis == 0:
        spatial_perm = [2, 3, 1]
        reverse_spatial_perm = [3, 1, 2]
    elif axis == 1:
        spatial_perm = [1, 3, 2]
        reverse_spatial_perm = [1, 3, 2]
    else:
        spatial_perm = [1, 2, 3]
        reverse_spatial_perm = [1, 2, 3]

    perm[1:4] = spatial_perm
    reverse_perm[1:4] = reverse_spatial_perm
    x = tf.transpose(inputs, perm)

    if isinstance(size, tf.Tensor):
        size = tf.unstack(size)
        size = [size[i-1] for i in spatial_perm]
        size = tf.stack(size)
    else:
        size = [size[i-1] for i in spatial_perm]

    real_size, static_shape = _calc_size_after_resize(x, size, [0, 1])
    real_size = size[:-1]
    array = tf.TensorArray(tf.float32, size=tf.shape(x)[-2])
    partial_sl = [slice(None)] * 5

    def _loop(idx, array):
        partial_sl[-2] = idx
        tensor = x[partial_sl]
        tensor = tf.image.resize_bilinear(tensor, size=real_size, name='resize_2d', **kwargs)
        array = array.write(idx, tensor)
        return [idx+1, array]
    i = 0
    _, array = tf.while_loop(lambda i, array: i < tf.shape(x)[-2], _loop, [i, array])
    array = array.stack()
    array = tf.transpose(array, [1, 2, 3, 0, 4])
    array.set_shape(static_shape)
    array = tf.transpose(array, reverse_perm)
    return array

def _calc_size_after_resize(inputs, size, axis):
    if not isinstance(axis, list):
        axis = [axis]
    except_axis = list(set(range(3)) - set(axis))
    if isinstance(size, tf.Tensor):
        size = tf.unstack(size)
        for i in except_axis:
            size[i] = tf.shape(inputs)[i+1]
        size = tf.stack(size)
        static_size = [None] * 4 + [inputs.get_shape().as_list()[-1]]
    else:
        size = size[:]
        static_size = inputs.get_shape().as_list()
        if None in static_size[1:]:
            size[except_axis] = tf.shape(inputs)[except_axis+1]
            size = tf.stack(size)
        else:
            for i in except_axis:
                size[i] = static_size[i+1]
            static_size[1:4] = size
    return size, static_size


class ResizeBilinear(UpsamplingLayer):
    """ Resize input tensor with bilinear method.
    Used for `b` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    factor : float
        Upsampling factor (not used if shape is specified).
    shape : tuple of int
        Shape to upsample to.
    name : str
        Scope name.
    data_format : {'channels_last', 'channels_first'}
        Position of the channels dimension.
    """

    def __call__(self, inputs):
        return resize_bilinear(inputs, **self.params_dict, **self.kwargs)



def resize_bilinear(inputs, factor=2, shape=None, name='resize', data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear method. """
    if shape is None:
        shape, _ = _calc_size(inputs, factor, data_format)

    with tf.variable_scope(name):
        x = inputs
        if data_format == 'channels_first':
            perm = [0] + list(range(2, inputs.shape.ndims)) + [1]
            perm_reverse = [0, inputs.shape.ndims-1] + list(range(1, inputs.shape.ndims-1))
            x = tf.transpose(x, perm)
        dim = inputs.shape.ndims - 2
        if dim == 1:
            x = resize_bilinear_1d(x, size=shape, name='resize_1d', **kwargs)
        elif dim == 2:
            x = tf.image.resize_bilinear(x, size=shape, name='resize_2d', **kwargs)
        elif dim == 3:
            x = resize_bilinear_3d(x, size=shape, name='resize_3d', **kwargs)
        if data_format == 'channels_first':
            x = tf.transpose(x, perm_reverse)
    return x


class ResizeNn(UpsamplingLayer):
    """ Resize input tensor with nearest neighbors method.
    Used for `N` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    factor : int
        Upsampling factor (not used if shape is specified).
    shape : tuple of int
        Shape to upsample to.
    name : str
        Scope name.
    data_format : {'channels_last', 'channels_first'}
        Position of the channels dimension.
    """
    def __call__(self, inputs):
        return resize_nn(inputs, **self.params_dict, **self.kwargs)

def resize_nn(inputs, factor=2, shape=None, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor with nearest neighbors method. """
    dim = inputs.shape.ndims
    if dim != 4:
        raise ValueError("inputs must be Tensor of rank 4 but {} was given".format(dim))
    if shape is None:
        shape, _ = _calc_size(inputs, factor, data_format)
    return tf.image.resize_nearest_neighbor(inputs, size=shape, name=name, **kwargs)



@add_as_function
class Upsample:
    """ Upsample inputs with a given factor.

    Parameters
    ----------
    factor : int
        An upsamping scale
    shape : tuple of int
        Shape to upsample to (used by bilinear and NN resize)
    layout : str
        Resizing technique, a sequence of:

        - A - use residual connection with bilinear additive upsampling
        - b - bilinear resize
        - B - bilinear additive upsampling
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :class:`.ConvBlock` layers are also allowed.

    Examples
    --------
    A simple bilinear upsampling::

        x = upsample(shape=(256, 256), layout='b')(x)

    Upsampling with non-linear normalized transposed convolution::

        x = Upsample(factor=2, layout='nat', kernel_size=3)(x)

    Subpixel convolution with a residual bilinear additive connection::

        x = Upsample(factor=2, layout='AX+')(x)
    """
    def __init__(self, factor=None, shape=None, layout='b', name='upsample', **kwargs):
        self.factor, self.shape, self.layout = factor, shape, layout
        self.name, self.kwargs = name, kwargs

    def __call__(self, inputs, *args, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        if np.all(self.factor == 1):
            return inputs

        if self.kwargs.get('filters') is None:
            self.kwargs['filters'] = get_num_channels(inputs,
                                                      data_format=self.kwargs.get('data_format', 'channels_last'))

        if 't' in self.layout or 'T' in self.layout:
            if 'kernel_size' not in self.kwargs:
                self.kwargs['kernel_size'] = self.factor
            if 'strides' not in kwargs:
                self.kwargs['strides'] = self.factor

        return ConvBlock(layout=self.layout, factor=self.factor, shape=self.shape,
                         name=self.name, **self.kwargs)(inputs, *args, **kwargs)
