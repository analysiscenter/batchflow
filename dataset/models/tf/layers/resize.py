""" Contains upsampling and resize layers """
import numpy as np
import tensorflow as tf

from .core import xip

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
    shape = list(np.array(shape, dtype=np.int32))
    return shape

def _dynamic_calc_shape(inputs, factor, data_format):
    shape = tf.cast(tf.shape(inputs), dtype=tf.float32)
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = shape * np.asarray(factor)
    shape = tf.cast(shape, dtype=tf.int32)
    return shape


def subpixel_conv(inputs, factor=2, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor with subpixel convolution (depth to space operation)

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2
    if dim == 3:
        dafo = 'NDHWC' if data_format == 'channels_last' else 'NCDHW'
    else:
        dafo = 'NHWC' if data_format == 'channels_last' else 'NCHW'

    _, channels = _calc_size(inputs, factor, data_format)

    with tf.variable_scope(name):
        x = conv(inputs, filters=channels*factor**dim, kernel_size=1, name='conv', **kwargs)
        x = tf.depth_to_space(x, block_size=factor, name='d2s', data_format=dafo)
    return x


def resize_bilinear_additive(inputs, factor=2, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear additive technique

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2
    size, channels = _calc_size(inputs, factor, data_format)
    layout = kwargs.get('layout', 'c')
    with tf.variable_scope(name):
        x = inputs
        if data_format == 'channels_first':
            x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.image.resize_bilinear(x, size=size, name='resize')
        if data_format == 'channels_first':
            x = tf.transpose(x, [0, 3, 1, 2])
        x = conv(x, layout, filters=channels*factor**dim, kernel_size=1, name='conv', **kwargs)
        x = xip(x, depth=factor**dim, reduction='sum', name='addition')
    return x

def resize_bilinear_1d(inputs, size, name, **kwargs):
    x = tf.expand_dims(inputs, axis=1)
    size = tf.concat([[1], size], axis=-1)
    x = tf.image.resize_bilinear(x, size=size, name='resize', **kwargs)
    x = tf.squeeze(x, [1])
    return x

def resize_bilinear_3d(tensor, size, name, **kwargs):
    tensor = _resize_along_axis(tensor, size, name, 2, **kwargs)
    tensor = _resize_except_axis(tensor, size, name, 2, **kwargs)
    return tensor

def _resize_along_axis(inputs, size, name, axis, **kwargs):
    except_axis = (axis + 1) % 3
    not_resized_axis = (axis + 2) % 3
    size, _ = _calc_size_after_resize(inputs, size, axis)
    output = _resize_except_axis(inputs, size, name, except_axis, **kwargs)
    return output

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

def _resize_except_axis(inputs, size, name, axis, **kwargs):
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
    
    real_size, static_shape = _calc_size_after_resize(x, size, [0,1])
    real_size = size[:-1]
    array = tf.TensorArray(tf.float32, size=tf.shape(x)[-2])
    sl = [slice(None)] * 5
    
    def _loop(idx, array):
        sl[-2] = idx
        tensor = x[sl]
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


def resize_bilinear(inputs, factor=2, name='resize', data_format='channels_last', **kwargs):
    """ Resize input tensor with bilinear method

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : float
        upsampling factor
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    size, _ = _calc_size(inputs, factor, data_format)
    with tf.variable_scope(name):
        x = inputs
        if data_format == 'channels_first':
            perm = [0] + list(range(2, inputs.shape.ndims)) + [1]
            perm_reverse = [0, inputs.shape.ndims-1] + list(range(1, inputs.shape.ndims-1))
        else:
            perm = list(range(inputs.shape.ndims))
            perm_reverse = perm
        x = tf.transpose(x, perm)
        dim = inputs.shape.ndims - 2
        if dim == 1:
            x = resize_bilinear_1d(x, size=size, name='resize_1d', **kwargs)
        elif dim == 2:
            x = tf.image.resize_bilinear(x, size=size, name='resize_2d', **kwargs)
        elif dim == 3:
            x = resize_bilinear_3d(x, size=size, name='resize_3d', **kwargs)
        x = tf.transpose(x, perm_reverse)
    return x


def resize_nn(inputs, factor=2, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor with nearest neighbors method

    Parameters
    ----------
    inputs : tf.Tensor
        a tensor to resize
    factor : int
        upsampling factor
    name : str
        scope name
    data_format : {'channels_last', 'channels_first'}
        position of the channels dimension

    Returns
    -------
    tf.Tensor
    """
    size, _ = _calc_size(inputs, factor, data_format)
    return tf.image.resize_nearest_neighbor(inputs, size=size, name=name, **kwargs)
