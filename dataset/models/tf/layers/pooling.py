""" Contains pooling layers """
import numpy as np
import tensorflow as tf


def max_pooling(inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None):
    """ Multi-dimensional max-pooling layer.

    Parameters
    ----------
    inputs: tf.Tensor
        input tensor
    pool_size: int
        the size of the pooling window
    strides: int
        the strides of the pooling operation
    padding: str
        'same' or 'valid'
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        scope name

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d>`_,
    `tf.layers.max_pooling2d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d>`_,
    `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d>`_
    """
    dim = inputs.shape.ndims - 2
    if dim == 1:
        return tf.layers.max_pooling1d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 2:
        return tf.layers.max_pooling2d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 3:
        return tf.layers.max_pooling3d(inputs, pool_size, strides, padding, data_format, name)
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)


def average_pooling(inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None):
    """ Multi-dimensional average-pooling layer.

    Parameters
    ----------
    inputs: tf.Tensor
        input tensor
    pool_size: int
        the size of the pooling window
    strides: int
        the strides of the pooling operation
    padding: str
        'same' or 'valid'
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        scope name

    Returns
    -------
    tf.Tensor

    See also
    --------
    `tf.layers.average_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d>`_,
    `tf.layers.average_pooling2d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling2d>`_,
    `tf.layers.average_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling3d>`_
    """
    dim = inputs.shape.ndims - 2
    if dim == 1:
        return tf.layers.average_pooling1d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 2:
        return tf.layers.average_pooling2d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 3:
        return tf.layers.average_pooling3d(inputs, pool_size, strides, padding, data_format, name)
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)


def global_average_pooling(inputs, data_format='channels_last', name=None):
    """ Multi-dimensional global average-pooling layer.

    Parameters
    ----------
    inputs: tf.Tensor
        input tensor
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        scope name

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2
    axis = 1 if data_format == 'channels_last' else 2
    if dim == 2:
        axis = [axis, axis+1]
    elif dim == 3:
        axis = [axis, axis+1, axis+2]
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

    return tf.reduce_mean(inputs, axis=axis, name=name)


def global_max_pooling(inputs, data_format='channels_last', name=None):
    """ Multi-dimensional global max-pooling layer.

    Parameters
    ----------
    inputs: tf.Tensor
        input tensor
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        scope name

    Returns
    -------
    tf.Tensor
    """
    dim = inputs.shape.ndims - 2
    axis = 1 if data_format == 'channels_last' else 2
    if dim == 1:
        pass
    elif dim == 2:
        axis = [axis, axis+1]
    elif dim == 3:
        axis = [axis, axis+1, axis+2]
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

    return tf.reduce_max(inputs, axis=axis, name=name)


def fractional_max_pooling(inputs, pooling_ratio, pseudo_random=False, overlapping=False,
                           data_format='channels_last', **kwargs):
    """ Fractional max-pooling layer.

    Parameters
    ----------
    inputs: tf.Tensor
        input tensor
    pooling_ratio : float
        pooling ratio
    pseudo_random : bool
        Default is False
    overlapping : bool
        Default is False
    name: str
        scope name

    Returns
    -------
    tf.Tensor

    Notes
    -----
    Be aware that it is not thread safe.
    ``tf.nn.fractional_max_pool>`` will likely cause segmentation fault in a multi-threading environment
    (e.g. in a pipeline with prefetch)
    """
    dim = inputs.shape.ndims - 2

    _pooling_ratio = np.ones(inputs.shape.ndims)
    axis = 1 if data_format == 'channels_last' else 2
    _pooling_ratio[axis:axis+dim] = pooling_ratio
    _pooling_ratio = list(_pooling_ratio)

    if dim == 1:
        with tf.variable_scope(kwargs.get('name') or 'fractional_max_pooling'):
            axis = 2 if data_format == 'channels_last' else -1
            x = tf.expand_dims(inputs, axis=axis)
            _pooling_ratio[axis] = 1
            x, _, _ = tf.nn.fractional_max_pool(x, _pooling_ratio, pseudo_random, overlapping, **kwargs)
            x = tf.squeeze(x, [axis])
    elif dim == 2:
        x, _, _ = tf.nn.fractional_max_pool(inputs, _pooling_ratio, pseudo_random, overlapping, **kwargs)
    else:
        raise ValueError("Number of dimensions in the inputs tensor should be 1 or 2, but given %d" % dim)

    return x
