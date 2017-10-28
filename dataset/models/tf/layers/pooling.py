""" Contains pooling layers """
import tensorflow as tf


def max_pooling(dim, inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None):
    """ Multi-dimensional max-pooling layer.

    For more detailes see tensorflow documentation:
    https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d
    https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d
    https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d

    Parameters
    ----------
    dim: int {1, 2, 3}
    inputs: tf.Tensor
    pool_size: int
    strides: int
    padding: str {'same', 'valid'}
    data_format: str {'channels_last', 'channels_first'}
    name: str

    Returns
    -------
    tf.Tensor
    """
    if dim == 1:
        return tf.layers.max_pooling1d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 2:
        return tf.layers.max_pooling2d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 3:
        return tf.layers.max_pooling3d(inputs, pool_size, strides, padding, data_format, name)
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)


def average_pooling(dim, inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None):
    """ Multi-dimensional average-pooling layer.

    For more detailes see tensorflow documentation:
    https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d
    https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling2d
    https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling3d

    Parameters
    ----------
    dim: int {1, 2, 3}
    inputs: tf.Tensor
    pool_size: int
    strides: int
    padding: str {'same', 'valid'}
    data_format: str {'channels_last', 'channels_first'}
    name: str

    Returns
    -------
    tf.Tensor
    """
    if dim == 1:
        return tf.layers.average_pooling1d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 2:
        return tf.layers.average_pooling2d(inputs, pool_size, strides, padding, data_format, name)
    elif dim == 3:
        return tf.layers.average_pooling3d(inputs, pool_size, strides, padding, data_format, name)
    else:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)
