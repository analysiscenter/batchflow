""" Contains common layers """
import numpy as np
import tensorflow as tf


def flatten2d(inputs, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) """
    x = tf.convert_to_tensor(inputs)
    dims = tf.reduce_prod(tf.shape(x)[1:])
    x = tf.reshape(x, [-1, dims], name=name)
    return x


def flatten(inputs, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) using inferred shape and numpy """
    x = tf.convert_to_tensor(inputs)
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim], name=name)
    return x


def alpha_dropout(inputs, rate=0.5, noise_shape=None, seed=None, training=False, name=None):
    """ Alpha dropout layer

    Alpha Dropout is a dropout that maintains the self-normalizing property.
    For an input with zero mean and unit standard deviation, the output of Alpha Dropout maintains
    the original mean and standard deviation of the input.

    Klambauer G. et al "`Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_"
    """
    def _dropped_inputs():
        return tf.contrib.nn.alpha_dropout(inputs, 1-rate, noise_shape=noise_shape, seed=seed)
    return tf.cond(training, _dropped_inputs, lambda: tf.identity(inputs), name=name)


def maxout(inputs, depth, axis=-1, name='max'):
    """ Shrink last dimension by making max pooling every ``depth`` channels """
    with tf.name_scope(name):
        x = tf.convert_to_tensor(inputs)

        shape = x.get_shape().as_list()
        shape[axis] = -1
        shape += [depth]
        for i, _ in enumerate(shape):
            if shape[i] is None:
                shape[i] = tf.shape(x)[i]

        out = tf.reduce_max(tf.reshape(x, shape), axis=-1, keep_dims=False)
        return out


_REDUCE_OP = {
    'max': tf.reduce_max,
    'mean': tf.reduce_mean,
    'sum': tf.reduce_sum,
}

def xip(inputs, depth, op='max', data_format='channels_last', name='xip'):
    """ Shrink the channels dimension with reduce ``op`` every ``depth`` channels """
    reduce_op = _REDUCE_OP[op]

    with tf.name_scope(name):
        x = tf.convert_to_tensor(inputs)

        axis = -1 if data_format == 'channels_last' else 1
        num_layers = x.get_shape().as_list()[axis]
        split_sizes = [depth] * (num_layers // depth)
        if num_layers % depth:
            split_sizes += [num_layers % depth]

        xips = [reduce_op(split, axis=axis) for split in tf.split(x, split_sizes, axis=axis)]
        xips = tf.stack(xips, axis=axis)

    return xips

def mip(inputs, depth, data_format='channels_last', name='mip'):
    """ Maximum intensity projection by shrinking the channels dimension with max pooling every ``depth`` channels """
    return xip(inputs, depth, 'max', data_format, name)


def resize_bilinear_additive(inputs, factor=2, name=None, data_format='channels_last', **kwargs):
    """ Resize input tensor

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
    shape = inputs.shape
    shape = shape[1:-1] if data_format == 'channels_last' else shape[2:]
    shape = list(np.array(shape) * factor)

    with tf.variable_scope(name):
        x = tf.image.resize_bilinear(inputs, size=shape, name='resize', **kwargs)
        x = xip(x, depth=factor**2, op='sum', name='addition')
    return x
