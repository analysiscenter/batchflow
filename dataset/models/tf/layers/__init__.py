""" Custom tf layers and operations """
import tensorflow as tf

from .conv import conv1d_block, conv2d_block, conv3d_block


def flatten(x, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) """
    dims = tf.reduce_prod(tf.shape(x)[1:])
    x = tf.reshape(x, [-1, dims], name=name)
    return x


def iflatten(x, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) using inferred shape and numpy """
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim], name=name)
    return x
