""" Contains common layers """
import numpy as np
import tensorflow as tf


def flatten2d(x, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) """
    x = tf.convert_to_tensor(x)
    dims = tf.reduce_prod(tf.shape(x)[1:])
    x = tf.reshape(x, [-1, dims], name=name)
    return x


def flatten(x, name=None):
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) using inferred shape and numpy """
    x = tf.convert_to_tensor(x)
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim], name=name)
    return x


def maxout(x, depth, axis=-1, name='max'):
    """ Shrink last dimension by making max pooling every ``depth`` channels """
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x)

        shape = x.get_shape().as_list()
        shape[axis] = -1
        shape += [depth]
        for i, _ in enumerate(shape):
            if shape[i] is None:
                shape[i] = tf.shape(x)[i]

        out = tf.reduce_max(tf.reshape(x, shape), axis=-1, keep_dims=False)
        return out

def mip(x, depth, name='mip'):
    """ Shrink last dimension by making max pooling every ``depth`` channels """
    with tf.name_scope(name):
        x = tf.convert_to_tensor(x)
        num_layers = x.get_shape().as_list()[-1]
        split_sizes = [depth] * (num_layers // depth)
        if num_layers % depth:
            split_sizes += [num_layers % depth]

        splits = tf.split(x, split_sizes, axis=-1)
        mips = []
        for split in splits:
            amip = tf.reduce_max(split, axis=-1)
            mips.append(amip)
        mips = tf.stack(mips, axis=-1)

    return mips
