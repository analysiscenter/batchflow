""" Contains common layers """
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, BatchNormalization

from .layer import Layer
from .utils import add_as_function



@add_as_function
class Flatten2D:
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, inputs):
        x = tf.convert_to_tensor(inputs)
        dims = tf.reduce_prod(tf.shape(x)[1:])
        x = tf.reshape(x, [-1, dims], **self.kwargs)
        return x



@add_as_function
class Flatten:
    """ Flatten tensor to two dimensions (batch_size, item_vector_size) using inferred shape and numpy """
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        x = tf.convert_to_tensor(inputs)
        shape = x.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(x, [-1, dim], **self.kwargs)
        return x



@add_as_function
class Dense(Layer):
    """ Wrapper for fully-connected layer. """
    def __init__(self, units, **kwargs):
        self.units = units
        self.kwargs = kwargs

    def __call__(self, inputs):
        if inputs.shape.ndims > 2:
            inputs = Flatten()(inputs)
        return Dense(**self.params_dict, **self.kwargs)(inputs)



class Dropout(Layer):
    """ Wrapper for dropout layer. """
    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        return Dropout(rate=self.dropout_rate, **self.kwargs)(inputs, training)



class AlphaDropout(Layer):
    """ Wrapper for self-normalizing dropout layer. """
    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        return AlphaDropout(rate=self.dropout_rate, **self.kwargs)(inputs, training)



class BatchNormalization(Layer):
    """ Wrapper for batch normalization layer. """
    def __init__(self, data_format='channels_last', **kwargs):
        self.data_format = data_format
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        axis = -1 if self.data_format == 'channels_last' else 1
        return BatchNormalization(fused=True, axis=axis, **self.kwargs)(inputs, training)



@add_as_function
class Maxout:
    """ Shrink last dimension by making max pooling every ``depth`` channels """
    def __init__(self, depth, axis=-1, name='max', *args, **kwargs):
        self.depth, self.axis = depth, axis
        self.name = name
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = tf.convert_to_tensor(inputs)

            shape = x.get_shape().as_list()
            shape[self.axis] = -1
            shape += [self.depth]
            for i, _ in enumerate(shape):
                if shape[i] is None:
                    shape[i] = tf.shape(x)[i]

            out = tf.reduce_max(tf.reshape(x, shape), axis=-1, keep_dims=False)
            return out



@add_as_function
class Xip:
    """ Shrink the channels dimension with reduce ``op`` every ``depth`` channels """
    REDUCE_OP = {
        'max': tf.reduce_max,
        'mean': tf.reduce_mean,
        'sum': tf.reduce_sum,
    }

    def __init__(self, depth, reduction='max', data_format='channels_last', name='max'):
        self.depth, self.reduction, self.data_format = depth, reduction, data_format
        self.name = name

    def __call__(self, inputs):
        reduce_op = self.REDUCE_OP[self.reduction]

        with tf.name_scope(self.name):
            x = tf.convert_to_tensor(inputs)

            axis = -1 if self.data_format == 'channels_last' else 1
            num_layers = x.get_shape().as_list()[axis]
            split_sizes = [self.depth] * (num_layers // self.depth)
            if num_layers % self.depth:
                split_sizes += [num_layers % self.depth]

            xips = [reduce_op(split, axis=axis) for split in tf.split(x, split_sizes, axis=axis)]
            xips = tf.stack(xips, axis=axis)

        return xips



@add_as_function
class Mip(Layer):
    """ Maximum intensity projection by shrinking the channels dimension with max pooling every ``depth`` channels """
    def __init__(self, depth, data_format='channels_last', name='max'):
        self.depth, self.data_format = depth, data_format
        self.name = name

    def __call__(self, inputs):
        return Xip(self.depth, 'max', self.data_format, self.name)(inputs)
