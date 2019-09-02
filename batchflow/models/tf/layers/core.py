""" Contains common layers """
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as K #pylint: disable=import-error

from .layer import Layer, add_as_function



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
        return K.Dense(**self.params_dict, **self.kwargs)(inputs)



class Dropout(Layer):
    """ Wrapper for dropout layer.

    Parameters
    ----------
    dropout_prob : float
        Fraction of the input units to drop.
    multisample: bool, number, sequence
        If evaluates to True, then batch is split into multiple parts,
        dropout applied to each of them separately and then parts are concatenated back.

        If True, then batch is split evenly into two parts.
        If integer, then batch is split evenly into that number of parts; must be a divisor of batch size.
        If float, then batch is split into parts of `multisample` and `1 - multisample` sizes.
        If sequence of ints, then batch is split into parts of given sizes. Must sum up to the batch size.
        If sequence of floats, then each float means proportion of sizes in batch and must sum up to 1.
    """
    def __init__(self, dropout_rate, multisample=False, **kwargs):
        self.dropout_rate = dropout_rate
        self.multisample = multisample
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        #pylint: disable=singleton-comparison
        d_layer = K.Dropout(rate=self.dropout_rate, **self.kwargs)

        if self.multisample != False:
            if self.multisample == True:
                self.multisample = 2
            elif isinstance(self.multisample, float):
                self.multisample = [self.multisample, 1 - self.multisample]

            if isinstance(self.multisample, int):
                sizes = self.multisample
            elif isinstance(self.multisample, (tuple, list)):
                if all([isinstance(item, int) for item in self.multisample]):
                    sizes = self.multisample
                elif all([isinstance(item, float) for item in self.multisample]):
                    batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
                    sizes = tf.convert_to_tensor([batch_size*item for item in self.multisample])
                    sizes = tf.cast(tf.math.round(sizes), dtype=tf.int32)
            elif isinstance(self.multisample, tf.Tensor):
                sizes = self.multisample

            splitted = tf.split(inputs, sizes, axis=0, name='mdropout_split')
            dropped = [d_layer(branch, training) for branch in splitted]
            output = tf.concat(dropped, axis=0, name='mdropout_concat')
        else:
            output = d_layer(inputs, training)
        return output



class AlphaDropout(Layer):
    """ Wrapper for self-normalizing dropout layer. """
    def __init__(self, dropout_rate, **kwargs):
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        return K.AlphaDropout(rate=self.dropout_rate, **self.kwargs)(inputs, training)



class BatchNormalization(Layer):
    """ Wrapper for batch normalization layer.

    Note that Keras layers does not add update operations to `UPDATE_OPS` collection,
    so we must do it manually.
    """
    def __init__(self, data_format='channels_last', **kwargs):
        self.data_format = data_format
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        axis = -1 if self.data_format == 'channels_last' else 1
        bn_layer = K.BatchNormalization(fused=True, axis=axis, **self.kwargs)
        output = bn_layer(inputs, training)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn_layer.updates[0])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn_layer.updates[1])
        return output



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
