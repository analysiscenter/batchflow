""" Contains common layers """
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as K # pylint: disable=import-error

from .layer import Layer, add_as_function
from ..utils import get_num_channels, get_num_dims



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
class Activation(Layer):
    """ Wrapper for activation functions.
    Used for `a` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, activation, **kwargs):
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        if self.activation is not None:
            return self.activation(inputs)
        return inputs



@add_as_function
class Dense(Layer):
    """ Wrapper for fully-connected layer.
    Used for `f` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.kwargs = kwargs

    def __call__(self, inputs):
        if inputs.shape.ndims > 2:
            inputs = Flatten()(inputs)
        return K.Dense(**self.params_dict, **self.kwargs)(inputs)



@add_as_function
class Combine(Layer):
    """ Combine inputs into one tensor via various transformations.
    Used for `.`, `+`, `*`, `&` letters in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    op : str
        Which operation to use for combining tensors.
        If 'concat', inputs are concated along channels axis.
        If one of 'avg', 'mean', takes average of inputs.
        If one of 'sum', 'add', inputs are summed.
        If one of 'softsum', 'convsum', every tensor is passed through 1x1 convolution in order to have
        the same number of channels as the first tensor, and then summed.
        If one of 'attention', 'gau', then the first tensor is used to create multiplicative attention for second,
        and the output is added to second tensor.

    data_format : str {'channels_last', 'channels_first'}
        Data format.
    leading_index : int
        Index of tensor to broadcast to. Allows to change the order of input tensors.
    force_resize : None or bool
        Whether to crop every tensor to the leading one.
    kwargs : dict
        Arguments for :class:`.ConvBlock`.
    """
    @staticmethod
    def concat(inputs, data_format='channels_last', axis=None, **kwargs):
        """ Concatenate tensors along specified axis or data format. """
        _ = kwargs
        axis = axis or (1 if data_format == "channels_first" or data_format.startswith("NC") else -1)
        return tf.concat(inputs, axis=axis, name='combine-concat')

    @staticmethod
    def sum(inputs, **kwargs):
        """ Addition with broadcasting. """
        _ = kwargs
        result = 0
        for item in inputs:
            result = result + item
        return result

    @staticmethod
    def mul(inputs, **kwargs):
        """ Multiplication with broadcasting. """
        _ = kwargs
        result = 1
        for item in inputs:
            result = result * item
        return result

    @staticmethod
    def mean(inputs, **kwargs):
        """ Mean with broadcasting. """
        _ = kwargs
        return Combine.sum(inputs, name='combine-softsum') / len(inputs)

    @staticmethod
    def softsum(inputs, data_format='channels_last', **kwargs):
        """ Softsum. """
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        axis = 1 if data_format == "channels_first" or data_format.startswith("NC") else -1
        filters = inputs[0].get_shape().as_list()[axis]
        args = {'layout': 'c', 'filters': filters, 'kernel_size': 1, **kwargs}
        for i in range(1, len(inputs)):
            inputs[i] = ConvBlock(name='combine-conv', **args)(inputs[i])
        return Combine.sum(inputs, name='combine-softsum')


    @staticmethod
    def attention(inputs, **kwargs):
        """ Attention combine module.
        Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang. Pyramid Attention Network
        for Semantic Segmentation <https://arxiv.org/abs/1805.10180>'_"
        """
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        x, skip = inputs[0], inputs[1]
        num_channels = get_num_channels(skip)
        num_dims = get_num_dims(skip) + 1
        conv1 = ConvBlock(layout='cna', kernel_size=3, filters=num_channels, name='conv-x', **kwargs)(x)
        conv2 = ConvBlock(layout='V > cna', kernel_size=1, filters=num_channels, dim=num_dims,
                          name='conv-skip', **kwargs)(skip)
        weighted = Combine.mul([conv1, conv2])
        return Combine.sum([weighted, skip])

    @staticmethod
    def gau(inputs, filters=None, **kwargs):
        """ Global Attention Upsample module. """
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        from .resize import Crop
        x, skip = inputs[0], inputs[1]
        num_dims = get_num_dims(skip) + 1
        leaky_relu = partial(tf.nn.leaky_relu, alpha=0.1)
        conv1 = ConvBlock(layout='ca', filters=filters, activation=leaky_relu, name='conv-low', **kwargs)(x)
        conv2 = ConvBlock(layout='Vfna >', units=filters, activation=leaky_relu, dim=num_dims,
                          name='conv-high', **kwargs)(skip)
        gated = Combine.mul([conv1, conv2])

        gated = ConvBlock(layout='caN', filters=filters, activation=leaky_relu, name='conv-gated')(gated)
        clamped = ConvBlock(layout='ca', filters=filters, activation=leaky_relu, name='conv-clamped', **kwargs)(skip)
        gated = Crop(resize_to=clamped)(gated)
        return Combine.sum([gated, clamped])

    OPS = {
        concat: ['concat', 'cat', '|'],
        sum: ['sum', 'plus', '+'],
        mul: ['multi', 'mul', '*'],
        mean: ['avg', 'mean', 'average'],
        softsum: ['softsum', '&'],
        attention: ['attention'],
        gau: ['gau'],
    }
    OPS = {alias: getattr(method, '__func__') for method, aliases in OPS.items() for alias in aliases}

    def __init__(self, op='softsum', data_format='channels_last', leading_index=0, force_resize=None,
                 name='combine', **kwargs):
        self.op = op
        self.data_format, self.name, self.idx, self.force_resize = data_format, name, leading_index, force_resize
        self.kwargs = kwargs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            if self.idx != 0:
                inputs[0], inputs[self.idx] = inputs[self.idx], inputs[0]

            if self.op in self.OPS:
                op = self.OPS[self.op]
            elif callable(self.op):
                op = self.op
            else:
                raise ValueError('Combine operation must be a callable or \
                                  one from {}, instead got {}.'.format(list(self.OPS.keys()), op))

            if self.force_resize is None:
                if op.__name__ in ['gau', 'softsum']:
                    force_resize = False
                else:
                    force_resize = True
            if force_resize:
                from .resize import Crop
                inputs = [Crop(resize_to=inputs[-1], data_format=self.data_format)(item) for item in inputs]

            return op(inputs, data_format=self.data_format, **self.kwargs)




class BaseDropout(Layer):
    """ Base class for dropout layers.

    Parameters
    ----------
    dropout_rate : float, tf.Tensor, callable
        If float or Tensor, then fraction of the input units to drop.
        If callable, then function to be called on `global_step`. Must return tensor of size 1.

    multisample: bool, number, sequence, tf.Tensor
        If evaluates to True, then either multiple dropout applied to the whole batch and then averaged, or
        batch is split into multiple parts, each passed through dropout and then concatenated back.

        If True, then two different dropouts are applied to whole batch.
        If integer, then that number of different dropouts are applied to whole batch.
        If float, then batch is split into parts of `multisample` and `1 - multisample` sizes.
        If sequence of ints, then batch is split into parts of given sizes. Must sum up to the batch size.
        If sequence of floats, then each float means proportion of sizes in batch and must sum up to 1.
        If Tensor, then it is used as the second parameter for splitting function, see
        `tf.split <https://www.tensorflow.org/api_docs/python/tf/split>`_,.
    """
    def __init__(self, dropout_rate, multisample=False, global_step=None, **kwargs):
        self.dropout_rate = dropout_rate
        self.global_step = global_step
        self.multisample = multisample
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        if callable(self.dropout_rate):
            step = tf.cast(self.global_step, dtype=tf.float32)
            self.dropout_rate = self.dropout_rate(step)
        d_layer = self.LAYER(rate=self.dropout_rate)

        if self.multisample is not False:
            if self.multisample is True:
                self.multisample = 2
            elif isinstance(self.multisample, float):
                self.multisample = [self.multisample, 1 - self.multisample]

            if isinstance(self.multisample, int): # dropout to the whole batch, then average
                dropped = [d_layer(inputs, training) for _ in range(self.multisample)]
                output = Combine(op='avg', **self.kwargs)(dropped)
            else: # split batch into separate-dropout branches
                if isinstance(self.multisample, (tuple, list)):
                    if all([isinstance(item, int) for item in self.multisample]):
                        sizes = self.multisample
                    elif all(isinstance(item, float) for item in self.multisample):
                        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
                        sizes = tf.convert_to_tensor([batch_size*item for item in self.multisample[:-1]])
                        sizes = tf.cast(tf.math.round(sizes), dtype=tf.int32)
                        residual = tf.convert_to_tensor(tf.shape(inputs)[0] - tf.reduce_sum(sizes))
                        residual = tf.reshape(residual, shape=(1,))
                        sizes = tf.concat([sizes, residual], axis=0)
                else: # case of Tensor
                    sizes = self.multisample

                splitted = tf.split(inputs, sizes, axis=0, name='mdropout_split')
                dropped = [d_layer(branch, training) for branch in splitted]
                output = tf.concat(dropped, axis=0, name='mdropout_concat')
        else:
            output = d_layer(inputs, training)
        return output


class Dropout(BaseDropout):
    """ Wrapper for dropout layer.
    Used for `d` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    LAYER = K.Dropout


class AlphaDropout(BaseDropout):
    """ Wrapper for self-normalizing dropout layer.
    Used for `D` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    LAYER = K.AlphaDropout



class BatchNormalization(Layer):
    """ Wrapper for batch normalization layer.
    Used for `n` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

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
    """ Maximum intensity projection by shrinking the channels dimension with max pooling every ``depth`` channels.
    Used for `m` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, depth, data_format='channels_last', name='max'):
        self.depth, self.data_format = depth, data_format
        self.name = name

    def __call__(self, inputs):
        return Xip(self.depth, 'max', self.data_format, self.name)(inputs)
