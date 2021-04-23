""" Contains pooling layers """
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as K #pylint: disable=import-error

from .layer import Layer, add_as_function



@add_as_function
class MaxPooling(Layer):
    """ Multi-dimensional max-pooling layer.

    Parameters
    ----------
    pool_size: int
        The size of the pooling window.
    strides: int
        The strides of the pooling operation.
    padding: str
        'same' or 'valid'
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        Scope name.

    See also
    --------
    `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/keras/layers/MaxPool1D>`__,
    `tf.layers.max_pooling2d <https://www.tensorflow.org/api_docs/python/keras/layers/MaxPool2D>`__,
    `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/keras/layers/MaxPool3D>`__.
    """
    LAYERS = {
        1: K.MaxPool1D,
        2: K.MaxPool2D,
        3: K.MaxPool3D
    }

    def __init__(self, pool_size, pool_strides, padding='same', data_format='channels_last', name=None, **kwargs):
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.padding, self.data_format = padding, data_format
        self.name = name
        self.kwargs = kwargs

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(self.pool_size, self.pool_strides,
                        self.padding, self.data_format, name=self.name, **self.kwargs)(inputs)



@add_as_function
class AveragePooling(Layer):
    """ Multi-dimensional average-pooling layer.

    Parameters
    ----------
    pool_size: int
        The size of the pooling window.
    strides: int
        The strides of the pooling operation.
    padding: str
        'same' or 'valid'
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        Scope name.

    See also
    --------
    `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/keras/layers/AveragePooling1D>`__,
    `tf.layers.max_pooling2d <https://www.tensorflow.org/api_docs/python/keras/layers/AveragePooling2D>`__,
    `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/keras/layers/AveragePooling3D>`__.
    """
    LAYERS = {
        1: K.AveragePooling1D,
        2: K.AveragePooling2D,
        3: K.AveragePooling3D
    }

    def __init__(self, pool_size, pool_strides, padding='same', data_format='channels_last', name=None, **kwargs):
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.padding, self.data_format = padding, data_format
        self.name = name
        self.kwargs = kwargs

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(self.pool_size, self.pool_strides,
                        self.padding, self.data_format, name=self.name, **self.kwargs)(inputs)



@add_as_function
class FractionalPooling:
    """ Fractional max-pooling layer.

    Parameters
    ----------
    pool_size : float
        pooling ratio (default=1.4142).
    pseudo_random : bool
        Default is False.
    overlapping : bool
        Default is False.
    strides: int
        The strides of the pooling operation.
    padding: str
        'same' or 'valid'
    data_format: str
        'channels_last' or 'channels_first'
    name: str
        Scope name.

    Notes
    -----
    Be aware that it is not thread safe.
    ``tf.nn.fractional_max_pool>`` will likely cause segmentation fault in a multi-threading environment
    (e.g. in a pipeline with prefetch)
    """
    def __init__(self, op, pool_size=1.4142, pseudo_random=False, overlapping=False,
                 data_format='channels_last', name=None, **kwargs):
        self.op, self.pseudo_random, self.overlapping = op, pseudo_random, overlapping
        self.pool_size = pool_size
        self.data_format = data_format
        self.name = name
        self.kwargs = kwargs

    def __call__(self, inputs):
        dim = inputs.shape.ndims - 2

        if self.op == 'max':
            op = tf.nn.fractional_max_pool
        elif self.op in ['mean', 'average', 'avg']:
            op = tf.nn.fractional_avg_pool

        _pooling_ratio = np.ones(inputs.shape.ndims)
        axis = 1 if self.data_format == 'channels_last' else 2
        _pooling_ratio[axis:axis+dim] = self.pool_size
        _pooling_ratio = list(_pooling_ratio)

        if dim == 1:
            with tf.variable_scope(self.kwargs.get('name') or 'fractional_pooling'):
                axis = 2 if self.data_format == 'channels_last' else -1
                x = tf.expand_dims(inputs, axis=axis)
                _pooling_ratio[axis] = 1
                x, _, _ = op(x, _pooling_ratio, self.pseudo_random, self.overlapping, **self.kwargs)
                x = tf.squeeze(x, [axis])
        elif dim in [2, 3]:
            x, _, _ = op(inputs, _pooling_ratio, self.pseudo_random, self.overlapping, **self.kwargs)
        else:
            raise ValueError("Number of dimensions in the inputs tensor should be 1, 2 or 3, but given %d" % dim)

        return x



@add_as_function
class Pooling(Layer):
    """ Multi-dimensional pooling layer
    Used for `p`, 'v' letters in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    op: str
        Pooling operation ('max', 'mean', 'average', 'avg').
    pool_size: int
        The size of the pooling window.
    strides: int
        The strides of the pooling operation.
    padding: str
        'same' or 'valid'.
    data_format: str
        'channels_last' or 'channels_first'.
    name: str
        Scope name.
    """
    LAYERS = {
        'max': MaxPooling,
        'mean': AveragePooling,
        'avg': AveragePooling,
        'average': AveragePooling,
        'frac-max': partial(FractionalPooling, 'max'),
        'fractional-max': partial(FractionalPooling, 'max'),
        'frac-avg': partial(FractionalPooling, 'avg'),
        'fractional-avg': partial(FractionalPooling, 'avg')
    }

    def __init__(self, op, pool_size, pool_strides, padding='same', data_format='channels_last', **kwargs):
        self.op = op
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

    def __call__(self, inputs):
        args = self.params_dict
        op = args.pop('op')
        layer_fn = self.LAYERS[op]
        return layer_fn(**args, **self.kwargs)(inputs)



@add_as_function
class GlobalPooling(Layer):
    """ Multi-dimensional global pooling layer.
    Used for `P`, `V` letters in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    op: str
        Pooling operation ('max', 'mean', 'average', 'avg').
    data_format: str
        'channels_last' or 'channels_first'.
    name: str
        Scope name.
    """
    def __init__(self, op, data_format='channels_last', keepdims=False, name=None):
        self.op, self.data_format, self.keepdims = op, data_format, keepdims
        self.name = name

    def __call__(self, inputs):
        dim = inputs.shape.ndims - 2
        axis = 1 if self.data_format == 'channels_last' else 2
        if dim == 1:
            pass
        elif dim == 2:
            axis = [axis, axis+1]
        elif dim == 3:
            axis = [axis, axis+1, axis+2]
        else:
            raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

        if self.op == 'max':
            x = tf.reduce_max(inputs, axis=axis, keepdims=self.keepdims, name=self.name)
        elif self.op in ['mean', 'average', 'avg']:
            x = tf.reduce_mean(inputs, axis=axis, keepdims=self.keepdims, name=self.name)

        return x



@add_as_function
class GlobalMaxPooling:
    """ Multi-dimensional global average-pooling layer.

    Parameters
    ----------
    data_format: str
        'channels_last' or 'channels_first'.
    name: str
        Scope name.
    """
    def __init__(self, data_format='channels_last', name=None):
        self.data_format = data_format
        self.name = name

    def __call__(self, inputs):
        return GlobalPooling('max', data_format=self.data_format, name=self.name)(inputs)



@add_as_function
class GlobalAveragePooling:
    """ Multi-dimensional global average-pooling layer.

    Parameters
    ----------
    data_format: str
        'channels_last' or 'channels_first'.
    name: str
        Scope name.
    """
    def __init__(self, data_format='channels_last', name=None):
        self.data_format = data_format
        self.name = name

    def __call__(self, inputs):
        return GlobalPooling('avg', data_format=self.data_format, name=self.name)(inputs)
