""" Contains convolution layers """
import tensorflow as tf

from .conv1d_tr import conv1d_transpose
from .pooling import max_pooling


ND_LAYERS = {
    'activation': None,
    'conv': [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d],
    'batch_norm': tf.layers.batch_normalization,
    'transposed_conv': [conv1d_transpose, tf.layers.conv2d_transpose, tf.layers.conv3d_transpose],
    'max_pooling': max_pooling,
    'dropout': tf.layers.dropout
}

C_LAYERS = {
    'a': 'activation',
    'c': 'conv',
    'n': 'batch_norm',
    't': 'transposed_conv',
    'p': 'max_pooling',
    'd': 'dropout'
}

def _get_layer_fn(fn, dim):
    f = ND_LAYERS[fn]
    return f if callable(f) or f is None else f[dim-1]


def conv_block(dim, input_tensor, filters, kernel_size, layout='cnap', name=None,
               strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=tf.nn.relu,
               pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex multi-dimensional convolution layer with batch normalization, activation, pooling and dropout

    Parameters
    ----------
    d : int {1, 2, 3}
        number of dimensions
    input_tensor : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    layout : str
        a sequence of layers:

        - c - convolution
        - t - transposed convolution
        - n - batch normalization
        - a - activation
        - p - max pooling
        - d - dropout

        Default is 'cnap'.
    name : str
        name of the layer that will be used as a scope.
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    dropout_rate : float
        Default is 0.
    is_training : bool or tf.Tensor
        Default is True.

    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict
        parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict
        parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict
        parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor : tf.Tensor
    """

    if not isinstance(dim, int) or dim < 1 or dim > 3:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    tensor = input_tensor
    for layer in layout:
        layer_name = C_LAYERS[layer]
        layer_fn = _get_layer_fn(layer_name, dim)

        if layer == 'a':
            tensor = activation(tensor)
        else:
            if layer == 'c':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format, dilation_rate=dilation_rate)
            elif layer == 't':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format)
            elif layer == 'n':
                args = dict(fused=True, axis=-1, training=is_training)
            elif layer == 'p':
                args = dict(dim=dim, pool_size=pool_size, strides=pool_strides, padding=padding,
                            data_format=data_format)
            elif layer == 'd' and (not isinstance(dropout_rate, float) or dropout_rate > 0):
                args = dict(rate=dropout_rate, training=is_training)

            args = {**args, **kwargs.get(layer_name, {})}
            tensor = layer_fn(inputs=tensor, **args)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor


def conv1d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 1d convolution with batch normalization, activation, pooling and dropout layers

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    layout : str
        a sequence of layers:

        - c - convolution
        - t - transposed convolution
        - n - batch normalization
        - a - activation
        - p - max pooling
        - d - dropout

        Default is 'cnap'.
    name : str
        name of the layer that will be used as a scope.
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    dropout_rate : float
        Default is 0.
    is_training : bool or tf.Tensor
        Default is True.

    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict
        parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict
        parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict
        parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(1, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv2d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation, pooling and dropout layers

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    layout : str
        a sequence of layers:

        - c - convolution
        - t - transposed convolution
        - n - batch normalization
        - a - activation
        - p - max pooling
        - d - dropout

        Default is 'cnap'.
    name : str
        name of the layer that will be used as a scope.
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    dropout_rate : float
        Default is 0.
    is_training : bool or tf.Tensor
        Default is True.

    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict
        parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict
        parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict
        parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(2, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv3d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation, pooling and dropout layers

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    layout : str
        a sequence of layers:

        - c - convolution
        - t - transposed convolution
        - n - batch normalization
        - a - activation
        - p - max pooling
        - d - dropout

        Default is 'cnap'.
    name : str
        name of the layer that will be used as a scope.
    strides : int
        Default is 1.
    padding : str
        padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    dropout_rate : float
        Default is 0.
    is_training : bool or tf.Tensor
        Default is True.

    conv : dict
        parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict
        parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict
        parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict
        parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict
        parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(3, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)
