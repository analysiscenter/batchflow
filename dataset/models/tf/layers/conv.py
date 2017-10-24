""" Contains convolution layers """
import tensorflow as tf

from .conv1d_tr import conv1d_transpose


ND_LAYERS = {
    'conv': [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d],
    'batch-norm': tf.layers.batch_normalization,
    'transposed-conv': [conv1d_transpose, tf.layers.conv2d_transpose, tf.layers.conv3d_transpose],
    'max-pooling': [tf.layers.max_pooling1d, tf.layers.max_pooling2d, tf.layers.max_pooling3d],
    'dropout': tf.layers.dropout
}

def _get_layer_fn(fn, dim):
    f = ND_LAYERS[fn]
    return f if callable(f) else f[dim-1]


def conv_block(dim, input_tensor, filters, kernel_size, layout='cnap', name=None,
               strides=1, padding='same', activation=tf.nn.relu,
               pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex multi-dimensional convolution layer with batch normalization, activation, pooling and dropout

    Parameters
    ----------
    d : int {1, 2, 3} - number of dimensions
    input_tensor : tf.Tensor
    filters : int - number of filters in the ouput tensor
    kernel_size  int - kernel size
    layout : str - a sequence of layers:
        c - convolution
        n - batch normalization
        a - activation
        p - max pooling
        Default is 'cnap'.
    name : str -  name of the layer that will be used as a scope
    strides : int. Default is 1.
    padding : str - padding mode, can be 'same' or 'valid'. Default - 'same'
    activation : callable. Default is `tf.nn.relu`.
    pool_size : int. Default is 2.
    pool_strides : int. Default is 2.
    dropout_rate : float. Default is 0.
    is_training : bool or tf.Tensor. Default is True.

    conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict - parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict - parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict - parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict - parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """

    if not isinstance(dim, int) or dim < 1 or dim > 3:
        raise ValueError("Number of dimensions could be 1, 2 or 3, but given %d" % dim)

    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    conv_layer, transposed_conv_layer = _get_layer_fn('conv', dim), _get_layer_fn('transposed-conv', dim)
    batch_norm, max_pooling = _get_layer_fn('batch-norm', dim), _get_layer_fn('max-pooling', dim)
    dropout = _get_layer_fn('dropout', dim)

    tensor = input_tensor
    for layer in layout:
        if layer == 'c':
            args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
            args = {**args, **kwargs.get('conv', {})}
            tensor = conv_layer(tensor, **args)
        elif layer == 't':
            args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
            args = {**args, **kwargs.get('transposed_conv', {})}
            tensor = transposed_conv_layer(tensor, **args)
        elif layer == 'a':
            tensor = activation(tensor)
        elif layer == 'n':
            args = kwargs.get('batch_norm', {})
            tensor = batch_norm(tensor, axis=-1, training=is_training, **args)
        elif layer == 'p':
            args = dict(pool_size=pool_size, strides=pool_strides, padding=padding)
            args = {**args, **kwargs.get('max_pooling', {})}
            tensor = max_pooling(tensor, **args)
        elif layer == 'd' and (not isinstance(dropout_rate, float) or dropout_rate > 0):
            args = dict(dropout_rate=dropout_rate)
            args = {**args, **kwargs.get('dropout', {})}
            tensor = dropout(tensor, **args, training=is_training)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor


def conv1d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor : tf.Tensor
    filter s: int - number of filters in the ouput tensor
    kernel_size:  int or tuple(int, int) - kernel size
    layout : str - a sequence of layers:
        c - convolution
        t - transposed convolution
        n - batch normalization
        a - activation
        p - max pooling
        Default is 'cnap'.
    name : str -  name of the layer that will be used as a scope
    strides : int. Default is 1.
    padding : str - padding mode, can be 'same' or 'valid'. Default - 'same'
    activation : callable. Default is `tf.nn.relu`.
    pool_size : int. Default is 2.
    pool_strides : int. Default is 2.
    dropout_rate : float. Default is 0.
    is_training : bool or tf.Tensor. Default is True.

    conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict - parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict - parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict - parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict - parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(1, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv2d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor : tf.Tensor
    filter s: int - number of filters in the ouput tensor
    kernel_size:  int or tuple(int, int) - kernel size
    layout : str - a sequence of layers:
        c - convolution
        t - transposed convolution
        n - batch normalization
        a - activation
        p - max pooling
        Default is 'cnap'.
    name : str -  name of the layer that will be used as a scope
    strides : int. Default is 1.
    padding : str - padding mode, can be 'same' or 'valid'. Default - 'same'
    activation : callable. Default is `tf.nn.relu`.
    pool_size : int. Default is 2.
    pool_strides : int. Default is 2.
    dropout_rate : float. Default is 0.
    is_training : bool or tf.Tensor. Default is True.

    conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict - parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict - parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict - parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict - parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(2, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv3d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor : tf.Tensor
    filter s: int - number of filters in the ouput tensor
    kernel_size:  int or tuple(int, int) - kernel size
    layout : str - a sequence of layers:
        c - convolution
        t - transposed convolution
        n - batch normalization
        a - activation
        p - max pooling
        Default is 'cnap'.
    name : str -  name of the layer that will be used as a scope
    strides : int. Default is 1.
    padding : str - padding mode, can be 'same' or 'valid'. Default - 'same'
    activation : callable. Default is `tf.nn.relu`.
    pool_size : int. Default is 2.
    pool_strides : int. Default is 2.
    dropout_rate : float. Default is 0.
    is_training : bool or tf.Tensor. Default is True.

    conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
    transposed_conv : dict - parameters for transposed conv layers, like initializers, regularalizers, etc
    batch_norm : dict - parameters for batch normalization layers, like momentum, intiializers, etc
    max_pooling : dict - parameters for max_pooling layers, like initializers, regularalizers, etc
    dropout : dict - parameters for dropout layers, like noise_shape, etc

    Returns
    -------
    output tensor: tf.Tensor
    """
    return conv_block(3, input_tensor, filters, kernel_size, layout, name,
                      strides, padding, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)
