""" Contains 2d conv layers """
import tensorflow as tf


def conv1d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=1, padding='same', use_bias=True, activation=tf.nn.relu,
                 pool_size=2, pool_strides=2, is_training=True):
    """ Complex 1d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor: tf.Tensor
    filters: int - number of filters in the ouput tensor
    kernel_size: int - kernel size
    layout: str - a sequence of layers
        c - convolution
        n - batch normalization
        a - activation
        p - max pooling
    name: str -  name of the layer that will be used as a scope
    strides: int
    padding: str - padding mode, can be 'same' or 'valid';
    use_bias: bool
    activation: callable
    pool_size: int
    pool_strides: int
    is_training: bool or tf.Tensor

    Returns
    -------
    output tensor: tf.Tensor
    """
    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    tensor = input_tensor
    for layer in layout:
        if layer == 'c':
            tensor = tf.layers.conv1d(tensor, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=padding, use_bias=use_bias)
        elif layer == 'a':
            tensor = activation(tensor)
        elif layer == 'n':
            tensor = tf.layers.batch_normalization(tensor, axis=-1, training=is_training)
        elif layer == 'p':
            tensor = tf.layers.max_pooling1d(tensor, pool_size, pool_strides, padding)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor


def conv2d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=(1, 1), padding='same', use_bias=True, activation=tf.nn.relu,
                 pool_size=(2, 2), pool_strides=(2, 2), is_training=True):
    """ Complex 2d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor: tf.Tensor
    filters: int - number of filters in the ouput tensor
    kernel_size: int or tuple(int, int) - kernel size
    layout: str - a sequence of layers
        c - convolution
        n - batch normalization
        a - activation
        p - max pooling
    name: str -  name of the layer that will be used as a scope
    strides: tuple(int, int)
    padding: str - padding mode, can be 'same' or 'valid';
    use_bias: bool
    activation: callable
    pool_size: tuple(int, int)
    pool_strides: tuple(int, int)
    is_training: bool or tf.Tensor

    Returns
    -------
    output tensor: tf.Tensor
    """
    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    tensor = input_tensor
    for layer in layout:
        if layer == 'c':
            tensor = tf.layers.conv2d(tensor, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=padding, use_bias=use_bias)
        elif layer == 'a':
            tensor = activation(tensor)
        elif layer == 'n':
            tensor = tf.layers.batch_normalization(tensor, axis=-1, training=is_training)
        elif layer == 'p':
            tensor = tf.layers.max_pooling2d(tensor, pool_size, pool_strides, padding)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor

def conv3d_block(input_tensor, filters, kernel_size, layout='cnap', name=None,
                 strides=(1, 1, 1), padding='same', use_bias=True, activation=tf.nn.relu,
                 pool_size=(2, 2, 2), pool_strides=(2, 2, 2), is_training=True):
    """ Complex 3d convolution with batch normalization, activation and pooling layers

    Parameters
    ----------
    input_tensor: tf.Tensor
    filters: int - number of filters in the ouput tensor
    kernel_size: int or tuple(int, int, int) - kernel size
    layout: str - a sequence of layers
        c - convolution
        n - batch normalization
        a - activation
        p - max pooling
    name: str -  name of the layer that will be used as a scope
    strides: tuple(int, int, int)
    padding: str - padding mode, can be 'same' or 'valid';
    use_bias: bool
    activation: callable
    pool_size: tuple(int, int, int)
    pool_strides: tuple(int, int, int)
    is_training: bool or tf.Tensor

    Returns
    -------
    output tensor: tf.Tensor
    """
    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    tensor = input_tensor
    for layer in layout:
        if layer == 'c':
            tensor = tf.layers.conv3d(tensor, filters=filters, kernel_size=kernel_size, strides=strides,
                                      padding=padding, use_bias=use_bias)
        elif layer == 'a':
            tensor = activation(tensor)
        elif layer == 'n':
            tensor = tf.layers.batch_normalization(tensor, axis=-1, training=is_training)
        elif layer == 'p':
            tensor = tf.layers.max_pooling3d(tensor, pool_size, pool_strides, padding)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor
