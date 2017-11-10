""" Contains convolution layers """
import tensorflow as tf

from .core import mip, flatten
from .conv import conv1d_transpose, separable_conv
from .pooling import max_pooling, average_pooling, global_max_pooling, global_average_pooling


ND_LAYERS = {
    'activation': None,
    'dense': tf.layers.dense,
    'conv': [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d],
    'transposed_conv': [conv1d_transpose, tf.layers.conv2d_transpose, tf.layers.conv3d_transpose],
    'separable_conv':separable_conv,
    'max_pooling': max_pooling,
    'average_pooling': average_pooling,
    'global_max_pooling': global_max_pooling,
    'global_average_pooling': global_average_pooling,
    'batch_norm': tf.layers.batch_normalization,
    'dropout': tf.layers.dropout,
    'mip': mip
}

C_LAYERS = {
    'a': 'activation',
    'f': 'dense',
    'c': 'conv',
    't': 'transposed_conv',
    'p': 'max_pooling',
    'v': 'average_pooling',
    'P': 'global_max_pooling',
    'V': 'global_average_pooling',
    'n': 'batch_norm',
    'd': 'dropout',
    'm': 'mip'
}

_LAYERS_KEYS = str(list(C_LAYERS.keys()))
_GROUP_KEYS = _LAYERS_KEYS.replace('t', 'c').replace('v', 'p')
C_GROUPS = dict(zip(_LAYERS_KEYS, _GROUP_KEYS))

def _get_layer_fn(fn, dim):
    f = ND_LAYERS[fn]
    return f if callable(f) or f is None else f[dim-1]

def _unpack_args(args, layer_no, layers_max):
    new_args = {}
    for arg in args:
        if isinstance(args[arg], (tuple, list)) and layers_max > 1:
            arg_value = args[arg][layer_no]
        else:
            arg_value = args[arg]
        new_args.update({arg: arg_value})

    return new_args

def conv_block(dim, inputs, filters=0, kernel_size=3, layout='cna', name=None,
               strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
               activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex multi-dimensional convolution layer with batch normalization, activation, pooling and dropout

    Parameters
    ----------
    d : int {1, 2, 3}
        number of dimensions
    inputs : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    layout : str
        a sequence of layers:

        - c - convolution (+ dilated, separable)
        - t - transposed convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - max pooling
        - v - average pooling
        - P - global max pooling
        - V - global average pooling
        - d - dropout
        - m - maximum intensity projection (:func:`.layers.mip`)

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

    dense : dict
        parameters for dense layers, like initializers, regularalizers, etc
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

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists/tuples.

    Examples
    --------
    A simple 2d block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = conv_block(2, x, 32, 3, layout='cnap')

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = conv_block(2, x, [64, 64, 256], [1, 3, 1], layout='cacac')

    A complex Nd block:

    - 5x5 conv with 32 filters
    - relu
    - 3x3 conv with 32 filters
    - relu
    - 3x3 conv with 64 filters and a spatial stride 2
    - relu
    - batch norm
    - dropout with rate 0.15

    ::

        x = conv_block(dim, x, [32, 32, 64], [5, 3, 3], layout='cacacand', strides=[1, 1, 2], dropout_rate=.15)
    """

    if not isinstance(dim, int) or dim < 1 or dim > 3:
        raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

    context = None
    if name is not None:
        context = tf.variable_scope(name)
        context.__enter__()

    layout_dict = {}
    for layer in layout:
        if C_GROUPS[layer] not in layout_dict:
            layout_dict[C_GROUPS[layer]] = [-1, 0]
        layout_dict[C_GROUPS[layer]][1] += 1

    layout = layout.replace(' ', '')
    tensor = inputs
    for i, layer in enumerate(layout):

        layout_dict[C_GROUPS[layer]][0] += 1
        layer_name = C_LAYERS[layer]
        layer_fn = _get_layer_fn(layer_name, dim)

        if layer == 'a':
            tensor = activation(tensor)
        else:
            if layer == 'f':
                if len(tensor.get_shape().as_list()) > 2:
                    tensor = flatten(tensor)
                units = kwargs.get('units')
                if units is None:
                    raise ValueError('units cannot be None if layout includes dense layers')
                args = dict(units=units)
            elif layer == 'c':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format, dilation_rate=dilation_rate, depth_multiplier=depth_multiplier)
            elif layer == 't':
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            data_format=data_format)
            elif layer == 'n':
                axis = -1 if data_format == 'channels_last' else 1
                args = dict(fused=True, axis=axis, training=is_training)
            elif C_GROUPS[layer] == 'p':
                args = dict(dim=dim, pool_size=pool_size, strides=pool_strides, padding=padding,
                            data_format=data_format)
            elif layer == 'd' and (not isinstance(dropout_rate, float) or dropout_rate > 0):
                args = dict(rate=dropout_rate, training=is_training)
            elif layer in ['P', 'V']:
                args = dict(dim=dim, data_format=data_format)
            elif layer == 'm':
                args = dict(data_format=data_format)

            args = {**args, **kwargs.get(layer_name, {})}
            args = _unpack_args(args, *layout_dict[C_GROUPS[layer]])

            if layer == 'c':
                if args['depth_multiplier'] > 1:
                    layer_fn = _get_layer_fn('separable_conv', dim)
                    args['dim'] = dim
                else:
                    args.pop('depth_multiplier')

            with tf.variable_scope('layer-%d' % i):
                tensor = layer_fn(inputs=tensor, **args)

    if context is not None:
        context.__exit__(None, None, None)

    return tensor


def conv1d_block(inputs, filters=0, kernel_size=3, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
                 activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 1d convolution with batch normalization, activation, pooling and dropout layers

    See also
    --------
    :func:`.conv_block`
    """
    return conv_block(1, inputs, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, depth_multiplier, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv2d_block(inputs, filters=0, kernel_size=3, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
                 activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 2d convolution with batch normalization, activation, pooling and dropout layers

    See also
    --------
    :func:`.conv_block`
    """
    return conv_block(2, inputs, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, depth_multiplier, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)


def conv3d_block(inputs, filters=0, kernel_size=3, layout='cnap', name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
                 activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True, **kwargs):
    """ Complex 3d convolution with batch normalization, activation, pooling and dropout layers

    See also
    --------
    :func:`.conv_block`
    """
    return conv_block(3, inputs, filters, kernel_size, layout, name,
                      strides, padding, data_format, dilation_rate, depth_multiplier, activation,
                      pool_size, pool_strides, dropout_rate, is_training, **kwargs)
