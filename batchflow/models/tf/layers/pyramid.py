""" Contains pyramid layers """
import numpy as np
import tensorflow.compat.v1 as tf

from . import ConvBlock, Upsample


class PyramidPooling:
    """ Pyramid Pooling module.

    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    layout : str
        Layout for convolution layers.
    filters : int
        Number of filters in each pyramid branch.
    kernel_size : int
        Kernel size
    pool_op : str
        Pooling operation ('mean' or 'max').
    pyramid : tuple of int
        Number of feature regions in each dimension, default is (0, 1, 2, 3, 6).
        `0` is used to include `inputs` into the output tensor.
    flatten : bool
        If True, then the output is reshaped to a vector of constant size.
        If False, spatial shape of the inputs is preserved.
    name : str
        Layer name that will be used as a scope.
    """
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        return pyramid_pooling(inputs, *self.args, **self.kwargs)


def pyramid_pooling(inputs, layout='cna', filters=None, kernel_size=1, pool_op='mean', pyramid=(0, 1, 2, 3, 6),
                    flatten=False, name='psp', **kwargs):
    """ Pyramid Pooling module. """
    shape = inputs.get_shape().as_list()
    data_format = kwargs.get('data_format', 'channels_last')

    static_shape = np.array(shape[1: -1] if data_format == 'channels_last' else shape[2:])
    dynamic_shape = tf.shape(inputs)[1: -1] if data_format == 'channels_last' else tf.shape(inputs)[2:]
    axis = -1 if data_format == 'channels_last' else 1
    num_channels = shape[axis]
    if filters is None:
        filters = num_channels // len(pyramid)

    with tf.variable_scope(name):
        layers = []
        for level in pyramid:
            if level == 0:
                x = inputs
            else:
                # Pooling
                if None not in static_shape:
                    x = _static_pyramid_pooling(inputs, static_shape, level, pool_op, name='pool-%d' % level)
                    upsample_shape = static_shape
                else:
                    x = _dynamic_pyramid_pooling(inputs, level, pool_op, num_channels, data_format)
                    upsample_shape = dynamic_shape

                # Conv block to set number of feature maps
                x = ConvBlock(layout=layout, filters=filters, kernel_size=kernel_size,
                              name='conv-%d' % level, **kwargs)(x)

                # Output either vector with fixed size or tensor with fixed spatial dimensions
                if flatten:
                    x = tf.reshape(x, shape=(-1, level*level*filters),
                                   name='reshape-%d' % level)
                    concat_axis = -1
                else:
                    x = Upsample(layout='b', shape=upsample_shape, name='upsample-%d' % level, **kwargs)(x)
                    concat_axis = axis

            layers.append(x)
        x = tf.concat(layers, axis=concat_axis, name='concat')
    return x

def _static_pyramid_pooling(inputs, spatial_shape, level, pool_op, **kwargs):
    pool_size = tuple(np.ceil(spatial_shape / level).astype(np.int32).tolist())
    pool_strides = tuple(np.floor((spatial_shape - 1) / level + 1).astype(np.int32).tolist())

    output = ConvBlock(layout='p', pool_op=pool_op, pool_size=pool_size, pool_strides=pool_strides, **kwargs)(inputs)
    return output

def _dynamic_pyramid_pooling(inputs, level, pool_op, num_channels, data_format):
    if data_format == 'channels_last':
        h_axis, w_axis = 1, 2
    else:
        h_axis, w_axis = -2, -1

    inputs_shape = tf.shape(inputs)
    h_float = tf.cast(tf.gather(inputs_shape, h_axis), tf.float32)
    w_float = tf.cast(tf.gather(inputs_shape, w_axis), tf.float32)

    if pool_op == 'mean':
        pooling_op = tf.reduce_mean
    elif pool_op == 'max':
        pooling_op = tf.reduce_max
    else:
        raise ValueError('Wrong mode')

    def calc_pos(idx, level, size):
        """ Compute floor(idx*size // level) and cast it to tf.int. """
        return tf.cast(tf.floor(tf.multiply(tf.divide(idx, level), size)), tf.int32)

    result = []
    for row in range(level):
        for col in range(level):
            start_h = calc_pos(row, level, h_float)
            end_h = calc_pos(row+1, level, h_float)
            start_w = calc_pos(col, level, w_float)
            end_w = calc_pos(col+1, level, w_float)

            if data_format == 'channels_last':
                pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            else:
                pooling_region = inputs[..., start_h:end_h, start_w:end_w]

            pool_result = pooling_op(pooling_region, axis=(h_axis, w_axis))
            result.append(pool_result)

    output = tf.reshape(tf.stack(result, axis=1), shape=(-1, level, level, num_channels))
    return output



class ASPP:
    """ Atrous Spatial Pyramid Pooling module.

    Chen L. et al. "`Rethinking Atrous Convolution for Semantic Image Segmentation
    <https://arxiv.org/abs/1706.05587>`_"

    Parameters
    ----------
    layout : str
        Layout for convolution layers.
    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size for dilated branches (default=3).
    rates : tuple of int
        Dilation rates for branches, default=(6, 12, 18).
    image_level_features : int or tuple of int
        Number of image level features in each dimension.

        Default is 2, i.e. 2x2=4 pooling features will be calculated for 2d images,
        and 2x2x2=8 features per 3d item.

        Tuple allows to define several image level features, e.g (2, 3, 4).
    name : str
        Layer name that will be used as a scope.

    See also
    --------
    PyramidPooling
    """
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        return aspp(inputs, *self.args, **self.kwargs)


def aspp(inputs, layout='cna', filters=None, kernel_size=3, rates=(6, 12, 18), image_level_features=2,
         name='aspp', **kwargs):
    """ Atrous Spatial Pyramid Pooling module. """
    data_format = kwargs.get('data_format', 'channels_last')
    axis = -1 if data_format == 'channels_last' else 1
    if filters is None:
        filters = inputs.get_shape().as_list()[axis]
    if isinstance(image_level_features, int):
        image_level_features = (image_level_features,)

    with tf.variable_scope(name):
        x = ConvBlock(layout=layout, filters=filters, kernel_size=1, name='conv-1x1', **kwargs)(inputs)
        layers = [x]

        for level in rates:
            x = ConvBlock(layout=layout, filters=filters, kernel_size=kernel_size, dilation_rate=level,
                          name='conv-%d' % level, **kwargs)(inputs)
            layers.append(x)

        x = pyramid_pooling(inputs, filters=filters, pyramid=image_level_features,
                            name='image_level_features', **kwargs)
        layers.append(x)

        x = tf.concat(layers, axis=axis, name='concat')
        x = ConvBlock(layout=layout, filters=filters, kernel_size=1, name='last_conv', **kwargs)(x)
    return x
