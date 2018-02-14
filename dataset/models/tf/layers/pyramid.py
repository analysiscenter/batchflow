""" Contains pyramid layers """
import tensorflow as tf

from .conv_block import conv_block


def pyramid_pooling(inputs, layout='cna', filters=None, kernel_size=1, pool_op='mean', pool_size=(1, 2, 3, 6),
                    upsample=None, name='psp', **kwargs):
    """ Pyramid Pooling module

    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        layout for convolution layers
    filters : int
        the number of filters in the output tensor
    kernel_size : int
        kernel size
    pool_size : tuple of int
        feature region sizes - pooling kernel sizes (e.g. [1, 2, 3, 6])
    upsample : dict
        upsample parameters
    name : str
        name of the layer that will be used as a scope.

    Returns
    -------
    tf.Tensor
    """
    data_format = kwargs.get('data_format', 'channels_last')
    axis = -1 if data_format == 'channels_axis' else 1
    if filters is None:
        filters = inputs.get_shape().as_list()[axis]
    upsample = upsample if upsample is not None else {}
    upsample_args = {**kwargs, **upsample}

    with tf.variable_scope(name):
        x = inputs
        layers = []
        for level in pool_size:
            if level == 1:
                pass
            else:
                x = conv_block(x, 'p', pool_op=pool_op, pool_size=level, pool_strides=level,
                               name='pool-%d' % level, **kwargs)
            x = conv_block(x, layout, filters=filters, kernel_size=kernel_size, name='conv-%d' % level, **kwargs)
            x = upsample(x, factor=level, name='upsample-%d' % level, **upsample_args)
            layers.append(x)
        x = tf.concat(layers, axis=axis)
    return x

def aspp(inputs, layout='cna', filters=None, kernel_size=3, rates=(6, 12, 18), name='aspp', **kwargs):
    """ Atrous Spatial Pyramid Pooling module

    Chen L. et al. "`Rethinking Atrous Convolution for Semantic Image Segmentation
    <https://arxiv.org/abs/1706.05587>`_"

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor
    layout : str
        layout for convolution layers
    filters : int
        the number of filters in the output tensor
    kernel_size : int
        kernel size (default=3)
    rates : tuple of int
        dilation rates for branches (default=(6, 12, 18))
    name : str
        name of the layer that will be used as a scope.

    Returns
    -------
    tf.Tensor
    """
    data_format = kwargs.get('data_format', 'channels_last')
    axis = -1 if data_format == 'channels_axis' else 1
    if filters is None:
        filters = inputs.get_shape().as_list()[axis]

    with tf.variable_scope(name):
        layers = []
        layers.append(conv_block(inputs, layout, filters=filters, kernel_size=1, name='conv-1x1', **kwargs))

        for level in rates:
            x = conv_block(inputs, layout, filters=filters, kernel_size=kernel_size, dilation_rate=level,
                           name='conv-%d' % level, **kwargs)
            layers.append(x)

        x = tf.concat(layers, axis=axis, name='concat')
        x = conv_block(x, layout, filters=filters, kernel_size=1, name='last_conv', **kwargs)
    return x
