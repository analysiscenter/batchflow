""" Contains convolution layers """
import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as K #pylint: disable=import-error

from .core import Mip, Flatten
from .conv import Conv, ConvTranspose, SeparableConv, SeparableConvTranspose
from .pooling import Pooling, GlobalPooling
from .drop_block import Dropblock
from .resize import ResizeBilinearAdditive, ResizeBilinear, ResizeNn, SubpixelConv
from .utils import add_as_function
from ...utils import unpack_args



logger = logging.getLogger(__name__)



@add_as_function
class ConvBlock:
    """ Complex multi-dimensional block with a sequence of convolutions, batch normalization, activation, pooling,
    dropout and even dense layers.

    Parameters
    ----------
    layout : str
        A sequence of operations:

        - c - convolution
        - t - transposed convolution
        - C - separable convolution
        - T - separable transposed convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - pooling (default is max-pooling)
        - v - average pooling
        - P - global pooling (default is max-pooling)
        - V - global average pooling
        - d - dropout
        - D - dropblock
        - m - maximum intensity projection (:func:`~.layers.mip`)
        - b - upsample with bilinear resize
        - B - upsample with bilinear additive resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:func:`~.layers.subpixel_conv`)
        - R - start residual connection
        - A - start residual connection with bilinear additive upsampling
        - `+` - end residual connection with summation
        - `.` - end residual connection with concatenation

        Default is ''.

    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size.
    name : str
        Name of the layer that will be used as a scope.
    units : int
        Number of units in the dense layer.
    strides : int
        Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
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
    pool_op : str
        Pooling operation ('max', 'mean', 'frac')
    dropout_rate : float
        Default is 0.
    factor : int or tuple of int
        Upsampling factor
    upsampling_layout : str
        Layout for upsampling layers
    is_training : bool or tf.Tensor
        Default is True.
    reuse : bool
        Whether to user layer variables if exist

    dense : dict
        Parameters for dense layers, like initializers, regularalizers, etc.
    conv : dict
        Parameters for convolution layers, like initializers, regularalizers, etc.
    transposed_conv : dict
        Parameters for transposed conv layers, like initializers, regularalizers, etc.
    batch_norm : dict or None
        Parameters for batch normalization layers, like momentum, intiializers, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    pooling : dict
        Parameters for pooling layers, like initializers, regularalizers, etc
    dropout : dict or None
        Parameters for dropout layers, like noise_shape, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    dropblock : dict or None
        Parameters for dropblock layers, like dropout_rate, block_size, etc
    subpixel_conv : dict or None
        Parameters for subpixel convolution like layout, activation, etc.
    resize_bilinear : dict or None
        Parameters for bilinear resize
    resize_bilinear_additive : dict or None
        Parameters for bilinear additive resize like layout, activation, etc

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.


    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = ConvBlock('cnap', filters=32, kernel_size=3)(x)

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = ConvBlock('nac nac nac', [64, 64, 256], [1, 3, 1])(x)

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

        x = ConvBlock('ca ca ca nd', [32, 32, 64], [5, 3, 3], strides=[1, 1, 2], dropout_rate=.15)(x)

    A residual block::

        x = ConvBlock('R nac nac +', [16, 16, 64], [1, 3, 1])(x)

    """
    C_LAYERS = {
        'a': 'activation',
        'R': 'residual_start',
        '+': 'residual_end',
        '.': 'residual_end',
        'f': 'dense',
        'c': 'conv',
        't': 'transposed_conv',
        'C': 'separable_conv',
        'T': 'separable_conv_transpose',
        'p': 'pooling',
        'v': 'pooling',
        'P': 'global_pooling',
        'V': 'global_pooling',
        'n': 'batch_norm',
        'd': 'dropout',
        'D': 'dropblock',
        'm': 'mip',
        'A': 'residual_bilinear_additive',
        'b': 'resize_bilinear',
        'B': 'resize_bilinear_additive',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    FUNC_LAYERS = {
        'activation': None,
        'residual_start': None,
        'residual_end': None,
        'dense': K.Dense,
        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'pooling': Pooling,
        'global_pooling': GlobalPooling,
        'batch_norm': K.BatchNormalization,
        'dropout': K.Dropout,
        'mip': Mip,
        'dropblock': Dropblock,
        'residual_bilinear_additive': None,
        'resize_bilinear': ResizeBilinear,
        'resize_bilinear_additive': ResizeBilinearAdditive,
        'resize_nn': ResizeNn,
        'subpixel_conv': SubpixelConv
    }

    LAYER_KEYS = ''.join(list(C_LAYERS.keys()))
    GROUP_KEYS = (
        LAYER_KEYS
        .replace('t', 'c')
        .replace('C', 'c')
        .replace('T', 'c')
        .replace('v', 'p')
        .replace('V', 'P')
        .replace('D', 'd')
        .replace('A', 'b')
        .replace('B', 'b')
        .replace('N', 'b')
        .replace('X', 'b')
    )

    C_GROUPS = dict(zip(LAYER_KEYS, GROUP_KEYS))

    def __init__(self, layout='', filters=0, kernel_size=3, name=None,
                 strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
                 activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., **kwargs):
        self.layout = layout
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format, self.name = padding, data_format, name
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kwargs = kwargs


    def add_letter(self, letter, func, name=None):
        name = name or letter
        self.C_LAYERS.update({letter: name})
        self.FUNC_LAYERS.update({name: func})
        self.C_GROUPS.update({letter: letter})


    def __call__(self, inputs, training=None):
        if training is None:
            training = self.kwargs.get('is_training') or self.kwargs.get('training')

        layout = self.layout or ''
        layout = layout.replace(' ', '')

        if len(layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do, just returning inputs.')
            return inputs

        context = None
        if self.name is not None:
            context = tf.variable_scope(self.name, reuse=self.kwargs.get('reuse'))
            context.__enter__()

        layout_dict = {}
        for layer in layout:
            if self.C_GROUPS[layer] not in layout_dict:
                layout_dict[self.C_GROUPS[layer]] = [-1, 0]
            layout_dict[self.C_GROUPS[layer]][1] += 1


        residuals = []
        tensor = inputs
        for i, layer in enumerate(layout):

            layout_dict[self.C_GROUPS[layer]][0] += 1
            layer_name = self.C_LAYERS[layer]
            layer_fn = self.FUNC_LAYERS[layer_name]

            args = {}
            call_args = {}

            if layer == 'a':
                args = dict(activation=self.activation)
                layer_fn = unpack_args(args, *layout_dict[self.C_GROUPS[layer]])['activation']
                if layer_fn is not None:
                    tensor = layer_fn(tensor)
            elif layer == 'R':
                residuals += [tensor]
            elif layer == 'A':
                args = dict(factor=self.kwargs.get('factor'), data_format=self.data_format)
                args = unpack_args(args, *layout_dict[self.C_GROUPS[layer]])
                t = self.FUNC_LAYERS['resize_bilinear_additive'](tensor, **args, name='rba-%d' % i)
                residuals += [t]
            elif layer == '+':
                tensor = tensor + residuals[-1]
                residuals = residuals[:-1]
            elif layer == '.':
                axis = -1 if self.data_format == 'channels_last' else 1
                tensor = tf.concat([tensor, residuals[-1]], axis=axis, name='concat-%d' % i)
                residuals = residuals[:-1]
            else:
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False or isinstance(layer_args, dict) and layer_args.get('disable', False)

                if skip_layer:
                    pass
                elif layer == 'f':
                    if tensor.shape.ndims > 2:
                        tensor = Flatten()(tensor)
                    units = self.kwargs.get('units')
                    if units is None:
                        raise ValueError('units cannot be None if layout includes dense layers')
                    args = dict(units=units)

                elif layer == 'c':
                    args = dict(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)
                    if self.filters is None or self.filters == 0:
                        raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

                elif layer == 'C':
                    args = dict(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate, depth_multiplier=self.depth_multiplier)
                    if self.filters is None or self.filters == 0:
                        raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

                elif layer == 't':
                    args = dict(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                data_format=self.data_format)
                    if self.filters is None or self.filters == 0:
                        raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

                elif layer == 'T':
                    args = dict(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, depth_multiplier=self.depth_multiplier)
                    if self.filters is None or self.filters == 0:
                        raise ValueError('filters cannot be None or 0 if layout includes convolutional layers')

                elif layer == 'n':
                    axis = -1 if self.data_format == 'channels_last' else 1
                    args = dict(fused=True, axis=axis)
                    call_args.update({'training': training})

                elif self.C_GROUPS[layer] == 'p':
                    pool_op = 'mean' if layer == 'v' else self.kwargs.pop('pool_op', 'max')
                    args = dict(op=pool_op, pool_size=self.pool_size, strides=self.pool_strides, padding=self.padding,
                                data_format=self.data_format)

                elif self.C_GROUPS[layer] == 'P':
                    pool_op = 'mean' if layer == 'V' else self.kwargs.pop('pool_op', 'max')
                    args = dict(op=pool_op, data_format=self.data_format, keepdims=self.kwargs.get('keep_dims', False))

                elif layer == 'd':
                    if self.dropout_rate:
                        args = dict(rate=self.dropout_rate)
                        call_args.update({'training': training})
                    else:
                        logger.warning('conv_block: dropout_rate is zero or undefined, so dropout layer is skipped')
                        skip_layer = True

                elif layer == 'D':
                    if not self.dropout_rate:
                        dropout_rate = layer_args.get('dropout_rate')
                    if not self.kwargs.get('block_size'):
                        block_size = layer_args.get('block_size')
                    if dropout_rate and block_size:
                        args = dict(dropout_rate=dropout_rate, block_size=block_size,
                                    seed=self.kwargs.get('seed'), data_format=self.data_format, global_step=self.kwargs.get('global_step'))
                        call_args.update({'training': training})
                    else:
                        logger.warning(('conv_block/dropblock: dropout_rate or block_size is'
                                        ' zero or undefined, so dropblock layer is skipped'))
                        skip_layer = True

                elif layer == 'm':
                    args = dict(depth=self.kwargs.get('depth'), data_format=self.data_format)

                elif layer in ['b', 'B', 'N', 'X']:
                    args = dict(factor=self.kwargs.get('factor'), shape=self.kwargs.get('shape'), data_format=self.data_format)
                    if self.kwargs.get('upsampling_layout'):
                        args['layout'] = self.kwargs.get('upsampling_layout')

                else:
                    if layer in self.C_LAYERS.keys():
                        pass
                    else:
                        raise ValueError('Unknown layer symbol - %s' % layer)

                if not skip_layer:
                    args = {**args, **layer_args}
                    args = unpack_args(args, *layout_dict[self.C_GROUPS[layer]])

                    with tf.variable_scope('layer-%d' % i):
                        tensor = layer_fn(**args)(tensor, **call_args)
        tensor = tf.identity(tensor, name='_output')

        if context is not None:
            context.__exit__(None, None, None)

        return tensor



@add_as_function
class Upsample:
    """ Upsample inputs with a given factor.

    Parameters
    ----------
    factor : int
        An upsamping scale
    shape : tuple of int
        Shape to upsample to (used by bilinear and NN resize)
    layout : str
        Resizing technique, a sequence of:

        - A - use residual connection with bilinear additive upsampling
        - b - bilinear resize
        - B - bilinear additive upsampling
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :class:`.ConvBlock` layers are also allowed.

    Examples
    --------
    A simple bilinear upsampling::

        x = upsample(shape=(256, 256), layout='b')(x)

    Upsampling with non-linear normalized transposed convolution::

        x = Upsample(factor=2, layout='nat', kernel_size=3)(x)

    Subpixel convolution with a residual bilinear additive connection::

        x = Upsample(factor=2, layout='AX+')(x)
    """
    def __init__(self, factor=None, shape=None, layout='b', name='upsample', **kwargs):
        self.factor, self.shape, self.layout = factor, shape, layout
        self.name, self.kwargs = name, kwargs

    def __call__(self, *args, **kwargs):
        if np.all(self.factor == 1):
            return inputs

        if 't' in self.layout or 'T' in self.layout:
            if 'kernel_size' not in self.kwargs:
                self.kwargs['kernel_size'] = self.factor
            if 'strides' not in kwargs:
                self.kwargs['strides'] = self.factor

        return ConvBlock(self.layout, name=self.name, factor=self.factor, shape=self.shape, **self.kwargs)(*args, **kwargs)
