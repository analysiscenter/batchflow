""" Contains convolution layers """
import logging
import numpy as np
import tensorflow as tf

from .core import Dense, Dropout, AlphaDropout, BatchNormalization, Mip
from .conv import Conv, ConvTranspose, SeparableConv, SeparableConvTranspose, DepthwiseConv, DepthwiseConvTranspose
from .pooling import Pooling, GlobalPooling
from .drop_block import Dropblock
from .resize import ResizeBilinearAdditive, ResizeBilinear, ResizeNn, SubpixelConv
from .layer import add_as_function
from ...utils import unpack_args


logger = logging.getLogger(__name__)



@add_as_function
class ConvBlock:
    """ Complex multi-dimensional block to apply sequence of different operations.

    Parameters
    ----------
    layout : str
        A sequence of letters, each letter meaning individual operation:

        - c - convolution
        - t - transposed convolution
        - C - separable convolution
        - T - separable transposed convolution
        - w - depthwise convolution
        - W - depthwise transposed convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - pooling (default is max-pooling)
        - v - average pooling
        - P - global pooling (default is max-pooling)
        - V - global average pooling
        - d - dropout
        - D - alpha dropout
        - O - dropblock
        - m - maximum intensity projection (:class:`~.layers.Mip`)
        - b - upsample with bilinear resize
        - B - upsample with bilinear additive resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:class:`~.layers.SubpixelConv`)
        - R - start residual connection
        - A - start residual connection with bilinear additive upsampling
        - `+` - end residual connection with summation
        - `*` - end residual connection with multiplication
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
        Parameters for pooling layers, like initializers, regularalizers, etc.
    dropout : dict or None
        Parameters for dropout layers, like noise_shape, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    dropblock : dict or None
        Parameters for dropblock layers, like dropout_rate, block_size, etc.
    subpixel_conv : dict or None
        Parameters for subpixel convolution like layout, activation, etc.
    resize_bilinear : dict or None
        Parameters for bilinear resize.
    resize_bilinear_additive : dict or None
        Parameters for bilinear additive resize like layout, activation, etc.

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
    LETTERS_LAYERS = {
        'a': 'activation',
        'R': 'residual_start',
        '+': 'residual_end',
        '.': 'residual_end',
        '*': 'residual_end',
        'f': 'dense',
        'c': 'conv',
        't': 'transposed_conv',
        'C': 'separable_conv',
        'T': 'separable_conv_transpose',
        'w': 'depthwise_conv',
        'W': 'depthwise_conv_transpose',
        'p': 'pooling',
        'v': 'pooling',
        'P': 'global_pooling',
        'V': 'global_pooling',
        'n': 'batch_norm',
        'd': 'dropout',
        'D': 'alpha_dropout',
        'O': 'dropblock',
        'm': 'mip',
        'A': 'residual_bilinear_additive',
        'b': 'resize_bilinear',
        'B': 'resize_bilinear_additive',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    LAYERS_CLASSES = {
        'activation': None,
        'residual_start': None,
        'residual_end': None,
        'dense': Dense,
        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'depthwise_conv': DepthwiseConv,
        'depthwise_conv_transpose': DepthwiseConvTranspose,
        'pooling': Pooling,
        'global_pooling': GlobalPooling,
        'batch_norm': BatchNormalization,
        'dropout': Dropout,
        'alpha_dropout': AlphaDropout,
        'dropblock': Dropblock,
        'mip': Mip,
        'residual_bilinear_additive': None,
        'resize_bilinear': ResizeBilinear,
        'resize_bilinear_additive': ResizeBilinearAdditive,
        'resize_nn': ResizeNn,
        'subpixel_conv': SubpixelConv
    }

    DEFAULT_LETTERS = LETTERS_LAYERS.keys()
    LETTERS_GROUPS = dict(zip(DEFAULT_LETTERS, DEFAULT_LETTERS))
    LETTERS_GROUPS.update({
        'C': 'c',
        't': 'c',
        'T': 'c',
        'w': 'c',
        'W': 'c',
        'v': 'p',
        'V': 'P',
        'D': 'd',
        'O': 'd',
        'n': 'd',
        'A': 'b',
        'B': 'b',
        'N': 'b',
        'X': 'b',
        })

    def __init__(self, layout='',
                 filters=0, kernel_size=3, strides=1, dilation_rate=1, depth_multiplier=1,
                 activation=tf.nn.relu,
                 pool_size=2, pool_strides=2,
                 dropout_rate=0.,
                 padding='same', data_format='channels_last', name=None,
                 **kwargs):
        self.layout = layout
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.dropout_rate = dropout_rate
        self.padding, self.data_format = padding, data_format
        self.name = name
        self.kwargs = kwargs


    def add_letter(self, letter, cls, name=None):
        """ Add custom letter to layout parsing procedure.

        Parameters
        ----------
        letter : str
            Letter to add.
        cls : class
            Tensor-processing layer. Must have layer-like signature (both init and call overloaded).
        name : str
            Name of parameter dictionary. Defaults to `letter`.

        Examples
        --------
        Add custom `Q` letter::

            block = ConvBlock('cnap Q', filters=32, custom_params={'key': 'value'})
            block.add_letter('Q', my_layer_class, 'custom_params')
            x = block(x)
        """
        name = name or letter
        self.LETTERS_LAYERS.update({letter: name})
        self.LAYERS_CLASSES.update({name: cls})
        self.LETTERS_GROUPS.update({letter: letter})


    def __call__(self, inputs, training=None):
        layout = self.layout or ''
        layout = layout.replace(' ', '')
        if len(layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do, just returning inputs.')
            return inputs

        # Getting `training` indicator from kwargs by its aliases
        if training is None:
            training = self.kwargs.get('is_training')
        if training is None:
            training = self.kwargs.get('training')

        context = None
        if self.name is not None:
            context = tf.variable_scope(self.name, reuse=self.kwargs.get('reuse'))
            context.__enter__()

        layout_dict = {}
        for letter in layout:
            letter_group = self.LETTERS_GROUPS[letter]
            letter_counts = layout_dict.setdefault(letter_group, [-1, 0])
            letter_counts[1] += 1

        tensor = inputs
        residuals = []
        for i, letter in enumerate(layout):
            # Arguments for layer creating; arguments for layer call
            args, call_args = {}, {}

            letter_group = self.LETTERS_GROUPS[letter]
            layer_name = self.LETTERS_LAYERS[letter]
            layer_class = self.LAYERS_CLASSES[layer_name]
            layout_dict[letter_group][0] += 1

            if letter == 'a':
                args = dict(activation=self.activation)
                activation_fn = unpack_args(args, *layout_dict[letter_group])['activation']
                if activation_fn is not None:
                    tensor = activation_fn(tensor)
            elif letter == 'R':
                residuals += [tensor]
            elif letter == 'A':
                args = dict(factor=self.kwargs.get('factor'), data_format=self.data_format)
                args = unpack_args(args, *layout_dict[letter_group])
                t = self.LAYERS_CLASSES['resize_bilinear_additive'](**args, name='rba-%d' % i)(tensor)
                residuals += [t]
            elif letter == '+':
                tensor = tensor + residuals[-1]
                residuals = residuals[:-1]
            elif letter == '*':
                tensor = tensor * residuals[-1]
                residuals = residuals[:-1]
            elif letter == '.':
                axis = -1 if self.data_format == 'channels_last' else 1
                tensor = tf.concat([tensor, residuals[-1]], axis=axis, name='concat-%d' % i)
                residuals = residuals[:-1]
            else:
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False or \
                             isinstance(layer_args, dict) and layer_args.get('disable', False)

                # Create params for the layer call
                if skip_layer:
                    pass
                elif letter in self.DEFAULT_LETTERS:
                    args = {param: getattr(self, param, self.kwargs.get(param, None))
                            for param in layer_class.params
                            if (hasattr(self, param) or param in self.kwargs)}
                else:
                    if letter not in self.LETTERS_LAYERS.keys():
                        raise ValueError('Unknown letter symbol - %s' % letter)

                # Additional params for some layers
                if letter_group == 'd':
                    # Layers that behave differently during train/test
                    call_args.update({'training': training})
                elif letter_group.lower() == 'p':
                    # Choosing pooling operation
                    pool_op = 'mean' if letter.lower() == 'v' else self.kwargs.pop('pool_op', 'max')
                    args['op'] = pool_op
                elif letter_group == 'b':
                    # Additional layouts for all the upsampling layers
                    if self.kwargs.get('upsampling_layout'):
                        args['layout'] = self.kwargs.get('upsampling_layout')

                if not skip_layer:
                    args = {**args, **layer_args}
                    args = unpack_args(args, *layout_dict[letter_group])

                    with tf.variable_scope('layer-%d' % i):
                        tensor = layer_class(**args)(tensor, **call_args)

        # Allows to easily get output from graph by name
        tensor = tf.identity(tensor, name='_output')

        if context is not None:
            context.__exit__(None, None, None)

        return tensor


def update_layers(letter, func, name=None):
    """ Add custom letter to layout parsing procedure.

    Parameters
    ----------
    letter : str
        Letter to add.
    func : class
        Tensor-processing layer. Must have layer-like signature (both init and call overloaded).
    name : str
        Name of parameter dictionary. Defaults to `letter`.

    Examples
    --------
    Add custom `Q` letter::

        block = ConvBlock('cnap Q', filters=32, custom_params={'key': 'value'})
        block.add_letter('Q', my_func, 'custom_params')
        x = block(x)
    """
    name = name or letter
    ConvBlock.LETTERS_LAYERS.update({letter: name})
    ConvBlock.LAYERS_CLASSES.update({name: func})
    ConvBlock.LETTERS_GROUPS.update({letter: letter})



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

    def __call__(self, inputs, *args, **kwargs):
        if np.all(self.factor == 1):
            return inputs

        if 't' in self.layout or 'T' in self.layout:
            if 'kernel_size' not in self.kwargs:
                self.kwargs['kernel_size'] = self.factor
            if 'strides' not in kwargs:
                self.kwargs['strides'] = self.factor

        return ConvBlock(self.layout, factor=self.factor, shape=self.shape,
                         name=self.name, **self.kwargs)(inputs, *args, **kwargs)
