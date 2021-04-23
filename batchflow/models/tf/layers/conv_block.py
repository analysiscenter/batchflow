""" Contains layers, that are used as separate letters in layout convention, as well as convenient combining block. """
import inspect
import logging
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as K # pylint: disable=import-error

from .core import Activation, Dense, Dropout, AlphaDropout, BatchNormalization, Combine, Mip
from .conv import Conv, ConvTranspose, SeparableConv, SeparableConvTranspose, DepthwiseConv, DepthwiseConvTranspose
from .pooling import Pooling, GlobalPooling
from .drop_block import Dropblock
from .resize import ResizeBilinearAdditive, ResizeBilinear, ResizeNn, SubpixelConv, IncreaseDim, Reshape
from .layer import add_as_function, Layer
from ..utils import get_num_channels, get_spatial_dim, get_channels_axis
from ...utils import unpack_args
from .... import Config


logger = logging.getLogger(__name__)



class Branch:
    """ Add side branch to a :class:`~.tf.layers.ConvBlock`.
    Used for `R` letter in layout convention of :class:`~.tf.layers.ConvBlock`.
    """
    def __init__(self, *args, name='branch', **kwargs):
        self.name = name
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            if self.kwargs.get('layout') is not None:
                return ConvBlock(self.args, **self.kwargs)(inputs)
            return inputs



class SelfAttention:
    """ Adds attention based on tensor itself.
    Used for `S` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    attention_mode : callable or str
        Operation to apply to tensor to generate output.
        If callable, then directly applied to tensor.
        If str, then one of predefined: 'se', 'scse'.
    """
    @staticmethod
    def squeeze_and_excitation(inputs, ratio=16, name='se', **kwargs):
        """ Squeeze and excitation operation.

        Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

        Parameters
        ----------
        ratio : int
            Squeeze ratio for the number of filters.
        """
        data_format = kwargs.get('data_format')
        in_filters = get_num_channels(inputs, data_format)

        activation = kwargs.pop('activation', None)
        if isinstance(activation, list) and len(activation) == 2:
            pass
        elif callable(activation):
            activation = [activation, tf.nn.sigmoid]
        else:
            activation = [tf.nn.relu, tf.nn.sigmoid]

        kwargs = {**kwargs, 'layout': 'Vfafa',
                  'units': [in_filters//ratio, in_filters],
                  'activation': activation,
                  'name': name}
        x = ConvBlock(**kwargs)(inputs)

        shape = [-1] + [1] * (get_spatial_dim(inputs) + 1)
        axis = get_channels_axis(data_format)
        shape[axis] = in_filters
        scale = tf.reshape(x, shape)
        x = inputs * scale
        return x

    @staticmethod
    def scse(inputs, ratio=2, **kwargs):
        """ Concurrent spatial and channel squeeze and excitation.

        Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
        in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

        Parameters
        ----------
        ratio : int, optional
            Squeeze ratio for the number of filters in spatial squeeze and channel excitation block.
        """
        cse = SelfAttention.squeeze_and_excitation(inputs, ratio, name='cse', **kwargs)

        kwargs = {**kwargs,
                  'layout': 'ca', 'filters': 1, 'kernel_size': 1,
                  'activation': tf.nn.sigmoid, 'name': 'sse'}
        x = ConvBlock(**kwargs)(inputs)
        return cse + tf.multiply(x, inputs)

    ATTENTIONS = {
        squeeze_and_excitation: ['se', 'squeeze_and_excitation'],
        scse: ['scse'],
    }
    ATTENTIONS = {alias: getattr(method, '__func__') for method, aliases in ATTENTIONS.items() for alias in aliases}

    def __init__(self, attention_mode='se', data_format='channels_last', name='attention', **kwargs):
        self.data_format, self.name = data_format, name
        self.attention_mode, self.kwargs = attention_mode, kwargs

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            if self.attention_mode in self.ATTENTIONS:
                op = self.ATTENTIONS[self.attention_mode]
            elif callable(self.attention_mode):
                op = self.attention_mode
            else:
                raise ValueError('Attention mode must be a callable or one from {}, instead got {}.'
                                 .format(list(self.ATTENTIONS.keys()), self.attention_mode))

            return op(inputs, data_format=self.data_format, **self.kwargs)



@add_as_function
class BaseConvBlock:
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
        - S - self attention
        - O - dropblock
        - m - maximum intensity projection (:class:`~.layers.Mip`)
        - > - increase tensor dimensionality
        - r - reshape tensor
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

    branch : dict
        Parameters for residual branches, that are passed directly to :class:`~.tf.layers.ConvBlock`.
    residual_end : dict
        Parameters for combining main flow and side branches. Passed directly to :class:`~.tf.layers.Combine`.
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
    self_attention : dict
        Parameters for self attention layers, like attention mode, ratio, etc.
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
        'S': 'self_attention',
        'O': 'dropblock',
        'm': 'mip',
        '>': 'increase_dim',
        'r': 'reshape',
        'b': 'resize_bilinear',
        'B': 'resize_bilinear_additive',
        'N': 'resize_nn',
        'X': 'subpixel_conv',
        'R': 'residual_start',
        'A': 'residual_bilinear_additive',
        '+': 'residual_end',
        '|': 'residual_end',
        '*': 'residual_end',
        '&': 'residual_end',
    }

    LAYERS_CLASSES = {
        'activation': Activation,
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
        'self_attention': SelfAttention,
        'mip': Mip,
        'increase_dim': IncreaseDim,
        'reshape': Reshape,
        'resize_bilinear': ResizeBilinear,
        'resize_bilinear_additive': ResizeBilinearAdditive,
        'resize_nn': ResizeNn,
        'subpixel_conv': SubpixelConv,
        'residual_start': Branch,
        'residual_end': Combine,
        'residual_bilinear_additive': ResizeBilinearAdditive,
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

    SKIP_LETTERS = ['R', 'A', 'B']
    COMBINE_LETTERS = ['+', '*', '|', '&']

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


    def fill_layer_params(self, layer_name, layer_class, counters):
        """ Inspect which parameters should be passed to the layer and get them from instance. """
        if hasattr(layer_class, 'params'):
            layer_params = layer_class.params
        else:
            layer_params = inspect.getfullargspec(layer_class.__init__)[0]
            layer_params.remove('self')
            if 'name' in layer_params:
                layer_params.remove('name')

        args = {param: getattr(self, param, self.kwargs.get(param, None))
                for param in layer_params
                if (hasattr(self, param) or param in self.kwargs)}

        layer_args = unpack_args(self.kwargs, *counters)
        layer_args = layer_args.get(layer_name, {})
        args = {**args, **layer_args}
        args = unpack_args(args, *counters)
        return args


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

            if letter in self.DEFAULT_LETTERS:
                args = self.fill_layer_params(layer_name, layer_class, layout_dict[letter_group])
            else:
                args = {}

            if letter in self.SKIP_LETTERS:
                skip = layer_class(**args)(tensor)
                residuals.append(skip)
            elif letter in self.COMBINE_LETTERS:
                tensor = layer_class(**args)([tensor, residuals.pop()])
            else:
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False or \
                             isinstance(layer_args, dict) and layer_args.get('disable', False)

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
class ConvBlock:
    """ Convenient wrapper for chaining multiple base blocks.

    Parameters
    ----------
    args : sequence
        Layers to be chained.
        If element of a sequence is a module, then it is used as is.
        If element of a sequence is a dictionary, then it is used as arguments of a layer creation.
        Function that is used as layer is either `base_block` or `base`/`base_block` keys inside the dictionary.

    base, base_block : Layer or K.Layer
        Tensor processing function.

    n_repeats : int
        Number of times to repeat the whole block.

    kwargs : dict
        Default arguments for layers creation in case of dicts present in `args`.

    Examples
    --------
    Simple encoder that reduces spatial dimensions by 32 times and increases number
    of features to maintain the same tensor size::

    layer = ConvBlock({layout='cnap', filters='same*2'}, inputs=inputs, n_repeats=5)
    """
    def __init__(self, *args, base_block=BaseConvBlock, n_repeats=1, **kwargs):
        base_block = kwargs.pop('base', None) or base_block
        self.n_repeats = n_repeats
        self.base_block = base_block
        self.args, self.kwargs = args, kwargs


    def __call__(self, inputs, training=None):
        for _ in range(self.n_repeats):
            inputs = self._apply_layers(*self.args, inputs=inputs, base_block=self.base_block, **self.kwargs)
        return inputs

    def _apply_layers(self, *args, inputs=None, base_block=BaseConvBlock, **kwargs):
        # each element in `args` is a dict or layer: make a sequential out of them
        if args:
            for item in args:
                if isinstance(item, dict):
                    block = item.pop('base_block', None) or item.pop('base', None) or base_block
                    block_args = {'inputs': inputs, **dict(Config(kwargs) + Config(item))}
                    inputs = block(**block_args)(inputs)
                elif isinstance(item, (Layer, K.Layer)):
                    inputs = item(inputs)
                else:
                    raise ValueError('Positional arguments of ConvBlock must be either dicts or nn.Modules, \
                                      got instead {}'.format(type(item)))
        else: # one block only
            if isinstance(base_block, type):
                inputs = base_block(inputs=inputs, **kwargs)(inputs)
            elif callable(base_block):
                inputs = base_block(inputs=inputs, **kwargs)
        return inputs
