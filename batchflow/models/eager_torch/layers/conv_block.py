""" Convenient combining block """
import logging
import inspect
from collections import OrderedDict

import torch.nn as nn

from .core import Activation, Dense, BatchNorm, Dropout, AlphaDropout
from .conv import Conv, ConvTranspose, DepthwiseConv, DepthwiseConvTranspose, \
                  SeparableConv, SeparableConvTranspose
from .pooling import Pool, GlobalPool
from .resize import IncreaseDim, ReduceDim, Reshape, Interpolate, SubPixelConv, SideBlock, SEBlock, Combine
from ..utils import get_shape
from ...utils import unpack_args


logger = logging.getLogger(__name__)



class ConvBlock(nn.Module):
    """ Complex multi-dimensional block to apply sequence of different operations.

    Parameters
    ----------
    layout : str
        A sequence of letters, each letter meaning individual operation:

        - `>` - add new axis to tensor
        - `<` - remove trailing axis from tensor
        - r - reshape tensor to desired shape
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
        - b - upsample with bilinear resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:class:`~.layers.SubpixelConv`)
        - R - start residual connection
        - A - start residual connection with bilinear additive upsampling
        - S - start residual connection with squeeze and excitation
        - B - start residual connection with auxilliary :class:`~.layers.ConvBlock`
        - `.` - end residual connection with concatenation
        - `+` - end residual connection with summation
        - `*` - end residual connection with multiplication
        - `&` - end residual connection with softsum

        Default is ''.

    filters : int or str
        If str, then number of filters is calculated by its evaluation. `S` and `same` stand for the
        number of filters in the previous tensor.
        If int, then number of filters in the output tensor.
    kernel_size : int
        Convolution kernel size.
    name : str
        Name of the layer that will be used as a scope.
    units : int or str
        If str, then number of units is calculated by its evaluation. `S` and `same` stand for the
        number of units in the previous tensor.
        If int, then number of units in the dense layer.
    strides : int
        Convolution stride.
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

    other named arguments : None, bool, dict or sequence
        If None, then no common parameters are passed to all the layers of a given type.
        If False, then all the layers of a given type are disabled.
        If dict, then contains common parameters for all the layers of a given type. If 'disable' is present in
        this dictionary and evaluates to True, then all the layers of a given type are disabled.
        If sequence, then each element must be a dict with parameters that are passed to corresponding layers
        of a given type.


        Name of the argument must be one of:

        - dense - parameters like initializers, regularalizers, etc.
        - conv - parameters like initializers, regularalizers, etc.
        - transposed_conv - parameters like initializers, regularalizers, etc.
        - batch_norm - parameters like initializers, momentum, etc.
        - pooling - parameters like initializers, regularalizers, etc.
        - dropout - parameters like noise_shape, dropout_rate, etc.
        - subpixel_conv - parameters for :class:`~.layers.SubPixelConv`.
        - resize_bilinear - parameters for parameters for :class:`~.layers.Interpolate`.
        - residual_bilinear_additive - parameters for parameters for :class:`~.layers.Interpolate`.
        - residual_se - parameters for parameters for :class:`~.layers.SEBlock`.
        - side_branch - parameters for parameters for :class:`~.layers.ConvBlock`.


    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.


    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = ConvBlock(layout='cnap', filters=32, kernel_size=3)

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = ConvBlock(layout='nac nac nac', filters=[64, 64, 256], kernel_size=[1, 3, 1])

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

        x = ConvBlock(layout='ca ca ca nd', filters=[32, 32, 64], kernel_size=[5, 3, 3],
                      strides=[1, 1, 2], dropout_rate=.15)

    A residual block::

        x = ConvBlock(layout='R nac +', filters='same')

    Squeeze and excitation block::

        x = ConvBlock(layout='S cna *', filters=64)

    """
    LETTERS_LAYERS = {
        'a': 'activation',
        'R': 'residual_start',
        'A': 'residual_bilinear_additive',
        'B': 'side_branch', # formally, it is residual too
        'S': 'residual_se',
        '+': 'residual_end',
        '.': 'residual_end',
        '*': 'residual_end',
        '&': 'residual_end',
        '>': 'increase_dim',
        '<': 'reduce_dim',
        'r': 'reshape',
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
        # 'O': 'dropblock',
        # 'm': 'mip',
        'b': 'resize_bilinear',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    LAYERS_MODULES = {
        'activation': Activation,
        'residual_start': nn.Identity,
        'side_branch': SideBlock,
        'residual_se': SEBlock,
        'residual_end': Combine,
        'increase_dim': IncreaseDim,
        'reduce_dim': ReduceDim,
        'reshape': Reshape,
        'dense': Dense,
        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'depthwise_conv': DepthwiseConv,
        'depthwise_conv_transpose': DepthwiseConvTranspose,
        'pooling': Pool,
        'global_pooling': GlobalPool,
        'batch_norm': BatchNorm,
        'dropout': Dropout,
        'alpha_dropout': AlphaDropout,
        # 'dropblock': None, # TODO
        # 'mip': None, # TODO?
        'residual_bilinear_additive': Interpolate,
        'resize_bilinear': Interpolate,
        'resize_nn': Interpolate,
        'subpixel_conv': SubPixelConv,
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
        # 'O': 'd',
        'n': 'd',
        'A': 'b',
        'N': 'b',
        'X': 'b',
        })


    SKIP_LETTERS = ['R', 'A', 'B', 'S']
    COMBINE_LETTERS = ['+', '*', '.', '&']

    def __init__(self, inputs=None, layout='',
                 filters=0, kernel_size=3, strides=1, dilation_rate=1, depth_multiplier=1,
                 activation='relu',
                 pool_size=2, pool_strides=2,
                 dropout_rate=0.,
                 padding='same', data_format='channels_first',
                 **kwargs):
        super().__init__()

        self.layout = layout
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.dropout_rate = dropout_rate
        self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

        block_modules, skip_modules, combine_modules = self.parse_params(inputs)
        self.block_modules = block_modules
        self.skip_modules = skip_modules if skip_modules else None
        self.combine_modules = combine_modules if combine_modules else None

    def forward(self, x):
        b_counter, s_counter, c_counter = 0, 0, 0
        residuals = []

        for letter in self.module_layout:
            if letter == '_':
                x = self.block_modules[b_counter](x)
                b_counter += 1
            elif letter in self.SKIP_LETTERS:
                residuals += [self.skip_modules[s_counter](x)]
                s_counter += 1
            elif letter in self.COMBINE_LETTERS:
                x = self.combine_modules[c_counter]([residuals.pop(), x])
                c_counter += 1
        return x


    def fill_layer_params(self, layer_name, layer_class, inputs, counters):
        """ Inspect which parameters should be passed to the layer and get them from instance. """
        layer_params = inspect.getfullargspec(layer_class.__init__)[0]
        layer_params.remove('self')

        args = {param: getattr(self, param) if hasattr(self, param) else self.kwargs.get(param, None)
                for param in layer_params
                if (hasattr(self, param) or (param in self.kwargs))}
        if 'inputs' in layer_params:
            args['inputs'] = inputs

        layer_args = unpack_args(self.kwargs, *counters)
        layer_args = layer_args.get(layer_name, {})
        args = {**args, **layer_args}
        args = unpack_args(args, *counters)
        return args

    def parse_params(self, inputs):
        """ Create necessary ModuleLists from instance parameters. """
        self.module_layout = ''
        device = inputs.device

        layout = self.layout or ''
        layout = layout.replace(' ', '')
        if len(layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do, just returning inputs.')
            return nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        layout_dict = {}
        for letter in layout:
            letter_group = self.LETTERS_GROUPS[letter]
            letter_counts = layout_dict.setdefault(letter_group, [-1, 0])
            letter_counts[1] += 1

        modules, skip_modules, combine_modules = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layers, residuals = [], []

        for i, letter in enumerate(layout):
            letter_group = self.LETTERS_GROUPS[letter]
            layer_name = self.LETTERS_LAYERS[letter]
            layer_class = self.LAYERS_MODULES[layer_name]
            layout_dict[letter_group][0] += 1

            if letter in self.SKIP_LETTERS + self.COMBINE_LETTERS:
                if len(layers) >= 1:
                    self.module_layout += '_'
                    modules.append(nn.Sequential(OrderedDict(layers)))
                    layers = []
                self.module_layout += letter

                if letter in self.SKIP_LETTERS:
                    args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])

                    layer = layer_class(**args).to(device)
                    skip = layer(inputs)
                    residuals.append(skip)

                    layer_desc = 'Layer {}, skip-letter "{}"; {} -> {}'.format(i, letter,
                                                                               get_shape(inputs),
                                                                               get_shape(skip))
                    layer = nn.Sequential(OrderedDict([(layer_desc, layer)]))
                    skip_modules.append(layer)

                elif letter in self.COMBINE_LETTERS:
                    args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                    args['inputs'] = [residuals.pop(), inputs]
                    layer = layer_class(op=letter, **args).to(device)
                    shape_before = get_shape(inputs)
                    inputs = layer(args['inputs'])
                    shape_after = get_shape(inputs)

                    shape_before, shape_after = (None, *shape_before[1:]), (None, *shape_after[1:])
                    layer_desc = 'Layer {}: combine; {} -> {}'.format(i, shape_before, shape_after)
                    layer = nn.Sequential(OrderedDict([(layer_desc, layer)]))
                    combine_modules.append(layer)
            else:
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False \
                             or isinstance(layer_args, dict) and layer_args.get('disable', False)

                # Create params for the layer call
                if skip_layer:
                    pass
                elif letter in self.DEFAULT_LETTERS:
                    args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                elif letter not in self.LETTERS_LAYERS.keys():
                    raise ValueError('Unknown letter symbol - %s' % letter)

                # Additional params for some layers
                if letter_group.lower() == 'p':
                    args['op'] = letter
                elif letter_group == 'b':
                    args['mode'] = args.get('mode', letter.lower())

                if not skip_layer:
                    layer = layer_class(**args).to(device)

                    shape_before = get_shape(inputs)
                    inputs = layer(inputs)
                    shape_after = get_shape(inputs)

                    shape_before, shape_after = (None, *shape_before[1:]), (None, *shape_after[1:])
                    layer_desc = 'Layer {}, letter "{}"; {} -> {}'.format(i, letter, shape_before, shape_after)
                    layers.append((layer_desc, layer))

        if len(layers) > 0:
            self.module_layout += '_'
            modules.append(nn.Sequential(OrderedDict(layers)))

        return modules, skip_modules, combine_modules


def update_layers(letter, module, name=None):
    """ Add custom letter to layout parsing procedure.

    Parameters
    ----------
    letter : str
        Letter to add.
    module : :class:`torch.nn.Module`
        Tensor-processing layer. Must have layer-like signature (both init and forward methods overloaded).
    name : str
        Name of parameter dictionary. Defaults to `letter`.

    Examples
    --------
    Add custom `Q` letter::

        block.add_letter('Q', my_module, 'custom_module_params')
        block = ConvBlock('cnap Q', filters=32, custom_module_params={'key': 'value'})
        x = block(x)
    """
    name = name or letter
    ConvBlock.LETTERS_LAYERS.update({letter: name})
    ConvBlock.LAYERS_MODULES.update({name: module})
    ConvBlock.LETTERS_GROUPS.update({letter: letter})
