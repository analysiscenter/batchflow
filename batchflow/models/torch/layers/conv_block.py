""" Convenient combining block """
import logging
import inspect

import numpy as np
import torch
import torch.nn as nn

from .core import Dense, BatchNorm, Dropout, AlphaDropout
from .conv import Conv, ConvTranspose, DepthwiseConv, DepthwiseConvTranspose, \
                  SeparableConv, SeparableConvTranspose
from .pooling import Pool, GlobalPool
from .resize import IncreaseDim, Reshape, Interpolate, SubPixelConv, Combine
from .attention import SelfAttention
from .activation import Activation
from ..utils import get_shape
from ...utils import unpack_args
from .... import Config


logger = logging.getLogger(__name__)



class Branch(nn.Module):
    """ Add side branch to a :class:`~.layers.ConvBlock`. """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()

        if kwargs.get('layout'):
            self.layer = ConvBlock(inputs=inputs, **kwargs)
        else:
            self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)



class BaseConvBlock(nn.ModuleDict):
    """ Complex multi-dimensional block to apply sequence of different operations.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation:

        - `>` - add new axis to tensor
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
        - S - attention based on tensor itself (for example, squeeze and excitation)
        - b - upsample with bilinear resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:class:`~.layers.SubpixelConv`)
        - B, R - start a new branch with auxilliary :class:`~.layers.BaseConvBlock`
        - `.` - end the most recent created branch with concatenation
        - `+` - end the most recent created branch with summation
        - `*` - end the most recent created branch with multiplication
        - `&` - end the most recent created branch with softsum

        Default is ''.

    filters : int or str
        If str, then number of filters is calculated by its evaluation. `S` and `same` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor.
    kernel_size : int
        Convolution kernel size.
    name : str
        Name of the layer that will be used as a scope.
    units : int or str
        If str, then number of units is calculated by its evaluation. `S` and `same` stand for the
        number of units in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of units in the dense layer.
    strides : int
        Convolution stride.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_first'.
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
        - resize_bilinear - parameters for :class:`~.layers.Interpolate`.
        - self_attention - parameters for :class:`~.layers.SelfAttention`.
        - branch - parameters for :class:`~.layers.ConvBlock`.
        - branch_end - parameters for :class:`~.layers.Combine`.


    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.


    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = BaseConvBlock(layout='cnap', filters=32, kernel_size=3)

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = BaseConvBlock(layout='nac nac nac', filters=[64, 64, 256], kernel_size=[1, 3, 1])

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

        x = BaseConvBlock(layout='ca ca ca nd', filters=[32, 32, 64], kernel_size=[5, 3, 3],
                      strides=[1, 1, 2], dropout_rate=.15)

    A residual block::

        x = BaseConvBlock(layout='R nac +', filters='same')

    Squeeze and excitation block::

        x = BaseConvBlock(layout='S cna *', filters=64)

    """
    LETTERS_LAYERS = {
        'a': 'activation',
        'B': 'branch',
        'R': 'branch', # stands for `R`esidual
        '+': 'branch_end',
        '|': 'branch_end',
        '*': 'branch_end',
        '&': 'branch_end',
        '>': 'increase_dim',
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
        'S': 'self_attention',
        'b': 'resize_bilinear',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    LAYERS_MODULES = {
        'activation': Activation,
        'branch': Branch,
        'branch_end': Combine,
        'increase_dim': IncreaseDim,
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
        'self_attention': SelfAttention,
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
        'n': 'd',
        'N': 'b',
        })


    BRANCH_LETTERS = ['R', 'B']
    COMBINE_LETTERS = ['+', '*', '|', '&']

    def __init__(self, inputs=None, layout='',
                 filters=0, kernel_size=3, strides=1, dilation_rate=1, depth_multiplier=1,
                 activation='relu',
                 pool_size=2, pool_strides=2,
                 dropout_rate=0.,
                 padding='same', data_format='channels_first',
                 **kwargs):
        super().__init__()

        self.layout = layout
        self.device = inputs.device
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.dropout_rate = dropout_rate
        self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

        self._make_modules(inputs)


    def forward(self, x):
        branches = []

        for letter, layer in zip(self.layout, self.values()):
            if letter in self.BRANCH_LETTERS:
                branches += [layer(x)]
            elif letter in self.COMBINE_LETTERS:
                x = layer([x, branches.pop()])
            else:
                x = layer(x)
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
        args = unpack_args(args, *counters)
        args = {**args, **layer_args}
        return args

    def _make_modules(self, inputs):
        """ Create necessary modules from instance parameters. """
        self.layout = self.layout or ''
        self.layout = self.layout.replace(' ', '')
        if len(self.layout) == 0:
            logger.warning('BaseConvBlock: layout is empty, so there is nothing to do, just returning inputs.')

        layout_dict = {}
        for letter in self.layout:
            letter_group = self.LETTERS_GROUPS[letter]
            letter_counts = layout_dict.setdefault(letter_group, [-1, 0])
            letter_counts[1] += 1
        branches = []

        for i, letter in enumerate(self.layout):
            # Get the current layer configuration
            letter_group = self.LETTERS_GROUPS[letter]
            layer_name = self.LETTERS_LAYERS[letter]
            layer_class = self.LAYERS_MODULES[layer_name]
            layout_dict[letter_group][0] += 1

            # Skip-connection letters: parallel execution branch
            if letter in self.BRANCH_LETTERS:
                # Make layer arguments
                args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])

                # Create layer and store the output
                layer = layer_class(**args).to(self.device)
                skip = layer(inputs)
                branches.append(skip)

                # Create layer description
                shape_before = (None, *get_shape(inputs)[1:])
                shape_after = (None, *get_shape(skip)[1:])
                layer_desc = 'Layer {},    skip "{}": {} -> {}'.format(i, letter, shape_before, shape_after)

            # Combine multiple inputs with addition, concatenation, etc
            elif letter in self.COMBINE_LETTERS:
                # Make layer arguments: pop additional inputs from storage
                args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                combine_inputs = [inputs, branches.pop()]
                args = {**args, 'inputs': combine_inputs, 'op': letter}

                # Create layer
                layer = layer_class(**args).to(self.device)
                shape_before = [get_shape(item) for item in combine_inputs]
                inputs = layer(args['inputs'])
                shape_after = get_shape(inputs)

                # Create layer description: one line for each of the inputs
                shape_before = [str((None, *shape[1:])) for shape in shape_before]
                shape_after = (None, *shape_after[1:])
                layer_desc = 'Layer {}, combine "{}": {}'.format(i, letter, shape_before[0])
                for shape in shape_before[1:]:
                    layer_desc += '\n' + ' '*(len(layer_desc) - len(shape)) + shape
                layer_desc += ' -> {}'.format(shape_after)

            # Regular layer
            else:
                # Check if we need to skip current layer
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False \
                             or isinstance(layer_args, dict) and layer_args.get('disable', False)

                # Make layer argument
                if skip_layer:
                    pass
                elif letter in self.DEFAULT_LETTERS:
                    args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                elif letter not in self.LETTERS_LAYERS.keys():
                    raise ValueError('Unknown letter symbol - %s' % letter)

                # Additional params for some of the layers
                if letter_group.lower() == 'p':
                    args['op'] = letter
                elif letter_group == 'b':
                    args['mode'] = args.get('mode', letter.lower())

                if not skip_layer:
                    # Create layer
                    layer = layer_class(**args).to(self.device)
                    shape_before = get_shape(inputs)
                    inputs = layer(inputs)
                    shape_after = get_shape(inputs)

                    # Create layer description
                    shape_before, shape_after = (None, *shape_before[1:]), (None, *shape_after[1:])
                    layer_desc = 'Layer {},  letter "{}": {} -> {}'.format(i, letter, shape_before, shape_after)

            self.update([(layer_desc, layer)])

    def extra_repr(self):
        return 'layout={}\n'.format(self.layout)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)

    def __repr__(self):
        if getattr(self, 'short_repr', False):
            msg = f'layout={self.layout}\n'
            msg += '\n'.join([f'{key}' for key in self.keys()])
            return f'{msg}\n'
        return super().__repr__()



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
        block = BaseConvBlock('cnap Q', filters=32, custom_module_params={'key': 'value'})
        x = block(x)
    """
    name = name or letter
    BaseConvBlock.LETTERS_LAYERS.update({letter: name})
    BaseConvBlock.LAYERS_MODULES.update({name: module})
    BaseConvBlock.LETTERS_GROUPS.update({letter: letter})



class ConvBlock(nn.Sequential):
    """ Convenient wrapper for chaining/splitting multiple base blocks.

    Parameters
    ----------
    args : sequence
        Layers to be chained.
        If element of a sequence is a module, then it is used as is.
        If element of a sequence is a dictionary, then it is used as arguments of a layer creation.
        Function that is used as layer is either `base_block` or `base`/`base_block` keys inside the dictionary.

    base, base_block : nn.Module
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

    Repeat the whole construction two times::

    repeated = splitted * 2
    """
    def __init__(self, *args, inputs=None, base_block=BaseConvBlock, n_repeats=1, **kwargs):
        base_block = kwargs.pop('base', None) or base_block
        self.input_shape, self.device = get_shape(inputs), inputs.device
        self.base_block, self.n_repeats = base_block, n_repeats
        self.args, self.kwargs = args, kwargs

        self._make_modules(inputs)
        super().__init__(*self.layers)


    def _make_modules(self, inputs):
        layers = []
        for _ in range(self.n_repeats):
            layer = self._make_layer(*self.args, inputs=inputs, base_block=self.base_block, **self.kwargs)
            inputs = layer(inputs)
            layers.append(layer)
        self.layers = layers

    def _make_layer(self, *args, inputs=None, base_block=BaseConvBlock, **kwargs):
        # each element in `args` is a dict or module: make a sequential out of them
        if args:
            layers = []
            for item in args:
                if isinstance(item, dict):
                    block = item.pop('base_block', None) or item.pop('base', None) or base_block
                    block_args = {'inputs': inputs, **dict(Config(kwargs) + Config(item))}
                    layer = block(**block_args)
                    inputs = layer(inputs)
                    layers.append(layer)
                elif isinstance(item, nn.Module):
                    inputs = item(inputs)
                    layers.append(item)
                else:
                    raise ValueError('Positional arguments of ConvBlock must be either dicts or nn.Modules, \
                                      got instead {}'.format(type(item)))
            return nn.Sequential(*layers)
        # one block only
        return base_block(inputs=inputs, **kwargs)

    def _make_inputs(self):
        inputs = np.zeros(self.input_shape, dtype=np.float32)
        inputs = torch.from_numpy(inputs).to(self.device)
        return inputs


    def __mul__(self, other):
        inputs = self._make_inputs()

        layers = []
        for _ in range(other):
            layer = ConvBlock(*self.args, inputs=inputs, base_block=self.base_block,
                              n_repeats=self.n_repeats, **self.kwargs)
            inputs = layer(inputs)
            layers.append(layer)
        return nn.Sequential(*layers)

    def __repr__(self):
        if getattr(self, 'short_repr', False):
            if len(self) == 1:
                msg = self.__class__.__name__ + '\n'
                msg += torch.nn.modules.module._addindent(repr(self[0]), 2)
                return msg
        return super().__repr__()
