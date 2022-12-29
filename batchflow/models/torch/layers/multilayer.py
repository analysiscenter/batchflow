""" Convenient combining block """
import inspect

import torch
from torch import nn

from .core import Dense, DenseAlongAxis, Dropout, AlphaDropout
from .normalization import Normalization
from .conv import (Conv, ConvTranspose,
                   DepthwiseConv, DepthwiseConvTranspose, SeparableConv, SeparableConvTranspose)
from .conv_complex import (MultiKernelConv, SharedKernelConv, AvgPoolConv, BilinearConvTranspose,
                           MultiScaleConv, DeformableConv2d)
from .pooling import AvgPool, MaxPool, GlobalAvgPool, GlobalMaxPool
from .resize import IncreaseDim, Reshape, Interpolate
from .activation import Activation
from .combine import Combine
from .pixel_shuffle import PixelShuffle, PixelUnshuffle
from .hamburger import Hamburger
from .wrapper_letters import Branch, AttentionWrapper

from ..repr_mixin import ModuleDictReprMixin
from ..utils import make_initialization_inputs, get_device, get_shape
from ...utils import unpack_args




class MultiLayer(ModuleDictReprMixin, nn.ModuleDict):
    """ Chain multiple layers together in a sequential manner.

    The main idea of this class is to create a layer for each letter in `layout` string,
    while using other supplied parameters for their initialization. For example::
    >>> MultiLayer(inputs=inputs, layout='cna', channels=17, activation='GELU')
    Creates a sequence of convolutional, batch normalization and activation layers.
    `channels` are used to initialize the first one.

    Layout can contain whitespaces to improve readability.

    If there are multiple occurences of the same letter in layout, parameters can be sequences, for example::
    >>> MultiLayer(inputs=inputs, layout='cnac', channels=[17, 26], activation='GELU')
    If the parameter is not a sequence, the same value is used across all layers.

    Some letters are merged into groups. Inside each group, the parameters are shared.
    This way, regular, transposed and separable convolutions share one parameter `channels` as a sequence::
    >>> MultiLayer(inputs=inputs, layout='cnaC', channels=[17, 26], activation='GELU')

    In order to pass some parameters only to a certain layer inside a group, one can use the name of this letter
    as keyword argument. This can also be used to avoid cluttering and confusion::
    >>> MultiLayer(inputs=inputs, layout='cnaC', channels=17, activation='GELU',
                   separable_conv={'channels': 26}, batch_norm={'eps': 1e-03, 'momentum': 0.1})

    Some of the letters allow for non-sequential tensor flow (sic!):
        - `R` and `B` letters start the residual connection by storing a tensor.
        It can be parametrized to include another multilayer inside of it.
        To do so, use `branch` keyword parameter to pass parameters of inner multilayer.
        By default, the residual branch does not include any operations and equivalent to identity op.

        - `+`, `|`, `*`, `!` letters end the residual connection.
        They take the last stored tensor and combine it with the current main flow.
        The operation for combination is defined by the exact letter: sum, concat, mul or droppath.
        `branch_end` keyword can be used to pass parameters to the combination function.

    In order to initialize all of the layers, we need an example of tensor to be used as inputs for this module.
    It can be either a torch.Tensor or a tuple with its shape. In case of tuple, one can also specify `device` to use.

    Under the hood, we inspect layer constructors to find which parameters should be passed for layer creation.
    If we see `inputs` argument, we also provide an example of input tensor to the layer.
    """
    LETTERS_LAYERS = {
        # Core
        'a': 'activation_layer',
        'f': 'dense',
        'F': 'dense_along_axis',
        'n': 'normalization',
        'l': 'layer_norm',
        'd': 'dropout',
        'D': 'alpha_dropout',

        # Conv
        'c': 'conv',
        't': 'transposed_conv',
        'C': 'separable_conv',
        'T': 'separable_conv_transpose',
        'w': 'depthwise_conv',
        'W': 'depthwise_conv_transpose',
        'k': 'shared_kernel_conv',
        'K': 'multi_kernel_conv',
        'q': 'avg_pool_conv',
        'Q': 'bilinear_conv_transpose',
        'm': 'multi_scale_conv',
        'y': 'deformable_conv2d',
        # Downsample / upsample
        'v': 'avg_pool',
        'p': 'max_pool',
        'V': 'global_avg_pool',
        'P': 'global_max_pool',
        'x': 'pixel_unshuffle',

        'b': 'resize_bilinear',
        'X': 'pixel_shuffle',

        # Shapes
        '>': 'increase_dim',
        'r': 'reshape',

        # Branches
        'B': 'branch',
        'R': 'branch',
        '+': 'branch_end',
        '|': 'branch_end',
        '*': 'branch_end',
        '!': 'branch_end',

        # Wrapper
        'S': 'self_attention',
        'H': 'hamburger',
    }

    LAYERS_MODULES = {
        'activation_layer': Activation,
        'dense': Dense,
        'dense_along_axis': DenseAlongAxis,
        'normalization': Normalization,
        'dropout': Dropout,
        'alpha_dropout': AlphaDropout,

        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'depthwise_conv': DepthwiseConv,
        'depthwise_conv_transpose': DepthwiseConvTranspose,
        'multi_kernel_conv': MultiKernelConv,
        'shared_kernel_conv': SharedKernelConv,
        'avg_pool_conv': AvgPoolConv,
        'bilinear_conv_transpose': BilinearConvTranspose,
        'multi_scale_conv': MultiScaleConv,
        'deformable_conv2d': DeformableConv2d,

        'avg_pool': AvgPool,
        'max_pool': MaxPool,
        'global_avg_pool': GlobalAvgPool,
        'global_max_pool': GlobalMaxPool,
        'pixel_unshuffle': PixelUnshuffle,

        'pixel_shuffle': PixelShuffle,
        'resize_bilinear': Interpolate,

        'increase_dim': IncreaseDim,
        'reshape': Reshape,

        'branch': Branch,
        'branch_end': Combine,
        'self_attention': AttentionWrapper,
        'hamburger': Hamburger,
    }

    DEFAULT_LETTERS = LETTERS_LAYERS.keys()
    LETTERS_GROUPS = dict(zip(DEFAULT_LETTERS, DEFAULT_LETTERS))
    LETTERS_GROUPS.update({
        'C': 'c', 't': 'c', 'T': 'c', 'w': 'c', 'W': 'c', 'y': 'c',
        'k': 'c', 'K': 'c', 'q': 'c', 'Q': 'c', 'm': 'c',
        'v': 'p',
        'V': 'P',
        'D': 'd',
        'n': 'd',
        'F': 'f',
    })


    BRANCH_LETTERS = ['R', 'B']
    COMBINE_LETTERS = ['+', '*', '|', '!']

    VERBOSITY_THRESHOLD = 4

    def __init__(self, inputs=None, layout='', device=None, **kwargs):
        super().__init__()
        inputs = make_initialization_inputs(inputs, device=device)
        self.device = get_device(inputs)

        self.shapes = {}
        self.configuration = {}
        self.layout = layout
        self.kwargs = kwargs
        self.initialize(inputs)


    def fill_layer_params(self, layer_name, layer_class, inputs, counters):
        """ Inspect which parameters should be passed to the layer and get them from stored `kwargs`. """
        layer_params = inspect.getfullargspec(layer_class.__init__)[0]
        layer_params.remove('self')

        args = {param: self.kwargs.get(param, None) for param in layer_params if param in self.kwargs}
        if 'inputs' in layer_params:
            args['inputs'] = inputs

        layer_args = unpack_args(self.kwargs, *counters)
        layer_args = layer_args.get(layer_name, {})
        args = unpack_args(args, *counters)
        args = {**args, **layer_args}
        return args

    def initialize(self, inputs):
        """ Create necessary modules from instance parameters. """
        # TODO: warning on empty layout?
        self.layout = self.layout or ''

        only_ascii_letters = self.layout.encode(encoding='ascii', errors='ignore').decode(encoding='ascii')
        non_ascii_letters = set(self.layout) - set(only_ascii_letters)
        if non_ascii_letters:
            raise ValueError(f'Layout `{self.layout}` contains non ASCII letters {non_ascii_letters}!')
        self.layout = self.layout.replace(' ', '').replace('(', '').replace(')', '')

        layout_dict = {}
        for letter in self.layout:
            letter_group = self.LETTERS_GROUPS[letter]
            letter_counts = layout_dict.setdefault(letter_group, [-1, 0])
            letter_counts[1] += 1
        branches = []

        for i, letter in enumerate(self.layout):
            input_shapes = get_shape(inputs)

            # Get the current layer configuration
            letter_group = self.LETTERS_GROUPS[letter]
            layer_name = self.LETTERS_LAYERS[letter]
            layer_class = self.LAYERS_MODULES[layer_name]
            layout_dict[letter_group][0] += 1

            # Make args by inspecting layer constructor signature and matching it with `self.kwargs`
            args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])

            # Skip-connection letters: parallel execution branch
            if letter in self.BRANCH_LETTERS:
                # Create layer and store the output
                layer = layer_class(**args).to(self.device)
                skip = layer(inputs)
                branches.append(skip)

                # Create layer description
                output_shapes = get_shape(skip)
                layer_name = f'Layer {i},    skip "{letter}"'

            # Combine multiple inputs with addition, concatenation, etc
            elif letter in self.COMBINE_LETTERS:
                # Update inputs with the latest stored tensor
                combine_inputs = [branches.pop(), inputs]
                args = {'op': letter, **args, 'inputs': combine_inputs}

                # Create layer
                input_shapes = get_shape(combine_inputs)
                layer = layer_class(**args).to(self.device)
                inputs = layer(combine_inputs)

                # Create layer description: one line for each of the inputs
                output_shapes = get_shape(inputs)
                layer_name = f'Layer {i}, combine "{letter}"'

            # Regular layer
            else:
                # Create layer
                layer = layer_class(**args).to(self.device)
                inputs = layer(inputs)

                # Create layer description
                output_shapes = get_shape(inputs)
                layer_name = f'Layer {i},  letter "{letter}"'

            self[layer_name] = layer
            self.shapes[layer_name] = (input_shapes, output_shapes)

            self.configuration[(i, letter)] = {
                'layer_name': layer_name,
                'letter_group': letter_group,
                'layer_class': layer_class,
                'indexer': layout_dict[letter_group][:],
                'args': {key : value for key, value in args.items()
                         if key != 'inputs' and not isinstance(value, torch.Tensor)},
            }

    def forward(self, x):
        branches = []

        for letter, layer in zip(self.layout, self.values()):
            if letter in self.BRANCH_LETTERS:
                branches += [layer(x)]
            elif letter in self.COMBINE_LETTERS:
                x = layer([branches.pop(), x])
            else:
                x = layer(x)
        return x

    def extra_repr(self):
        return f'layout={self.layout}'


    @classmethod
    def add_letter(cls, letter, module, name=None):
        """ Add new letter to layout parsing procedure.

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
        Add custom `E` letter::

            MultiLayer.add_letter('E', my_module, 'custom_module_params')
            block = MultiLayer(inputs=x, layout='cnap E', channels=32,
                               custom_module_params={'key': 'value'})
            x = block(x)
        """
        name = name or letter
        cls.LETTERS_LAYERS[letter] = name
        cls.LAYERS_MODULES[name] = module
        cls.LETTERS_GROUPS[letter] = letter
