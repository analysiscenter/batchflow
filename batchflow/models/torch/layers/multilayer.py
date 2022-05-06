""" Convenient combining block """
import inspect

from torch import nn

from .core import Dense, DenseAlongAxis, BatchNorm, LayerNorm, Dropout, AlphaDropout
from .conv import Conv, ConvTranspose, DepthwiseConv, DepthwiseConvTranspose, \
                  SeparableConv, SeparableConvTranspose
from .pooling import Pool, GlobalPool
from .resize import IncreaseDim, Reshape, Interpolate, SubPixelConv
from .activation import Activation
from .combine import Combine
from .wrapper_letters import Branch, AttentionWrapper

from ..repr_mixin import ModuleDictReprMixin
from ..utils import get_shape
from ...utils import unpack_args




class MultiLayer(ModuleDictReprMixin, nn.ModuleDict):
    """ !!. """
    LETTERS_LAYERS = {
        'a': 'activation_layer',
        'B': 'branch',
        'R': 'branch', # stands for `R`esidual
        '+': 'branch_end',
        '|': 'branch_end',
        '*': 'branch_end',
        '!': 'branch_end',
        '>': 'increase_dim',
        'r': 'reshape',
        'f': 'dense',
        'F': 'dense_along_axis',
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
        'l': 'layer_norm',
        'd': 'dropout',
        'D': 'alpha_dropout',
        'S': 'self_attention',
        'b': 'resize_bilinear',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    LAYERS_MODULES = {
        'activation_layer': Activation,
        'branch': Branch,
        'branch_end': Combine,
        'increase_dim': IncreaseDim,
        'reshape': Reshape,
        'dense': Dense,
        'dense_along_axis': DenseAlongAxis,
        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'depthwise_conv': DepthwiseConv,
        'depthwise_conv_transpose': DepthwiseConvTranspose,
        'pooling': Pool,
        'global_pooling': GlobalPool,
        'batch_norm': BatchNorm,
        'layer_norm': LayerNorm,
        'dropout': Dropout,
        'alpha_dropout': AlphaDropout,
        'self_attention': AttentionWrapper,
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
        'F': 'f',
    })


    BRANCH_LETTERS = ['R', 'B']
    COMBINE_LETTERS = ['+', '*', '|', '!']

    VERBOSITY_THRESHOLD = 4

    def __init__(self, inputs=None, layout='', **kwargs):
        super().__init__()

        self.layout = layout
        self.device = inputs.device
        # self.channels, self.kernel_size, self.stride = channels, kernel_size, stride
        # self.dilation, self.depth_multiplier = dilation, depth_multiplier
        # self.activation = activation
        # self.pool_size, self.pool_stride = pool_size, pool_stride
        # self.dropout_rate = dropout_rate
        # self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

        self._make_modules(inputs)

        # ModuleDict uses not only the keys, manually put in it, but also every `nn.Module` attribute
        # To combat that, we manually remove keys, corresponding to passed parameters that can be nn.Modules
        if 'activation' in self._modules:
            self._modules.pop('activation')


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
        self.shapes = {}
        self.layout = self.layout or ''
        self.layout = self.layout.replace(' ', '')
        # TODO: warning on empty layout

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

            if letter in self.BRANCH_LETTERS:
                # Skip-connection letters: parallel execution branch
                # Make layer arguments
                args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])

                # Create layer and store the output
                layer = layer_class(**args).to(self.device)
                skip = layer(inputs)
                branches.append(skip)

                # Create layer description
                layer_name = f'Layer {i},    skip "{letter}"'
                output_shapes = get_shape(skip)

            elif letter in self.COMBINE_LETTERS:
                # Combine multiple inputs with addition, concatenation, etc
                # Make layer arguments: pop additional inputs from storage
                args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                combine_inputs = [branches.pop(), inputs]
                args = {'op': letter, **args, 'inputs': combine_inputs}

                # Create layer
                input_shapes = get_shape(combine_inputs)
                layer = layer_class(**args).to(self.device)
                inputs = layer(combine_inputs)

                # Create layer description: one line for each of the inputs
                output_shapes = get_shape(inputs)
                layer_name = f'Layer {i}, combine "{letter}"'

            else:
                # Regular layer
                # Check if we need to skip current layer. #TODO: remove this behavior?
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False \
                             or isinstance(layer_args, dict) and layer_args.get('disable', False)

                # Make layer argument
                if skip_layer:
                    pass
                elif letter in self.DEFAULT_LETTERS:
                    args = self.fill_layer_params(layer_name, layer_class, inputs, layout_dict[letter_group])
                elif letter not in self.LETTERS_LAYERS.keys():
                    raise ValueError(f'Unknown letter symbol "{letter}"!')

                # Additional params for some of the layers
                if letter_group.lower() == 'p':
                    args['op'] = letter
                elif letter_group == 'b':
                    args['mode'] = args.get('mode', letter.lower())

                if not skip_layer:
                    # Create layer
                    layer = layer_class(**args).to(self.device)
                    inputs = layer(inputs)

                    # Create layer description
                    output_shapes = get_shape(inputs)
                    layer_name = f'Layer {i},  letter "{letter}"'

            self[layer_name] = layer
            self.shapes[layer_name] = (input_shapes, output_shapes)

    def extra_repr(self):
        return f'layout={self.layout}'



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
        block = BaseConvBlock('cnap Q', channels=32, custom_module_params={'key': 'value'})
        x = block(x)
    """
    name = name or letter
    MultiLayer.LETTERS_LAYERS.update({letter: name})
    MultiLayer.LAYERS_MODULES.update({name: module})
    MultiLayer.LETTERS_GROUPS.update({letter: letter})
