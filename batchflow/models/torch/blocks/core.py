""" Blocks: large parts that implement idea/named entity from popular articles. """
from torch import nn

from ..repr_mixin import ModuleDictReprMixin
from ..layers import MultiLayer
from ..utils import get_shape, make_initialization_inputs
from .... import Config



class Block(ModuleDictReprMixin, nn.ModuleDict):
    """ Convenient wrapper for chaining multiple base large blocks.
    Serves as a link between smaller, like individual layers and their sequences
    and larger, like named blocks and stages, model parts.

    Depending on the supplied arguments, does the following:
        - no positional `args`, no explicit `base_block`: the same as :class:`~.torch.layers.MultiLayer`.
        - no positional `args`,    explicit `base_block`: uses `base_block` to construct nn.Module from the rest kwargs
        This way `Block` acts as a selector of different constructors.

        -    positional `args: each element of `args` is either a ready-to-use nn.Module, or
        a dictionary with parameters to create one. If no explicit `base_block` is provided, dictionaries are processed
        by :class:`~.torch.layers.MultiLayer`; otherwise, `base_block` is used as constructor.
        This type of initialization is mostly used in directly inherited blocks,
        see :class:`~.torch.blocks.ResBlock`, :class:`~.torch.blocks.DenseBlock` for examples.

        - `n_repeats` keyword-only argument can be used to repeat the same logical structure multiple times.
        This way, we can create entire `Stages` of some popular named networks with one additional parameter.

    Parameters
    ----------
    args : sequence
        Layers to be chained.
        If element of a sequence is a module, then it is used as is.
        If element of a sequence is a dictionary, then it is used as arguments of a block creation.
        Function that is used as block constructor is either `base_block` or `base_block` key inside the dictionary.

    base_block : type
        Constructor for blocks. By default, uses :class:`~.torch.layers.MultiLayer` to chain multiple simple layers.

    n_repeats : int
        Number of times to repeat the whole block.

    kwargs : dict
        Default arguments for layers creation in case of dicts present in `args`.

    Examples
    --------
    Simple encoder that reduces spatial dimensions by 32 times and increases number
    of features to maintain the same tensor size::

        layer = Block({layout='cnap', channels='same*2'}, inputs=inputs, n_repeats=5)
    """
    VERBOSITY_THRESHOLD = 3

    def __init__(self, *args, inputs=None, base_block=MultiLayer, n_repeats=1, **kwargs):
        super().__init__()
        inputs = make_initialization_inputs(inputs, device=kwargs.get('device'))

        self.input_shape, self.device = get_shape(inputs), inputs.device
        self.base_block, self.n_repeats = base_block, n_repeats
        self.args, self.kwargs = args, kwargs

        self.shapes = {}
        self.initialize(inputs, base_block, n_repeats, *args, **kwargs)


    def initialize(self, inputs, base_block, n_repeats, *args, **kwargs):
        """ Construct blocks. If needed, repeat them multiple times. """
        for repeat in range(n_repeats):
            if args:
                for i, item in enumerate(args):
                    # Make block
                    if isinstance(item, dict):
                        block_constructor = item.pop('base_block', None) or base_block
                        block_args = {'inputs': inputs, **dict(Config(kwargs) + Config(item))}
                        block = block_constructor(**block_args)
                    elif isinstance(item, nn.Module):
                        block = item
                    else:
                        raise ValueError(f'Positional arguments of Block must be either dicts or nn.Modules, '
                                         f'got {type(item)} instead!')

                    inputs = self.initialize_block(inputs, block, f'repeat{repeat}-args{i}')

            else:
                # Make block
                block = base_block(inputs=inputs, **kwargs)
                inputs = self.initialize_block(inputs, block, f'repeat{repeat}')

    def initialize_block(self, inputs, block, block_name):
        """ Construct one block. """
        input_shapes = get_shape(inputs)
        inputs = block(inputs)
        output_shapes = get_shape(inputs)

        self[block_name] = block
        self.shapes[block_name] = (input_shapes, output_shapes)
        return inputs


    def forward(self, x):
        for layer in self.values():
            x = layer(x)
        return x


    @classmethod
    def make(cls, *args, **kwargs):
        """ Make a block without additional level of nestedness. """
        if not args and 'n_repeats' not in kwargs:
            return kwargs.pop('base_block', MultiLayer)(**kwargs)
        return cls(*args, **kwargs)


class DefaultBlock(Block):
    """ Block with default layout: convolution, normalization, activation. """
    DEFAULTS = {
        'layout': 'cna',
        'channels': 'same',
        'kernel_size': 3,
    }
    def __init__(self, inputs=None, **kwargs):
        kwargs = {**self.DEFAULTS, **kwargs}
        super().__init__(inputs=inputs, **kwargs)


class Upsample(Block):
    """ Block with additional defaults for upsampling layers.

    Parameters
    ----------
    factor : int
        The upsampling factor to apply.
    """
    def __init__(self, inputs=None, layout='b', factor=2, shape=None, **kwargs):
        if kwargs.get('base_block') is None:
            if set('b').intersection(layout):
                kwargs = {
                    'scale_factor': factor,
                    **kwargs
                }
            elif set('X').intersection(layout):
                kwargs = {
                    'upscale_factor': factor,
                    **kwargs
                }
            elif set('tT').intersection(layout):
                kwargs = {
                    'kernel_size': 2*factor - 1,
                    'stride': factor,
                    'channels': 'same',
                    **kwargs
                }
        super().__init__(inputs=inputs, layout=layout, shape=shape, **kwargs)


class Downsample(Block):
    """ Block with additional defaults for downsampling layers.

    Parameters
    ----------
    factor : int
        The downsampling factor to apply.
    """
    def __init__(self, inputs=None, layout='p', factor=2, **kwargs):
        if kwargs.get('base_block') is None:
            if set('pv').intersection(layout):
                kwargs = {
                    'pool_size': factor,
                    'pool_stride': factor,
                    **kwargs
                }
            elif set('x').intersection(layout):
                kwargs = {
                    'downscale_factor': factor,
                    **kwargs
                }
            elif set('cCvV').intersection(layout):
                kwargs = {
                    'kernel_size': factor,
                    'stride': factor,
                    'channels': 'same',
                    **kwargs
                }
        super().__init__(inputs=inputs, layout=layout, **kwargs)
