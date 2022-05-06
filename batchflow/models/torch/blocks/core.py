""" !!. """
from torch import nn

from ..repr_mixin import ModuleDictReprMixin
from ..layers import MultiLayer
from ..utils import get_shape
from .... import Config



class Block(ModuleDictReprMixin, nn.ModuleDict):
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

        layer = Block({layout='cnap', channels='same*2'}, inputs=inputs, n_repeats=5)

    Repeat the whole construction two times::

        repeated = splitted * 2
    """
    VERBOSITY_THRESHOLD = 3

    def __init__(self, *args, inputs=None, base_block=MultiLayer, n_repeats=1, **kwargs):
        super().__init__()

        self.input_shape, self.device = get_shape(inputs), inputs.device
        self.base_block, self.n_repeats = base_block, n_repeats
        self.args, self.kwargs = args, kwargs

        self.shapes = {}
        self.initialize(inputs, base_block, n_repeats, *args, **kwargs)


    def initialize(self, inputs, base_block, n_repeats, *args, **kwargs):
        """ !!. """
        for r in range(n_repeats):
            if args:
                for i, item in enumerate(args):
                    # Make block
                    if isinstance(item, dict):
                        block = item.pop('base_block', None) or base_block
                        block_args = {'inputs': inputs, **dict(Config(kwargs) + Config(item))}
                        layer = block(**block_args)
                    elif isinstance(item, nn.Module):
                        layer = item
                    else:
                        raise ValueError(f'Positional arguments of Block must be either dicts or nn.Modules, \
                                           got {type(item)} instead!')

                    # Apply block and store shapes
                    input_shapes = get_shape(inputs)
                    inputs = layer(inputs)
                    output_shapes = get_shape(inputs)
                    layer_name = f'r{r}-i{i}'

                    self[layer_name] = layer
                    self.shapes[layer_name] = (input_shapes, output_shapes)

            else:
                # Make block
                layer = base_block(inputs=inputs, **kwargs)

                # Apply block and store shapes
                input_shapes = get_shape(inputs)
                inputs = layer(inputs)
                output_shapes = get_shape(inputs)
                layer_name = f'r{r}'

                self[layer_name] = layer
                self.shapes[layer_name] = (input_shapes, output_shapes)


    def forward(self, x):
        for layer in self.values():
            x = layer(x)
        return x



class Upsample(Block):
    """ !!. """
    def __init__(self, inputs=None, layout='b', factor=2, shape=None, **kwargs):
        if 't' in layout or 'T' in layout:
            kwargs = {
                'kernel_size': factor,
                'stride': factor,
                'channels': 'same',
                **kwargs
            }
        if 'b' in layout:
            kwargs = {
                'scale_factor': factor,
                **kwargs
            }
        super().__init__(inputs=inputs, layout=layout, shape=shape, **kwargs)


class Downsample(Block):
    """ !!. """
    def __init__(self, inputs=None, layout='p', factor=2, **kwargs):
        if 'p' in layout or 'v' in layout:
            kwargs = {
                'pool_size': factor,
                'pool_stride': factor,
                **kwargs
            }
        elif 'c' in layout or 'C' in layout or 'w' in layout or 'W' in layout:
            kwargs = {
                'kernel_size': factor,
                'stride': factor,
                'channels': 'same',
                **kwargs
            }
        super().__init__(inputs=inputs, layout=layout, **kwargs)
