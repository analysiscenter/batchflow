""" Encoder and decoders: process tensors in multiple stages with reducing or increasing spatial dimensionality. """
from torch import nn


from ..repr_mixin import ModuleDictReprMixin
from ..blocks import Block, Branch, Upsample, Downsample, Combine, DefaultBlock
from ..utils import get_shape
from ...utils import unpack_args
from ....config import Config



class EncoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ EncoderModule: create compressed representation of an input by reducing its spatial dimensions.

    Consists of multiple stages, parametrized by the `num_stages` key.
    Each stage consists of multiple blocks. Order of blocks is parametrized by `order` key.
    The contents (`base_block`, `layout`, etc) parametrized in corresponding `blocks` and `downsample` keys.

    This module can work with either individual tensor / list of them as the input.
    Usually it expects one tensor. If the input is list, then it is simply sliced with `input_index`.
    Depending on `output_type`:
        - if `output_type` is `tensor`, then only the last activation is returned.
        - if `output_type` is   `list`, then all hidden activations (from each stage) are returned in a list.

    Parameters
    ----------
    num_stages : int
        Number of stages.
    order : str, sequence of str
        Determines order of applying layers.
        If str, then each letter stands for operation:
        'b' for 'block', 'd'/'p' for 'downsampling', 's' for 'skip'.
        If sequence, than the first letter of each item stands for operation:
        For example, `'sbd'` allows to use throw skip connection -> block -> downsampling.
    downsample : dict, optional
        Parameters for downsampling, see :class:`~.blocks.Downsample`.
    skip : dict, optional
        Parameters for additional operations on skip connections, see :class:`~.blocks.Block`.
    blocks : dict, optional
        Parameters for processing blocks, see :class:`~.blocks.Block`.
    """
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'num_stages': None,
        'order': ['skip', 'block', 'downsampling'],
        'skip': {},
        'blocks': {'base_block': DefaultBlock},
        'downsample': {'layout': 'p', 'pool_size': 2, 'pool_stride': 2}
    }

    def __init__(self, inputs=None, output_type='list', input_index=-1, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs
        self.input_index = input_index
        self.output_type = output_type

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ Chain stages and their contents. """
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs

        # Parse parameters
        num_stages = kwargs.pop('num_stages')
        order = ''.join([item[0] for item in kwargs.pop('order')])

        skip_params = kwargs.pop('skip')
        block_params = kwargs.pop('blocks')
        downsample_params = kwargs.pop('downsample')

        for i in range(num_stages):
            for letter in order:

                input_shapes = get_shape(inputs)
                if letter in {'b'}:
                    args = {**kwargs, **block_params, **unpack_args(block_params, i, num_stages)}
                    block = Block(inputs=inputs, **args)
                    inputs = block(inputs)
                    block_name = f'block-{i}'
                    output_shapes = get_shape(inputs)

                elif letter in {'d', 'p'}:
                    args = {**kwargs, **downsample_params, **unpack_args(downsample_params, i, num_stages)}
                    block = Downsample(inputs=inputs, **args)
                    inputs = block(inputs)
                    block_name = f'downsample-{i}'
                    output_shapes = get_shape(inputs)

                elif letter in {'s'}:
                    args = {**kwargs, **skip_params, **unpack_args(skip_params, i, num_stages)}
                    block = Branch(inputs=inputs, **args)
                    skip = block(inputs)
                    block_name = f'skip-{i}'
                    output_shapes = get_shape(skip)
                else:
                    raise ValueError(f'Unknown letter "{letter}" in order, use one of "b", "d", "p", "s"!')

                self[block_name] = block
                self.shapes[block_name] = (input_shapes, output_shapes)

    def forward(self, inputs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        tensor = inputs[self.input_index] if inputs_is_list else inputs

        # Apply encoder
        outputs = []

        for block_name, block in self.items():
            letter = block_name[0]

            if letter in {'b', 'd', 'p'}:
                tensor = block(tensor)
            elif letter in {'s'}:
                skip = block(tensor)
                outputs.append(skip)
        outputs.append(tensor)

        # Prepare output type: sequence or individual tensor
        if self.output_type == 'list':
            if inputs_is_list:
                output = inputs + outputs
            else:
                output = outputs
        else:
            output = outputs[-1]
        return output


class DecoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ Decoder: increasing spatial dimensions.

    General idea is to sequentially increase spatial dimensionality, while concatenating provided skip-connections
    to facilitate information flow.

    Consists of multiple stages, parametrized by the `num_stages` key.
    If not provided, uses length of the inputs as default.
    Each stage consists of multiple blocks. Order of blocks is parametrized by `order` key.
    The contents (`base_block`, `layout`, etc) parametrized in corresponding `blocks`, `upsample` and `combine` keys.

    This module can work with either individual tensor / list of them as the input.
    Usually it expects a list of hidden activations. If the input is list, then it is sliced with `input_index`.
    Outputs one tensor.

    Parameters
    ----------
    num_stages : int
        Number of stages.
    order : str, sequence of str
        Determines order of applying layers.
        If str, then each letter stands for operation:
        'b' for 'block', 'u' for 'upsampling', 'c' for 'combine'
        If sequence, than the first letter of each item stands for operation.
        For example, `'ucb'` allows to use upsampling -> combine -> block.
    upsample : dict, optional
        Parameters for upsampling, see :class:`~.blocks.Upsample`.
    blocks : dict, optional
        Parameters for processing blocks, see :class:`~.blocks.Block`.
    combine : dict, optional
        Parameters for processing blocks, see :class:`~.layers.Combine`.
    """
    # TODO: add meaningful functionality for `output_type=='list'`
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'num_stages': None,
        'order': ['upsampling', 'block', 'combine'],
        'skip': True,
        'blocks': {'base_block': DefaultBlock},
        'upsample': {'layout': 'b', 'factor': 2},
        'combine': {'op': 'concat', 'leading_index': 1}
    }

    def __init__(self, inputs=None, indices=None, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs

        self.indices = indices if indices is not None else list(range(-3, -len(inputs)-1, -1))

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ Chain stages and their contents. """
        # Parse inputs
        inputs = inputs if isinstance(inputs, list) else [inputs]
        tensor = inputs[-1]

        # Parse parameters
        num_stages = kwargs.pop('num_stages') or len(inputs) - 2
        order = ''.join([item[0] for item in kwargs.pop('order')])
        skip = kwargs.pop('skip')

        block_params = kwargs.pop('blocks')
        upsample_params = kwargs.pop('upsample')
        combine_params = kwargs.pop('combine')

        for i in range(num_stages):
            for letter in order:

                input_shapes = get_shape(tensor)
                if letter in {'b'}:
                    args = {**kwargs, **block_params, **unpack_args(block_params, i, num_stages)}
                    block = Block(inputs=tensor, **args)
                    tensor = block(tensor)
                    block_name = f'block-{i}'

                elif letter in {'u'}:
                    args = {**kwargs, **upsample_params, **unpack_args(upsample_params, i, num_stages)}
                    block = Upsample(inputs=tensor, **args)
                    tensor = block(tensor)
                    block_name = f'upsample-{i}'

                elif letter in {'c'}:
                    if skip:
                        skip_index = self.indices[i]
                        if (skip_index is not None and
                            ((0 <= skip_index < len(inputs)) or (-len(inputs) <= skip_index < 0))):
                            args = {**kwargs, **combine_params, **unpack_args(combine_params, i, num_stages)}
                            combine_inputs = [tensor, inputs[skip_index]]
                            input_shapes = get_shape(combine_inputs)
                            block = Combine(inputs=combine_inputs, **args)
                            tensor = block(combine_inputs)
                            block_name = f'combine-{i}'
                        else:
                            continue
                    else:
                        raise ValueError('Using "c" letter with `skip=False`!')
                else:
                    raise ValueError('Unknown letter "{letter}" in order, use one of ("b", "u", "c")!')
                output_shapes = get_shape(tensor)

                self[block_name] = block
                self.shapes[block_name] = (input_shapes, output_shapes)


    def forward(self, inputs):
        inputs = inputs if isinstance(inputs, list) else [inputs]
        tensor = inputs[-1]

        for block_name, block in self.items():
            letter = block_name[0]

            if letter in ['b', 'u']:
                tensor = block(tensor)
            elif letter in ['c']:
                stage_index = int(block_name.split('-')[-1])
                skip_index = self.indices[stage_index]
                tensor = block([tensor, inputs[skip_index]])
        return tensor


class MLPDecoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ Decoder: increasing spatial dimensions.

    General idea is to separately increase spatial dimensionality of all of the provided inputs, then combine them
    into one resulting tensor.

    This module can work with either individual tensor / list of them as the input.
    Usually it expects a list of hidden activations. If the input is list, then it is sliced with `input_index`.
    Outputs one tensor.

    Parameters
    ----------
    size : tuple of ints, optional
        Desired spatial size of resulting tensor. If not provided, uses the biggest of the inputs.
    upsample : dict, optional
        Parameters for upsampling, see :class:`~.blocks.Upsample`.
    block : dict, optional
        Parameters for processing block, see :class:`~.blocks.Block`.
    combine : dict, optional
        Parameters for processing blocks, see :class:`~.layers.Combine`.
    """
    # TODO: add meaningful functionality for `output_type=='list'`
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'size': None,
        'upsample': {'layout': 'cb', 'channels': 'same'},
        'combine': {'op': 'concat', 'force_resize': False},
        'block': {'base_block': DefaultBlock}
    }

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ Upsample every tensor to the same shape, combine, postprocess. """
        # Parse inputs
        inputs = inputs if isinstance(inputs, list) else [inputs]

        # Parse desired shape
        size = kwargs.pop('size') or inputs[0].shape[2:]
        if size is None:
            shapes = get_shape(inputs)
            size = (max([shape[-2] for shape in shapes]),
                    max([shape[-1] for shape in shapes]))
        self.size = size

        upsample_params = kwargs.pop('upsample')
        combine_params = kwargs.pop('combine')
        block_params = kwargs.pop('block')

        # Upsample blocks
        upsampled_tensors = []
        for i, tensor in enumerate(inputs):
            input_shapes = get_shape(tensor)
            args = {'shape': size, 'factor': None,
                    **kwargs, **upsample_params, **unpack_args(upsample_params, i, len(inputs))}
            block = Upsample(inputs=tensor, **args)
            tensor = block(tensor)
            output_shapes = get_shape(tensor)

            upsampled_tensors.append(tensor)

            block_name = f'upsample-{i}'
            self[block_name] = block
            self.shapes[block_name] = (input_shapes, output_shapes)

        # Combine
        input_shapes = get_shape(upsampled_tensors)
        args = {**kwargs, **combine_params}
        combine_block = Combine(inputs=upsampled_tensors, **args)
        combined_tensor = combine_block(upsampled_tensors)
        output_shapes = get_shape(combined_tensor)

        block_name = 'combine'
        self[block_name] = combine_block
        self.shapes[block_name] = (input_shapes, output_shapes)

        # Postprocess
        input_shapes = get_shape(combined_tensor)
        args = {**kwargs, **block_params}
        postprocess_block = Block(inputs=combined_tensor, **args)
        output = postprocess_block(combined_tensor)
        output_shapes = get_shape(output)

        block_name = 'block'
        self[block_name] = postprocess_block
        self.shapes[block_name] = (input_shapes, output_shapes)


    def forward(self, inputs):
        inputs = inputs if isinstance(inputs, list) else [inputs]

        upsampled_tensors = []
        for i, tensor in enumerate(inputs):
            tensor = self[f'upsample-{i}'](tensor)
            upsampled_tensors.append(tensor)

        combined_tensor = self['combine'](upsampled_tensors)
        output = self['block'](combined_tensor)
        return output
