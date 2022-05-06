""" !!. """
from torch import nn


from ..repr_mixin import ModuleDictReprMixin
from ..blocks import Block, Upsample, Downsample, Combine
from ..utils import get_shape
from ...utils import unpack_args
from ....config import Config



class EncoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ Encoder: create compressed representation of an input by reducing its spatial dimensions. """
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'num_stages': None,
        'order': ['skip', 'block', 'downsampling'],
        'blocks': {'layout': 'cna', 'channels': 'same', 'kernel_size': 3},
        'downsample': {'layout': 'p', 'pool_size': 2, 'pool_stride': 2}
    }

    def __init__(self, inputs=None, output_hidden_states=True, input_index=-1, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs
        self.output_hidden_states = output_hidden_states
        self.input_index = input_index

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ !!. """
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs

        # Parse parameters
        num_stages = kwargs.pop('num_stages')
        order = ''.join([item[0] for item in kwargs.pop('order')])

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

                elif letter in {'d', 'p'}:
                    args = {**kwargs, **downsample_params, **unpack_args(downsample_params, i, num_stages)}
                    block = Downsample(inputs=inputs, **args)
                    inputs = block(inputs)
                    block_name = f'downsample-{i}'

                elif letter in {'s'}:
                    block = nn.Identity()
                    block_name = f'skip-{i}'
                else:
                    raise ValueError(f'Unknown letter "{letter}" in order, use one of "b", "d", "p", "s"')

                output_shapes = get_shape(inputs)

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
                outputs.append(tensor)
        outputs.append(tensor)

        # Prepare output type: sequence or individual tensor
        if self.output_hidden_states:
            if inputs_is_list:
                output = inputs + outputs
            else:
                output = outputs
        else:
            output = outputs[-1]
        return output


class DecoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ Decoder: increasing spatial dimensions. """
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'num_stages': None,
        'order': ['upsampling', 'block', 'combine'],
        'skip': True,
        'blocks': {'layout': 'cna', 'channels': 'same', 'kernel_size': 3},
        'upsample': {'layout': 'b', 'factor': 2},
        'combine': {'op': 'concat', 'leading_index': 1}
    }

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ !!. """
        # Parse inputs
        inputs = inputs if isinstance(inputs, list) else [inputs]
        tensor = inputs[-1]

        # Parse parameters
        num_stages = kwargs.pop('num_stages') or len(inputs) - 2
        order = ''.join([item[0] for item in kwargs.pop('order')])
        self.skip = kwargs.pop('skip')

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
                    if self.skip:
                        if i < len(inputs) - 2:
                            args = {**kwargs, **combine_params, **unpack_args(combine_params, i, num_stages)}
                            input_shapes = get_shape([tensor, inputs[-i - 3]])
                            block = Combine(inputs=[tensor, inputs[-i - 3]], **args)
                            tensor = block([tensor, inputs[-i - 3]])
                            block_name = f'combine-{i}'
                    else:
                        continue
                else:
                    raise ValueError('Unknown letter "{letter}" in order, use one of ("b", "u", "c")')
                output_shapes = get_shape(tensor)

                self[block_name] = block
                self.shapes[block_name] = (input_shapes, output_shapes)


    def forward(self, inputs):
        inputs = inputs if isinstance(inputs, list) else [inputs]
        tensor = inputs[-1]
        i = 0

        for block_name, block in self.items():
            letter = block_name[0]

            if letter in ['b', 'u']:
                tensor = block(tensor)
            elif letter in ['c'] and self.skip and (i < len(inputs) - 2):
                tensor = block([tensor, inputs[-i - 3]])
                i += 1
        return tensor


class MLPDecoderModule(ModuleDictReprMixin, nn.ModuleDict):
    """ Decoder: increasing spatial dimensions. """
    VERBOSITY_THRESHOLD = 2

    DEFAULTS = {
        'size': None,
        'upsample': {'layout': 'Fb', 'features': 'same'},
        'combine': {'op': 'concat', 'force_resize': False},
        'block': {'layout': 'cnad', 'channels': 'same', 'kernel_size': 1, 'dropout_rate': 0.0}
    }

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        kwargs = Config(self.DEFAULTS) + kwargs
        self.kwargs = kwargs

        self.shapes = {}
        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        """ !!. """
        # Parse inputs
        inputs = inputs if isinstance(inputs, list) else [inputs]

        # Parse parameters
        size = kwargs.pop('size') or inputs[0].shape[2:]
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
