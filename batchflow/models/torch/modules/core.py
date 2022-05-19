""" Defaults module. """
import inspect
from torch import nn

from ..blocks import Block
from ..repr_mixin import LayerReprMixin


class DefaultModule(LayerReprMixin, nn.Module):
    """ Module for default model parts.
    Relies on :class:`~.torch.layers.MultiLayer` for actual operations.

    Implements additional logic of working with inputs and outputs:
        - if `input_type` is `tensor` and `output_type` is `tensor`,
        then this module expects one tensor and outputs one tensor.
        - if `input_type` is   `list` and `output_type` is `tensor`,
        then this module slices the list with `input_index` and outputs one tensor.
        - if `input_type` is `tensor` and `output_type` is `list`,
        then this module expects one tensor and wraps the output in list.
        - if `input_type` is `list` and `output_type` is `list`,
        then this module slices the list wth `input_index` and appends the output to the same list, which is returned.
    """
    VERBOSITY_THRESHOLD = 3

    def __init__(self, inputs=None, input_type='tensor', output_type='tensor', input_index=-1, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.input_type = input_type
        self.input_index = input_index
        self.output_type = output_type

        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        if inputs_is_list and self.input_type != 'list':
            raise TypeError(f'Input type is list with `input_type={self.input_type}`!')
        inputs = inputs[self.input_index] if inputs_is_list else inputs

        self.block = Block(inputs=inputs, **kwargs)

    def forward(self, inputs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        tensor = inputs[self.input_index] if inputs_is_list else inputs

        # Apply layer
        output = self.block(tensor)

        # Prepare output type: sequence or individual tensor
        if self.output_type == 'list':
            if inputs_is_list:
                output = inputs + [output]
            else:
                output = [output]
        return output


class WrapperModule(DefaultModule):
    """ Module for wrapping external nn.Modules.

    Allows to use `module` key for initialization:
        - if the value is a nn.Module, then it is used directly
        - otherwise, `module` is expected to be a module constructor, which is initialized with the rest of the kwargs.
    """
    VERBOSITY_THRESHOLD = 4

    def initialize(self, inputs, **kwargs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        if inputs_is_list and self.input_type != 'list':
            raise TypeError(f'Input type is list with `input_type={self.input_type}`!')
        inputs = inputs[self.input_index] if inputs_is_list else inputs

        # Parse module
        module_constructor = kwargs['module']

        if isinstance(module_constructor, nn.Module):
            module = module_constructor
        else:
            kwargs = {**kwargs, **kwargs.get('module_kwargs', {})}
            if 'inputs' in inspect.getfullargspec(module_constructor.__init__)[0]:
                kwargs['inputs'] = inputs
            module = module_constructor(**kwargs)

        self.block = module
