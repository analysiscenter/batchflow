""" !!. """
import inspect
from torch import nn

from ..blocks import Block
from ..repr_mixin import LayerReprMixin
from ..utils import get_shape


class DefaultModule(LayerReprMixin, nn.Module):
    """ !!. """
    VERBOSITY_THRESHOLD = 3

    def __init__(self, inputs=None, input_list=True, output_list=False, input_index=-1, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.input_list = input_list
        self.input_index = input_index
        self.output_list = output_list

        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs
        self.block = Block(inputs=inputs, **kwargs)

    def forward(self, inputs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        if inputs_is_list and not self.input_list:
            raise TypeError('Input type is list with `input_list=False`!')
        tensor = inputs[self.input_index] if inputs_is_list else inputs

        # Apply layer
        output = self.block(tensor)

        # Prepare output type: sequence or individual tensor
        if self.output_list:
            if inputs_is_list:
                output = inputs + [output]
            else:
                output = [output]
        return output


class WrapperModule(DefaultModule):
    """ !!. """
    VERBOSITY_THRESHOLD = 4

    def initialize(self, inputs, **kwargs):
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs
        module_constructor = kwargs['module']

        if isinstance(module_constructor, nn.Module):
            module = module_constructor
        else:
            kwargs = {**kwargs, **kwargs.get('module_kwargs', {})}
            if 'inputs' in inspect.getfullargspec(module_constructor.__init__)[0]:
                kwargs['inputs'] = inputs
            module = module_constructor(**kwargs)

        self.block = module
