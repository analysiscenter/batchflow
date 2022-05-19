""" Network class: a model, made from multiple parts (modules). """
from torch import nn

from .repr_mixin import ModuleDictReprMixin
from .modules import (DefaultModule, WrapperModule,
                      EncoderModule, DecoderModule, MLPDecoderModule,
                      HuggingFaceLoader, TIMMLoader)
from .utils import get_shape, get_device, make_initialization_inputs, make_shallow_dict



class Network(ModuleDictReprMixin, nn.ModuleDict):
    """ Class to chain multiple modules together.
    The main functionality is to select required module, based on type, and chain multiple of them together.

    !!.
    """
    VERBOSITY_THRESHOLD = 1

    MODULE_TO_TYPE = {
        DefaultModule: ['default'],
        WrapperModule: ['wrapper'],
        EncoderModule: ['encoder'],
        DecoderModule: ['decoder'],
        MLPDecoderModule: ['mlp-decoder', 'mlpdecoder'],
        HuggingFaceLoader: ['hugging-face', 'huggingface', 'hf'],
        TIMMLoader: ['timm'],
    }
    TYPE_TO_MODULE = {alias: module for module, aliases in MODULE_TO_TYPE.items() for alias in aliases}


    def __init__(self, inputs=None, config=None, device=None):
        super().__init__()

        # Parse parameters
        inputs = make_initialization_inputs(inputs, device=device)
        self.device = get_device(inputs)

        self.shapes = {}
        self.config = {}
        self.initialize(inputs, config)


    def initialize(self, inputs, config):
        """ Make multiple modules. Mark some of them as frozen (not trainable). """
        order = self.config['order'] = config['order']
        for name in order:
            # Make module based on its type, parameters, and current `inputs` tensor
            module_params = {**config.get('common', {}), **config[name]}
            module = self.make_module(inputs, module_params)

            if module is not None:
                module = module.to(self.device)

                # Update `inputs` tensor by applying the module
                input_shapes = get_shape(inputs)
                inputs = module(inputs)
                output_shapes = get_shape(inputs)

                # Store shapes for later introspection
                self[name] = module
                self.config[name] = module_params
                self.shapes[name] = (input_shapes, output_shapes)

        # Freeze some of the model parts
        trainable = self.config['trainable'] = config.get('trainable') or order
        frozen = self.config['frozen'] = set(order) - set(trainable)
        for name in frozen:
            for parameter in self[name].parameters():
                parameter.requires_grad = False

        # Set `batchflow_attribute_name` attribute for every children, so that every submodule knows its full name
        shallow_dict = make_shallow_dict(self)
        for attribute_name, module_instance in shallow_dict.items():
            setattr(module_instance, 'batchflow_attribute_name', attribute_name)

    def make_module(self, inputs, module_params):
        """ Parse the type of one module and make it. """
        if module_params:
            module_type = module_params.pop('type', 'default')

            if module_type in self.TYPE_TO_MODULE:
                module = self.TYPE_TO_MODULE[module_type](inputs=inputs, **module_params)
            else:
                raise ValueError(f'Unknown type of module "{module_type}"!')

            # Set `batchflow_module_type` attribute for every children, so that every submodule its place
            module.apply(lambda submodule: setattr(submodule, 'batchflow_module_type', module_type))
        else:
            module = None

        return module

    def forward(self, inputs):
        for module in self.values():
            inputs = module(inputs)
        return inputs
