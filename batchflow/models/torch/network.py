""" !!. """
from torch import nn

from .repr_mixin import ModuleDictReprMixin
from .modules import DefaultModule, WrapperModule, EncoderModule, DecoderModule, MLPDecoderModule, HuggingFaceLoader
from .utils import get_shape, get_device, make_initialization_inputs



class Network(ModuleDictReprMixin, nn.ModuleDict):
    """ !!. """
    VERBOSITY_THRESHOLD = 1

    MODULE_TO_TYPE = {
        DefaultModule: ['default'],
        WrapperModule: ['wrapper'],
        EncoderModule: ['encoder'],
        DecoderModule: ['decoder'],
        MLPDecoderModule: ['mlp-decoder', 'mlpdecoder'],
        HuggingFaceLoader: ['hugging-face', 'huggingface', 'hf'],
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
        """ !!. """
        for name in config['order']:
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

        self.config['order'] = config['order']

    def make_module(self, inputs, module_params):
        """ !!. """
        if module_params:
            module_type = module_params.pop('type', 'default')

            if module_type in self.TYPE_TO_MODULE:
                module = self.TYPE_TO_MODULE[module_type](inputs=inputs, **module_params)
            else:
                raise ValueError(f'Unknown type of module "{module_type}"!')
        else:
            module = None

        return module

    def forward(self, inputs):
        for module in self.values():
            inputs = module(inputs)
        return inputs
