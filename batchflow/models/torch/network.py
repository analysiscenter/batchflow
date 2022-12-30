""" Network class: a model, made from multiple parts (modules). """
from torch import nn

from .repr_mixin import ModuleDictReprMixin
from .modules import (DefaultModule,
                      EncoderModule, DecoderModule, MLPDecoderModule,
                      HuggingFaceLoader, TIMMLoader)
from .utils import get_shape, get_device, make_initialization_inputs, make_shallow_dict



class Network(ModuleDictReprMixin, nn.ModuleDict):
    """ Class to chain multiple modules together.
    The main functionality is to select required module, based on type, and chain multiple of them together.

    Works by analyzing supplied `config`:
        - `order` key defines the order and number of modules to chain.
        Each element is a string with a module name, for example, `'head'`.

        - each element in `order` is used as a key to get value from `config`.
        A value should be a dictionary with `type` key, that defines the type of module to use:
            - `default`, used also if no `type` is provided.
            Relies on `:class:~.layers.MultiLayer` for making actual operations.
            - `encoder`, `decoder`, `mlp-decoder`.
            Process tensor sequentially with increase/decrease in spatial dimensionality.
            - `timm`, `hugging-face`.
            Import a module from library and use it as tensor processing operation.
        Each module accepts optional `input_type` and `output_type` keys, which can modify
        the behavior to input or output to work with individual tensor or list.

        - optional `trainable` key can be used to freeze some of the network parts.
        If not provided, uses the same names as in `order` and enables all of them to train.

    Parameters
    ----------
    inputs : torch.Tensor, tuple of ints or list of them
        Example of the input tensor(s) to the network.
        Instead of instance of torch.Tensor, one can use its tuple shape (with batch dim included).
    device : str or torch.cuda.Device
        Device to use for tensor definition. Used only if some of the `inputs` are shapes.
    config : dict
        Configuration of network to initialize.
    """
    VERBOSITY_THRESHOLD = 1

    MODULE_TO_TYPE = {
        DefaultModule: ['default', 'wrapper'],
        EncoderModule: ['encoder'],
        DecoderModule: ['decoder'],
        MLPDecoderModule: ['mlp-decoder', 'mlpdecoder'],
        TIMMLoader: ['timm'],
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
        self._last_used_modules = None


    # Initialization of the network: making modules and forward pass
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
            if isinstance(module_type, str):
                module_type = module_type.lower()

            if callable(module_type):
                module = module_type(inputs=inputs, **module_params)
            elif module_type in self.TYPE_TO_MODULE:
                module = self.TYPE_TO_MODULE[module_type](inputs=inputs, **module_params)
            else:
                raise ValueError(f'Unknown type of module "{module_type}"!')

            # Set `batchflow_module_type` attribute for every children, so that every submodule knows its type
            module.apply(lambda submodule: setattr(submodule, 'batchflow_module_type', module_type))
        else:
            module = None

        return module

    def forward(self, inputs):
        used_modules = []

        for name, module in self.items():
            inputs_id = id(inputs)
            inputs = module(inputs)

            if id(inputs) != inputs_id:
                used_modules.append(name)

        self._last_used_modules = used_modules
        return inputs
