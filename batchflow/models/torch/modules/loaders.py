""" Modules to wrap the process of using other libraries. """
from torch import nn

try:
    import transformers
except ImportError:
    transformers = None

try:
    import timm
except ImportError:
    timm = None

from ..utils import get_num_channels
from ....config import Config


class Loader(nn.Module):
    """ Base class for loaders: modules that import ready-to-use networks from other libraries.

    This module outputs a list of hidden states, independent of the input data (one tensor or list of tensors).
    If the input data is a list, then hidden states are appended to it and then returned.
    """
    DEFAULTS = {}

    VERBOSITY_THRESHOLD = 4

    def __init__(self, inputs=None, input_type='list', output_type='list', input_index=-1, **kwargs):
        super().__init__()
        self.input_type = input_type
        self.input_index = input_index
        self.output_type = output_type

        inputs_is_list = isinstance(inputs, list)
        if inputs_is_list and self.input_type != 'list':
            raise TypeError(f'Input type is list with `input_type={self.input_type}`!')
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs

        kwargs = Config(self.DEFAULTS) + kwargs

        self.initialize(inputs, **kwargs)

    def initialize(self, inputs, **kwargs):
        raise NotImplementedError

    def imported_forward(self, inputs):
        raise NotImplementedError

    def forward(self, inputs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        tensor = inputs[self.input_index] if inputs_is_list else inputs

        # Apply loaded model
        outputs = self.imported_forward(tensor)

        # Prepare output type: sequence or individual tensor
        if self.output_type == 'list':
            if inputs_is_list:
                output = inputs + outputs
            else:
                output = outputs
        else:
            output = outputs[-1]
        return output


    def extra_repr(self):
        return f'path="{self.path}"'

    def __repr__(self):
        if hasattr(self, 'verbosity'):
            if self.verbosity < 3:
                return ''
            if self.verbosity < self.VERBOSITY_THRESHOLD:
                return f'{self.__class__.__name__}({self.extra_repr()})'
        return super().__repr__()




class HuggingFaceLoader(Loader):
    """ Module for importing ready-to-use models from the `HuggingFace` library.
    "`Hugging Face github page <https://github.com/huggingface/transformers>`_"

    In order to make a valid module, `path` should be provided as one of the names from model hub:
    "`Hugging Face models page <https://huggingface.co/models>`_".

    `kwargs` dictionary is used to update model creation config with new parameters:
    works only if model is not loaded from pretrained weights.

    Parameters
    ----------
    path : str
        A valid name for a model from Hugging Face models hub.
    task : {'ImageClassification', 'SemanticSegmentation'}
        Model type.
    pretrained : bool
        Whether to use pretrained weights.
    attribute_name : str or None
        Attribute to extract from loaded network and use as a module. If None, no attribute is extracted.
        If string, can contain dots for nested getattr, e.g. 'encoder.encoder.stage1'.
    remove_layernorm : bool
        Whether to remove the final layer (named `layernorm`).

    Examples
    --------
    An example of config to use for SegFormer creation::
        'encoder': {
            'type': 'hugging-face',
            'path': 'nvidia/segformer-b0-finetuned-ade-512-512',
            'num_encoder_blocks': 4,                              # ~num_stages
            'strides': [2, 2, 2, 2],
            'hidden_sizes': [48, 64, 128, 128],                   # ~num_channels
            'num_attention_heads': [1, 4, 8, 4],                  # should be divisor of num_channels
        }

    An example of config to use for ConvNext creation::
        'encoder': {
            'type': 'hugging-face',
            'path': 'facebook/convnext-tiny-224',
            'num_stages': 4,
            'depths': [2, 2, 4, 2],                               # number of blocks in corresponding stage
            'hidden_sizes': [48, 64, 96, 128],                    # ~num_channels
        },
    """
    VERBOSITY_THRESHOLD = 4

    def initialize(self, inputs=None, path=None, task='ImageClassification', pretrained=False,
                   attribute_name='auto', remove_layernorm=True, **kwargs):
        """ Import a module and extract required attribute from it. """
        if transformers is None:
            raise ImportError('Install the HuggingFace library! `pip install transformers`')
        self.path = path

        # Make model: either download with fixed structure and weights, or re-initialize
        model_class_name = f'AutoModelFor{task}'
        model_class = getattr(transformers, model_class_name)
        if pretrained:
            model = model_class.from_pretrained(path)
        else:
            config = self.load_config(path)
            for key, value in kwargs.items():
                setattr(config, key, value)

            in_channels = get_num_channels(inputs)
            config.id2label = [None] * in_channels
            config.label2id = [None] * in_channels
            config.num_channels = in_channels

            model = model_class.from_config(config)
            self.config = config

        # Extract part of the model
        if attribute_name == 'auto':
            model = list(model.named_children())[0][1]
        elif isinstance(attribute_name, str):
            attribute_list = attribute_name.split('.')
            for attr in attribute_list:
                model = getattr(model, attr)

        # Remove some of the modules
        if remove_layernorm:
            try:
                model.layernorm = nn.Identity()
                model.layer_norm = nn.Identity()
            except AttributeError:
                pass

        self.model = model.to(inputs.device)

    def imported_forward(self, inputs):
        """ Correct forward for imported module. """
        outputs = self.model.forward(inputs, output_hidden_states=True).hidden_states
        outputs = list(outputs)
        return outputs

    @staticmethod
    def load_config(path):
        return transformers.AutoConfig.from_pretrained(path)



class TIMMLoader(Loader):
    """ Module for importing ready-to-use models from the `PytorchImageModels` library.
    "`PyTorch Image Models github page <https://github.com/rwightman/pytorch-image-models>`_"

    In order to make a valid module, `path` should be provided as one of the names from model hub.

    `kwargs` dictionary is used to update model creation config with new parameters:
    works only if model is not loaded from pretrained weights.

    Parameters
    ----------
    path : str
        A valid name for a model from Hugging Face models hub.
    pretrained : bool
        Whether to use pretrained weights.
    features_only : bool
        Used at model initialization and allows for outputting all hidden states.

    Examples
    --------
    An example of config to use for ResNet34 creation::
        'body': {
            'type': 'timm',
            'output_type': 'tensor',
            'path': 'resnet34d',
            'pretrained': True,
        }
    """
    VERBOSITY_THRESHOLD = 4

    def initialize(self, inputs=None, path=None, pretrained=False, features_only=True, **kwargs):
        """ Import a module. """
        if timm is None:
            raise ImportError('Install the PytorchImageModels library! `pip install timm`')
        self.path = path

        # Make model: rely on `timm` for pretty much everything
        model = timm.create_model(path, pretrained=pretrained, features_only=features_only, **kwargs)
        self.model = model.to(inputs.device)

    def imported_forward(self, inputs):
        return list(self.model.forward(inputs))

    @staticmethod
    def list_models(filter='', module='', pretrained=False, exclude_filters='', name_matches_cfg=False):
        return timm.list_models(filter=filter, module=module, pretrained=pretrained,
                                exclude_filters=exclude_filters, name_matches_cfg=name_matches_cfg)
