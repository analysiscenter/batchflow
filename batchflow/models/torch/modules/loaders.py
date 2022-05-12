""" !!. """
from torch import nn

try:
    import transformers
except ImportError:
    transformers = None

from ..utils import get_num_channels



class HuggingFaceLoader(nn.Module):
    """ !!. """
    VERBOSITY_THRESHOLD = 4

    def __init__(self, inputs=None, path=None, task='ImageClassification', pretrained=False, attribute_name='auto',
                 output_type='list', input_index=-1, **kwargs):
        if transformers is None:
            raise ImportError('Install the HuggingFace library! `pip install transformers`')
        super().__init__()

        self.path = path
        self.output_type = output_type
        self.input_index = input_index
        inputs = inputs[self.input_index] if isinstance(inputs, list) else inputs

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
        try:
            model.layernorm = nn.Identity()
            model.layer_norm = nn.Identity()
        except AttributeError:
            pass

        self.model = model.to(inputs.device)

    @staticmethod
    def load_config(path):
        return transformers.AutoConfig.from_pretrained(path)

    def forward(self, inputs):
        # Parse inputs type: list or individual tensor
        inputs_is_list = isinstance(inputs, list)
        tensor = inputs[self.input_index] if inputs_is_list else inputs

        # Apply loaded model
        outputs = self.model.forward(tensor, output_hidden_states=True).hidden_states
        outputs = list(outputs)

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
                return f'HuggingFaceLoader({self.extra_repr()})'
        return super().__repr__()
