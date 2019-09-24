""" Test for EncoderDecoder model architecture.
First of all, we define possible types of encoders, embeddings and decoders.
Later every combination of encoder, embedding, decoder is combined into one model and we initialize it.
"""
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

# from .base import TFModel
from batchflow.models.tf import VGG16, VGG19, VGG7
from batchflow.models.tf import LinkNet
from batchflow.models.tf import UNet
from batchflow.models.tf import VNet
from batchflow.models.tf import FCN32, FCN16, FCN8
from batchflow.models.tf import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                    ResNeXt18, ResNeXt34 #, ResNeXt50, ResNeXt101, ResNeXt152
# from batchflow.models.tf import Inception_v1
# from batchflow.models.tf import Inception_v3
# from batchflow.models.tf import Inception_v4
# from batchflow.models.tf import InceptionResNet_v2
from batchflow.models.tf import SqueezeNet
from batchflow.models.tf import MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small
from batchflow.models.tf import DenseNet121 #, DenseNet169, DenseNet201, DenseNet264
# from batchflow.models.tf import FasterRCNN
from batchflow.models.tf import ResNetAttention56, ResNetAttention92
from batchflow.models.tf import DenseNetFC56, DenseNetFC67, DenseNetFC103
from batchflow.models.tf import RefineNet
from batchflow.models.tf import GCN
from batchflow.models.tf import PyramidNet18, PyramidNet34, PyramidNet50, PyramidNet101, PyramidNet152
from batchflow.models.tf import XceptionS, Xception41, Xception64
from batchflow.models.tf import DeepLabXS, DeepLabX8, DeepLabX16


MODELS_SEG = [
    LinkNet,
    UNet,
    VNet,  # fails
    FCN32, FCN16, FCN8,
    DenseNetFC56, DenseNetFC67, DenseNetFC103,
    RefineNet, # fails
    GCN,
    DeepLabXS, DeepLabX8, DeepLabX16
]

MODELS_CLF = [
    VGG16, VGG19, VGG7,
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    ResNeXt18, ResNeXt34,  # ResNeXt50, ResNeXt101, ResNeXt152, # too heavy ?
    # Inception_v1, Inception_v3, Inception_v4, InceptionResNet_v2,  # heavy fail
    SqueezeNet,
    MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small,
    DenseNet121,  # DenseNet169, DenseNet201, DenseNet264, # too heavy &
    ResNetAttention56, ResNetAttention92,  # fail
    PyramidNet18,  # fail
    PyramidNet34,
    PyramidNet50,  # fail
    PyramidNet101,
    PyramidNet152,
    XceptionS, Xception41, Xception64
]


@pytest.fixture()
def base_config_segment():
    """ Fixture to hold default configuration. """
    config = {
        'inputs': {'images': {'shape': (16, 16, 1)},
                   'masks': {'name': 'targets', 'shape': (16, 16, 1)}},
        'initial_block': {'inputs': 'images'},
        'loss': 'mse'
    }
    return config


@pytest.fixture()
def base_config_clf():
    """ Fixture to hold default configuration. """
    config = {'inputs/images/shape': (16, 16, 1),
              'inputs/labels/classes': 10,
              'initial_block/inputs': 'images',
              'loss': 'ce'}
    return config


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_SEG)
def test_seg(base_config_segment, model):
    """ Create encoder-decoder architecture from every possible combination
    of encoder, embedding, decoder, listed in global variables defined above.
    """
    _ = model(base_config_segment)


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_CLF)
def test_clf(base_config_clf, model):
    """ Create encoder-decoder architecture from every possible combination
    of encoder, embedding, decoder, listed in global variables defined above.
    """
    _ = model(base_config_clf)
