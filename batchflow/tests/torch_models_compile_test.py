""" Test that all Torch can be constructed """
# pylint: disable=import-error, no-name-in-module, redefined-outer-name, unused-import
import pytest

from batchflow.models.torch import VGG7, VGG16, VGG19
from batchflow.models.torch import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                                   ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, \
                                   SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152, \
                                   SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152
from batchflow.models.torch import DenseNet121, DenseNet169, DenseNet201, DenseNet264

from batchflow.models.torch import UNet, ResUNet, DenseUNet



MODELS_CLF = [
    VGG7, VGG16, VGG19,
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152,
    SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152,
    SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152,
    DenseNet121, # DenseNet169, DenseNet201, DenseNet264,
]


MODELS_SEG = [
    UNet, ResUNet, DenseUNet,
]



@pytest.fixture()
def base_config_clf():
    """ Fixture to hold default configuration for classification. """
    config = {
        'classes': 10,
        'loss': 'ce',
        'device': 'cpu',
    }
    return config


@pytest.fixture()
def base_config_segment():
    """ Fixture to hold default configuration for segmentation. """
    config = {
        'inputs_shapes': (1, 64, 64),
        'classes': 10,
        'loss': 'ce',
        'device': 'cpu',
    }
    return config



@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_CLF)
def test_clf(base_config_clf, model):
    """ Test models for classification """
    _ = model(base_config_clf)


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_SEG)
def test_seg(base_config_segment, model):
    """ Test models for segmentation """
    _ = model(base_config_segment)
