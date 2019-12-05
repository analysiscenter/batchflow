""" Test that all EagerTorch can be constructed """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest


from batchflow.models.eager_torch import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                                         ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, \
                                         SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152, \
                                         SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152

from batchflow.models.eager_torch import UNet, ResUNet


MODELS_SEG = [
    UNet, ResUNet
]

MODELS_CLF = [
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152,
    SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152,
    SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152,
]


@pytest.fixture()
def base_config_segment():
    """ Fixture to hold default configuration for segmentation. """
    config = {
        'inputs': {'images': {'shape': (1, 16, 16)},
                   'masks': {'name': 'targets', 'shape': (1, 16, 16)}},
        'initial_block': {'inputs': 'images'},
        'loss': 'mse'
    }
    return config


@pytest.fixture()
def base_config_clf():
    """ Fixture to hold default configuration for classification. """
    config = {'inputs/images/shape': (1, 16, 16),
              'inputs/labels/classes': 10,
              'initial_block/inputs': 'images'}
    return config


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_SEG)
def test_seg(base_config_segment, model):
    """ Test models for segmentation """
    _ = model(base_config_segment)


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS_CLF)
def test_clf(base_config_clf, model):
    """ Test models for classification """
    _ = model(base_config_clf)
