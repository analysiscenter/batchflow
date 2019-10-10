""" Test that all TFmodels can be constructed """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

from batchflow.models.tf import VGG16, VGG19, VGG7, \
    LinkNet, \
    UNet, \
    VNet, \
    FCN32, FCN16, FCN8, \
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
    ResNeXt18, ResNeXt34, \
    ResNeXt50, ResNeXt101, ResNeXt152, \
    Inception_v1, \
    Inception_v3, \
    Inception_v4, \
    InceptionResNet_v2, \
    SqueezeNet, \
    MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small, \
    DenseNet121, \
    DenseNet169, DenseNet201, DenseNet264, \
    ResNetAttention56, ResNetAttention92, \
    DenseNetFC56, DenseNetFC67, DenseNetFC103, \
    RefineNet, \
    GCN, \
    PyramidNet18, PyramidNet34, PyramidNet50, PyramidNet101, PyramidNet152, \
    XceptionS, Xception41, Xception64, \
    DeepLabXS, DeepLabX8, DeepLabX16, \
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7


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
    ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, # too heavy ?
    Inception_v1, Inception_v3, Inception_v4, InceptionResNet_v2,  # heavy fail
    SqueezeNet,
    MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small,
    DenseNet121, DenseNet169, DenseNet201, DenseNet264, # too heavy ?
    ResNetAttention56, ResNetAttention92,  # fail
    PyramidNet18,  # fail
    PyramidNet34,
    PyramidNet50,  # fail
    PyramidNet101,
    PyramidNet152,
    XceptionS, Xception41, Xception64,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
]


@pytest.fixture()
def base_config_segment():
    """ Fixture to hold default configuration for segmentation. """
    config = {
        'inputs': {'images': {'shape': (16, 16, 1)},
                   'masks': {'name': 'targets', 'shape': (16, 16, 1)}},
        'initial_block': {'inputs': 'images'},
        'loss': 'mse'
    }
    return config


@pytest.fixture()
def base_config_clf():
    """ Fixture to hold default configuration for classification. """
    config = {'inputs/images/shape': (16, 16, 1),
              'inputs/labels/classes': 10,
              'initial_block/inputs': 'images',
              'loss': 'ce'}
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
