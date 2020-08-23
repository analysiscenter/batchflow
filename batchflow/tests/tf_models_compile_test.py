""" Test that all TFmodels can be constructed. Some of the bigger models are omitted due to slow initialization. """
# pylint: disable=import-error, no-name-in-module, unused-import
# pylint: disable=redefined-outer-name
import pytest

from batchflow.models.tf import LinkNet, UNet, VNet, \
                                FCN8, FCN16, FCN32, \
                                DenseNetFC56, DenseNetFC67, DenseNetFC103, \
                                RefineNet, GCN, DeepLabXS, DeepLabX8, DeepLabX16, \
                                VGG16, VGG19, VGG7, \
                                ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                                ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, \
                                SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152, \
                                SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152, \
                                Inception_v1, InceptionResNet_v2, Inception_v3, Inception_v4, \
                                SqueezeNet, MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small, \
                                DenseNet121, DenseNet169, DenseNet201, DenseNet264, \
                                ResNetAttention56, ResNetAttention92, \
                                PyramidNet18, PyramidNet34, PyramidNet50, PyramidNet101, \
                                XceptionS, Xception41, Xception64, \
                                EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
                                EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, \
                                PSPNet18, PSPNet34, PSPNet50



MODELS_CLF = [
    VGG7, VGG16, VGG19,
    ResNet18, ResNet34, # ResNet50, ResNet101, ResNet152,
    ResNeXt18, ResNeXt34, # ResNeXt50, ResNeXt101, ResNeXt152,
    SEResNet18, SEResNet34, # SEResNet50, SEResNet101, SEResNet152,
    SEResNeXt18, SEResNeXt34, # SEResNeXt50, SEResNeXt101, SEResNeXt152,
    Inception_v1,
    # InceptionResNet_v2, Inception_v3, Inception_v4, # need bigger spatial size of inputs (256)
    SqueezeNet,
    MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small,
    DenseNet121, # DenseNet169, DenseNet201, DenseNet264,
    # ResNetAttention56, # ResNetAttention92, # not working
    # PyramidNet18, PyramidNet34, # PyramidNet50, PyramidNet101, # not working
    XceptionS, Xception41, # Xception64,
    EfficientNetB0, EfficientNetB1, # EfficientNetB2, EfficientNetB3, \
    # EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
]

MODELS_SEG = [
    LinkNet,
    UNet,
    VNet,
    FCN8, FCN16, FCN32,
    DenseNetFC56, # DenseNetFC67, DenseNetFC103,
    RefineNet,
    GCN,
    DeepLabXS, DeepLabX8, # DeepLabX16,
    PSPNet18, # PSPNet34, PSPNet50
]



@pytest.fixture()
def base_config_clf():
    """ Fixture to hold default configuration for classification. """
    config = {'inputs/images/shape': (16, 16, 1),
              'inputs/labels/classes': 10,
              'initial_block/inputs': 'images',
              'loss': 'ce'}
    return config


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
