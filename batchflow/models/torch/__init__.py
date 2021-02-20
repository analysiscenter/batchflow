""" Various Torch models and blocks.

.. note::
    This module requries PyTorch package.
"""
from .base import TorchModel
from .encoder_decoder import Encoder, Decoder, EncoderDecoder, AutoEncoder, VariationalAutoEncoder
from .vgg import VGG, VGG7, VGG16, VGG19
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                    ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, \
                    SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152, \
                    SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152
from .resnest import ResNeSt, ResNeSt18, ResNeSt34
from .densenet import DenseNet, DenseNetS, DenseNet121, DenseNet169, DenseNet201, DenseNet264
from .unet import UNet, ResUNet, DenseUNet
from .vnet import VNet
from .densenet import SegmentationDenseNet, DenseNetFC56, DenseNetFC67, DenseNetFC103
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, \
                          EfficientNetB5, EfficientNetB6, EfficientNetB7
from .utils import get_shape, get_num_channels, get_num_dims, calc_padding, unpack_fn_from_config, safe_eval
from .blocks import DefaultBlock, XceptionBlock, VGGBlock, ResBlock, ResNeStBlock, DenseBlock, MBConvBlock, InvResBlock
