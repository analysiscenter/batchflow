""" Contains tensorflow models and functions.

.. note::
    This module requires TensorFlow package.
 """
import sys


from .base import TFModel
from .utils import get_shape, get_num_dims, get_channels_axis, get_num_channels, \
                   get_batch_size, get_spatial_dim, get_spatial_shape
from .vgg import VGG, VGG16, VGG19, VGG7
from .linknet import LinkNet
from .unet import UNet, UNetPP
from .vnet import VNet
from .fcn import FCN, FCN32, FCN16, FCN8
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, \
                    ResNeXt18, ResNeXt34, ResNeXt50, ResNeXt101, ResNeXt152, \
                    SEResNet18, SEResNet34, SEResNet50, SEResNet101, SEResNet152, \
                    SEResNeXt18, SEResNeXt34, SEResNeXt50, SEResNeXt101, SEResNeXt152
from .inception_v1 import Inception_v1
from .inception_v3 import Inception_v3
from .inception_v4 import Inception_v4
from .inception_resnet_v2 import InceptionResNet_v2
from .squeezenet import SqueezeNet
from .mobilenet import MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small
from .densenet import DenseNet, DenseNet121, DenseNet169, DenseNet201, DenseNet264
from .faster_rcnn import FasterRCNN
from .resattention import ResNetAttention, ResNetAttention56, ResNetAttention92
from .densenet_fc import DenseNetFC, DenseNetFC56, DenseNetFC67, DenseNetFC103
from .refinenet import RefineNet
from .gcn import GlobalConvolutionNetwork as GCN
from .encoder_decoder import EncoderDecoder, AutoEncoder, VariationalAutoEncoder
from .pyramidnet import PyramidNet, PyramidNet18, PyramidNet34, PyramidNet50, PyramidNet101, PyramidNet152
from .tf_sampler import TfSampler
from .xception import Xception, XceptionS, Xception41, Xception64
from .deeplab import DeepLab, DeepLabXS, DeepLabX8, DeepLabX16
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from .pspnet import PSPNet, PSPNet18, PSPNet34, PSPNet50
