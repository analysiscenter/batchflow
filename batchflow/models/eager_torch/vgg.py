""" Simonyan K., Zisserman A. "`Very Deep Convolutional Networks for Large-Scale Image Recognition
<https://arxiv.org/abs/1409.1556>`_"
"""
import torch.nn as nn

from .layers import ConvBlock
from .encoder_decoder import Encoder
from .utils import get_num_channels, safe_eval



class VGGBlock(nn.Module):

    def __init__(self, inputs=None, layout='cna', filters=None, depth3=1, depth1=0, **kwargs):
        super().__init__()

        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        layout = layout * (depth3 + depth1)
        kernels = [3]*depth3 + [1]*depth1
        self.layer = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=kernels)

    def forward(self, x):
        return self.layer(x)


class VGG(Encoder):
    """ Base VGG architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False

        config['body/encoder/order'] = ['block', 'downsampling']
        config['body/encoder/blocks'] += dict(base=VGGBlock, layout='cna')
        config['head'] += dict(layout='Vdf', dropout_rate=.2)
        config['loss'] = 'ce'
        return config



class VGG7(VGG):
    """ VGG7 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/num_stages'] = 3
        config['body/encoder/blocks'] += dict(filters=[64, 128, 256],
                                              depth3=2,
                                              depth1=[0, 0, 1])
        return config


class VGG16(VGG):
    """ VGG16 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/num_stages'] = 5
        config['body/encoder/blocks'] += dict(filters=[64, 128, 256, 512, 512],
                                              depth3=[2, 2, 3, 3, 3], depth1=0)
        return config


class VGG19(VGG):
    """ VGG19 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/num_stages'] = 5
        config['body/encoder/blocks'] += dict(filters=[64, 128, 256, 512, 512],
                                              depth3=[2, 2, 4, 4, 4], depth1=0)
        return config
