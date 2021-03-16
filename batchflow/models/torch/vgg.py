""" Simonyan K., Zisserman A. "`Very Deep Convolutional Networks for Large-Scale Image Recognition
<https://arxiv.org/abs/1409.1556>`_"
"""
from .encoder_decoder import Encoder
from .blocks import VGGBlock



class VGG(Encoder):
    """ Base VGG architecture. """
    @classmethod
    def default_config(cls):
        """ Define model's defaults: general architecture. """
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
