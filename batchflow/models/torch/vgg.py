""" Simonyan K., Zisserman A. "`Very Deep Convolutional Networks for Large-Scale Image Recognition
<https://arxiv.org/abs/1409.1556>`_"
"""
from .base import TorchModel
from .blocks import VGGBlock



class VGG(TorchModel):
    """ Base VGG architecture. """
    @classmethod
    def default_config(cls):
        """ Define model's defaults: general architecture. """
        config = super().default_config()
        config.update({
            'body': {
                'type': 'encoder',
                'output_type': 'tensor',
                'order': ['block', 'downsampling'],
                'blocks': {
                    'base_block': VGGBlock,
                }
            },
            'head': {
                'layout': 'Vdf',
                'dropout_rate': 0.4,
            },

            'common/conv/bias' : False,
            'loss': 'ce',
        })
        return config


class VGG7(VGG):
    """ VGG7 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body': {
                'num_stages': 3,
                'blocks': {
                    'channels': [64, 128, 256],
                    'depth3': 2,
                    'depth1': [0, 0, 1],
                }
            }
        })
        return config

class VGG16(VGG):
    """ VGG16 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body': {
                'num_stages': 5,
                'blocks': {
                    'channels': [64, 128, 256, 512, 512],
                    'depth3': [2, 2, 3, 3, 3],
                    'depth1': 0,
                }
            }
        })
        return config

class VGG19(VGG):
    """ VGG19 network. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body': {
                'num_stages': 5,
                'blocks': {
                    'channels': [64, 128, 256, 512, 512],
                    'depth3': [2, 2, 4, 4, 4],
                    'depth1': 0,
                }
            }
        })
        return config
