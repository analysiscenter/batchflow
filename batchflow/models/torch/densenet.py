"""
Huang G. et al. "`Densely Connected Convolutional Networks
<https://arxiv.org/abs/1608.06993>`_"
Jegou S. et al "`The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation
<https://arxiv.org/abs/1611.09326>`_"
"""
from .base import TorchModel
from .blocks import DenseBlock



class DenseNet(TorchModel):
    """ DenseNet architecture. """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TorchModel.default_config`"""
        config = super().default_config()
        config.update({
            'initial_block' : {
                'layout': 'cnap',
                'channels': 16,
                'kernel_size': 7,
                'stride': 2,
                'pool_size': 3,
                'pool_stride': 2
            },

            'body': {
                'type': 'encoder',
                'output_type': 'tensor',
                'num_stages': 4,
                'order': ['block', 'downsampling'],
                'blocks': {
                    'base_block': DenseBlock,
                    'layout': 'nacd',
                    'num_layers': None,
                    'growth_rate': None,
                    'skip': True,
                    'bottleneck': True,
                    'dropout_rate': 0.2,
                    'channels': None
                },
                'downsample': {
                    'layout': 'nacv',
                    'kernel_size': 1,
                    'stride': 1,
                    'pool_size': 2,
                    'pool_stride': 2,
                    'channels': 'same'
                }
            },
            'head': {
                'layout': 'Vf',
            },

            'common/bias': False,
            'loss': 'ce',
        })

        config['head'] += dict(layout='Vf')
        return config


class DenseNetS(DenseNet):
    """ Small version of DenseNet architecture. Intended to be used for testing purposes. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body/blocks': {
                'num_layers': [6, 6, 6, 6],
                'growth_rate': 6,
            }
        })
        return config

class DenseNet121(DenseNet):
    """ DenseNet-121 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body/blocks': {
                'num_layers': [6, 12, 24, 32],
                'growth_rate': 32,
            }
        })
        return config

class DenseNet169(DenseNet):
    """ DenseNet-169 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body/blocks': {
                'num_layers': [6, 12, 32, 16],
                'growth_rate': 32,
            }
        })
        return config

class DenseNet201(DenseNet):
    """ DenseNet-201 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body/blocks': {
                'num_layers': [6, 12, 48, 32],
                'growth_rate': 32,
            }
        })
        return config

class DenseNet264(DenseNet):
    """ DenseNet-264 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'body/blocks': {
                'num_layers': [6, 12, 64, 48],
                'growth_rate': 32,
            }
        })
        return config
