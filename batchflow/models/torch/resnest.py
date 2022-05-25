"""
Hang Zhang et al. "`ResNeSt: Split-Attention Networks
<https://arxiv.org/abs/2004.08955>`_"
"""
from .base import TorchModel
from .blocks import ResNeStBlock

class ResNeSt(TorchModel):
    """ Base ResNeSt architecture. """
    @classmethod
    def default_config(cls):
        """ Define model's defaults: general architecture. """
        config = super().default_config()

        config.update({
            'initial_block': {
                'layout': 'cnacnacnap',
                'channels': [32, 32, 64],
                'kernel_size': 3,
                'bias': False,
                'padding': 'same',
                'pool_size': 3,
                'pool_stride': 2,
                'stride': [2, 1, 1]
            },
            'body': {
                'type': 'encoder',
                'output_type': 'tensor',
                'num_stages': 4,
                'order': ['block'],
                'blocks': {
                    'base_block': ResNeStBlock,
                    'n_reps': [1, 1, 1, 1],
                    'stride': [1, 2, 2, 2],
                    'channels': [64, 128, 256, 512],
                    'radix': 2,
                    'cardinality': 1,
                    'reduction_factor': 4,
                    'scaling_factor': 1,
                }
            },
            'head': {
                'layout': 'Vdf',
                'dropout_rate': 0.4,
            },

            'loss': 'ce'
        })
        return config



class ResNeSt18(ResNeSt):
    """ ResNeSt-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [2, 2, 2, 2]
        return config

class ResNeSt34(ResNeSt):
    """ ResNeSt-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [3, 4, 6, 3]
        return config
