"""  Howard A. et al "`MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
<https://arxiv.org/abs/1704.04861>`_"
"""
from .encoder_decoder import Encoder
from .blocks import MobileBlock

MOBILENET_V1_ENCODER_CONFIG = dict(layout = 'wna cna',
                                   strides=[1, 2, 1, 2, 1, 2, 1, 2, 1],
                                   rescale_filters=[True, True, False, True, False, True, False, True, False],
                                   n_reps=[1, 1, 1, 1, 1, 1, 5, 1, 1])

class MobileNet(Encoder):
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['initial_block'] += dict(layout='cna', filters=32, kernel_size=3, strides=2)

        config['body/encoder/num_stages'] = 9
        config['body/encoder/order'] = ['block']
        config['body/encoder/blocks'] += dict(base=MobileBlock, **MOBILENET_V1_ENCODER_CONFIG)

        config['head'] += dict(layout='Vf')

        config['loss'] = 'ce'
        return config
