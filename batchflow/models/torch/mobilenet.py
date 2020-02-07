""" Howard A. et al. "`MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
<https://arxiv.org/abs/1704.04861>`_"

Sandler M. et al. "`MobileNetV2: Inverted Residuals and Linear Bottlenecks
<https://arxiv.org/abs/1801.04381>`_"

Howard A. et al. "`Searching for MobileNetV3
<https://arxiv.org/abs/1905.02244>`_"
"""
import torch

from .encoder_decoder import Encoder
from .blocks import MobileBlock

MOBILENET_V1_ENCODER_CONFIG = dict(layout='wna cna',
                                   strides=[1, 2, 1, 2, 1, 2, 1, 2, 1],
                                   final_filters=[64, 128, 128, 256, 256, 512, 512, 512, 1024],
                                   n_reps=[1, 1, 1, 1, 1, 1, 5, 1, 1])

MOBILENET_V2_ENCODER_CONFIG = dict(layout='cna wna c',
                                   residual=True,
                                   strides=[1, 2, 2, 2, 1, 2, 1],
                                   expansion=[1, 6, 6, 6, 6, 6, 6],
                                   final_filters=[16, 24, 32, 64, 96, 160, 320],
                                   n_reps=[1, 2, 3, 4, 3, 3, 1])

class MobileNetV1(Encoder):
    """ MobileNet.

    Parameters
    ----------
    body : dict
        encoder : dict
                    num_stages : int
                        number of downsampling blocks
                    blocks : dict
                        Parameters for MobileBlock.
                        See :class:`~.blocks.MobileBlock` documentation.

    Notes
    -----
    For more parameters see :class:`~.EncoderDecoder`.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['initial_block'] += dict(layout='cna', filters=32, kernel_size=3, strides=2)

        config['body/encoder/num_stages'] = len(MOBILENET_V1_ENCODER_CONFIG['n_reps'])
        config['body/encoder/order'] = ['block']
        config['body/encoder/blocks'] += dict(base=MobileBlock, **MOBILENET_V1_ENCODER_CONFIG)

        config['head'] += dict(layout='Vf')

        config['loss'] = 'ce'
        return config

class MobileNetV2(Encoder):
    """ MobileNetV2.

    Parameters
    ----------
    body : dict
        encoder : dict
                    num_stages : int
                        number of downsampling blocks
                    blocks : dict
                        Parameters for MobileBlock.
                        See :class:`~.block.MobileBlock` documentation.

    Notes
    -----
    For more parameters see :class:`~.EncoderDecoder`.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/activation'] = torch.nn.ReLU6

        config['initial_block'] += dict(layout='cna', filters=32, kernel_size=3, strides=2)

        config['body/encoder/num_stages'] = len(MOBILENET_V2_ENCODER_CONFIG['n_reps'])
        config['body/encoder/order'] = ['block']
        config['body/encoder/blocks'] += dict(base=MobileBlock, **MOBILENET_V2_ENCODER_CONFIG)

        config['head'] += dict(layout='cnaV rcr', filters=[1280, None], kernel_size=1,
                               strides=1, reshape_to=[[-1, 1], [-1]])

        config['loss'] = 'ce'
        return config
