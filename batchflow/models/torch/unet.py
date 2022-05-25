"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import warnings

from .base import TorchModel
from .blocks import ResBlock, DenseBlock



class UNet(TorchModel):
    """ UNet-like model.

    Parameters
    ----------
    auto_build : dict, optional
        Parameters for auto-building `channels` in accordance with the idea described in the original paper.
        Note that any of `channels`, if passed, will be replaced by auto-built ones.

        num_stages : int
            number of encoder/decoder stages — defines network depth and the number of its skip connections
        channels : int, optional
            number of channels in first encoder block — each of the following ones will be doubled until embedding

    encoder : dict
        num_stages : int
            number of downsampling blocks (default=4)
        blocks : dict
            Parameters for pre-processing blocks:

            channels : None, int, list of ints or list of lists of ints
                The number of channels in the output tensor.
                If int, same number of channels applies to all layers on all stages
                If list of ints, specifies number of channels in each layer of different stages
                If list of list of ints, specifies number of channels in different layers on different stages
                If not given or None, channels parameters in encoder/blocks, decoder/blocks and decoder/upsample
                default to values which make number of channels double
                on each stage of encoding and halve on each stage of decoding,
                provided that `decoder/skip` is `True`. Specify `channels=None` explicitly
                if you want to use custom `num_steps` and infer `channels`

    decoder : dict
        num_stages : int
            number of upsampling blocks. Defaults to the number of downsamplings.

        blocks : dict
            Parameters for post-processing blocks:

            channels : None, int, list of ints or list of lists of ints
                same as encoder/blocks/channels

        upsample : dict
            Parameters for upsampling (see :func:`~.layers.upsample`).

            channels : int, list of ints or list of lists of ints
                same as encoder/blocks/channels

    Notes
    -----
    For more parameters see :class:`~.EncoderDecoder`.
    """
    @classmethod
    def default_config(cls):
        """ Define model's defaults. """
        config = super().default_config()

        config.update({
            'auto_build': False,

            'order': ['initial_block', 'encoder', 'embedding', 'decoder', 'head'],

            'encoder': {
                'type': 'encoder',
                'num_stages': 4,
                'order': ['block', 'skip', 'downsampling'],
                'blocks': {
                    'layout': 'cna cna',
                    'kernel_size': 3,
                    'channels': [64, 128, 256, 512]
                }
            },

            'embedding': {
                'input_type': 'list',
                'input_idx': -1,
                'output_type': 'list',
                'layout': 'cna cna',
                'kernel_size': 3,
                'channels': 1024,
            },

            'decoder': {
                'type': 'decoder',
                'order': ['upsampling', 'combine', 'block'],
                'blocks': {
                    'layout': 'cna cna',
                    'kernel_size': 3,
                    'channels': [512, 256, 128, 64]
                }
            },

            'head': {
                'layout': 'c',
                'channels': 1
            }
        })

        config['loss'] = 'ce'
        return config

    def update_config(self):
        """ Update architecture configuration, if needed. """
        super().update_config()
        config = self.full_config

        if config.get('auto_build'):
            num_stages = config.get('auto_build/num_stages', 4)
            channels = config.get('auto_build/channels', 64)
            encoder_channels = [channels * 2**i for i in range(num_stages)]

            config['encoder/num_stages'] = num_stages
            config['encoder/blocks/channels'] = encoder_channels
            config['embedding/channels'] = encoder_channels[-1] * 2
            config['decoder/num_stages'] = num_stages
            config['decoder/blocks/channels'] = encoder_channels[::-1]
            config['decoder/upsample/channels'] = encoder_channels[::-1]

        if not config.get('decoder/upsample/channels'):
            warnings.warn("'decoder/upsample/channels' are not set and " + \
                          "can be inconsistent with 'decoder/blocks/channels'! Please revise your model's config. " + \
                          "In the future, upsample channels can be made to match decoder block's channels by default.")


class ResUNet(UNet):
    """ UNet with residual blocks. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['encoder/blocks'] += dict(base_block=ResBlock, layout='cna', n_reps=1)
        config['decoder/blocks'] += dict(base_block=ResBlock, layout='cna', n_reps=1)
        return config


class DenseUNet(UNet):
    """ UNet with dense blocks. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['encoder/blocks'] += dict(base_block=DenseBlock, layout='nacd', skip=True)
        config['decoder/blocks'] += dict(base_block=DenseBlock, layout='nacd', skip=False)
        return config
