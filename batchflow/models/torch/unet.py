"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import warnings

from .encoder_decoder import EncoderDecoder
from .blocks import ResBlock, DenseBlock



class UNet(EncoderDecoder):
    """ UNet-like model.

    Parameters
    ----------
    auto_build : dict, optional
        Parameters for auto-building `filters` in accordance with the idea described in the original paper.
        Note that any of `filters`, if passed, will be replaced by auto-built ones.

        num_stages : int
            number of encoder/decoder stages — defines network depth and the number of its skip connections
        filters : int, optional
            number of filters in first encoder block — each of the following ones will be doubled until embedding

    body : dict
        encoder : dict
            num_stages : int
                number of downsampling blocks (default=4)
            blocks : dict
                Parameters for pre-processing blocks:

                filters : None, int, list of ints or list of lists of ints
                    The number of filters in the output tensor.
                    If int, same number of filters applies to all layers on all stages
                    If list of ints, specifies number of filters in each layer of different stages
                    If list of list of ints, specifies number of filters in different layers on different stages
                    If not given or None, filters parameters in encoder/blocks, decoder/blocks and decoder/upsample
                    default to values which make number of filters double
                    on each stage of encoding and halve on each stage of decoding,
                    provided that `decoder/skip` is `True`. Specify `filters=None` explicitly
                    if you want to use custom `num_steps` and infer `filters`

        decoder : dict
            num_stages : int
                number of upsampling blocks. Defaults to the number of downsamplings.

            factor : None, int or list of ints
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage
                If not given or None, defaults to [2]*num_stages

            blocks : dict
                Parameters for post-processing blocks:

                filters : None, int, list of ints or list of lists of ints
                    same as encoder/blocks/filters

            upsample : dict
                Parameters for upsampling (see :func:`~.layers.upsample`).

                filters : int, list of ints or list of lists of ints
                    same as encoder/blocks/filters

    Notes
    -----
    For more parameters see :class:`~.EncoderDecoder`.
    """
    @classmethod
    def default_config(cls):
        """ Define model's defaults. """
        config = super().default_config()

        config['body/encoder/num_stages'] = 4
        config['body/encoder/order'] = ['block', 'skip', 'downsampling']
        config['body/encoder/blocks'] += dict(layout='cna cna', kernel_size=3, filters=[64, 128, 256, 512])

        config['body/embedding'] += dict(layout='cna cna', kernel_size=3, filters=1024)

        config['body/decoder/order'] = ['upsampling', 'combine', 'block']
        config['body/decoder/blocks'] += dict(layout='cna cna', kernel_size=3, filters=[512, 256, 128, 64])

        config['loss'] = 'ce'
        return config

    def build_config(self):
        """ Update architecture configuration, if needed. """
        config = super().build_config()

        if config.get('auto_build'):
            num_stages = config.get('auto_build/num_stages', 4)
            filters = config.get('auto_build/filters', 64)
            encoder_filters = [filters * 2**i for i in range(num_stages)]

            config['body/encoder/num_stages'] = num_stages
            config['body/encoder/blocks/filters'] = encoder_filters
            config['body/embedding/filters'] = encoder_filters[-1] * 2
            config['body/decoder/num_stages'] = num_stages
            config['body/decoder/blocks/filters'] = encoder_filters[::-1]
            config['body/decoder/upsample/filters'] = encoder_filters[::-1]

        if not config.get('body/decoder/upsample/filters'):
            warnings.warn("'decoder/upsample/filters' are not set and " +
                          "can be inconsistent with 'decoder/blocks/filters'! Please revise your model's config. " +
                          "In future, upsample filters can be made to match decoder block's filters by default.")

        return config


class ResUNet(UNet):
    """ UNet with residual blocks. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(base=ResBlock, layout='cna', n_reps=2)
        config['body/decoder/blocks'] += dict(base=ResBlock, layout='cna', n_reps=2)
        return config


class DenseUNet(UNet):
    """ UNet with dense blocks. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(base=DenseBlock, layout='nacd', skip=True)
        config['body/decoder/blocks'] += dict(base=DenseBlock, layout='nacd', skip=False)
        return config
