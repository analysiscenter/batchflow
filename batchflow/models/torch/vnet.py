"""  Milletari F. et al "`V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
<https://arxiv.org/abs/1606.04797>`_"
"""
from .encoder_decoder import EncoderDecoder
from .blocks import ResBlock



class VNet(EncoderDecoder):
    """ VNet-like model.

    Parameters
    ----------
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
        config = super().default_config()

        config['body/encoder/num_stages'] = 4
        config['body/encoder/order'] = ['block', 'skip', 'downsampling']
        config['body/encoder/blocks'] += dict(base=ResBlock, layout=['cna', 'cna'*2, 'cna'*3, 'cna'*3],
                                              filters=[16, 32, 64, 128], kernel_size=5)
        config['body/encoder/downsample'] += dict(layout='cna', filters=[32, 64, 128, 256], kernel_size=2, strides=2)

        config['body/embedding'] += dict(base=ResBlock, layout='cna'*3, filters=256, kernel_size=5)

        config['body/decoder/order'] = ['upsampling', 'combine', 'block']
        config['body/decoder/blocks'] += dict(base=ResBlock, layout=['cna'*3, 'cna'*3, 'cna'*2, 'cna'],
                                              filters=[256, 128, 64, 32], kernel_size=5)
        config['body/decoder/upsample'] += dict(layout='cna', filters=[128, 64, 32, 16], kernel_size=2, strides=2)

        config['loss'] = 'ce'
        return config

    def build_config(self):
        config = super().build_config()

        num_stages = config.get('body/encoder/num_stages')

        if config.get('body/encoder/blocks/filters') is None:
            config['body/encoder/blocks/filters'] = [16 * 2**i for i in range(num_stages)]

        if config.get('body/encoder/blocks/layout') is None:
            encoder_layout = ['cna', 'cna'*2] + ['cna'*3] * (num_stages - 2)
            config['body/encoder/blocks/layout'] = encoder_layout[:num_stages]

        if config.get('body/encoder/downsample/filters') is None:
            config['body/encoder/downsample/filters'] = [32 * 2**i for i in range(num_stages)]

        if config.get('body/embedding/filters') is None:
            downsample_filters = config['body/encoder/downsample/filters']
            config['body/embedding/filters'] = downsample_filters[-1]

        if config.get('body/embedding/filters') is None:
            config['body/embedding/filters'] = 16 * 2**num_stages

        if config.get('body/embedding/layout') is None:
            config['body/embedding/layout'] = 'cna' * 3

        if config.get('body/decoder/blocks/filters') is None:
            downsample_filters = config.get('body/encoder/downsample/filters')
            config['body/decoder/blocks/filters'] = downsample_filters[::-1]

        if config.get('body/decoder/blocks/layout') is None:
            layout = config.get('body/encoder/blocks/layout')
            config['body/decoder/blocks/layout'] = layout[::-1]

        if config.get('body/decoder/upsample/filters') is None:
            encoder_filters = config.get('body/encoder/blocks/filters')
            config['body/decoder/upsample/filters'] = encoder_filters[::-1]

        return config
