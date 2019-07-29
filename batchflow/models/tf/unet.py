"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""

from .encoder_decoder import EncoderDecoder


class UNet(EncoderDecoder):
    """ UNet-like model

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder/num_stages : int
            number of downsampling/upsampling blocks (default=4)
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common'] = dict(conv=dict(use_bias=False))

        config['initial_block'] = dict(layout='cna cna', kernel_size=3, filters=64)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/blocks'] = dict(layout='cna cna', kernel_size=3)
        config['body/embedding'] = dict(layout='')  # identity
        config['body/decoder/blocks'] = dict(layout='cna cna', kernel_size=3)

        config['head'] = dict(layout='cna cna', kernel_size=[3, 3], strides=1, filters=64)

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        num_stages = config.get('body/encoder/num_stages')

        if config.get('body/encoder/blocks/filters') is None:
            config['body/encoder/blocks/filters'] = [128*2**i for i in range(num_stages)]

        if config.get('body/decoder/blocks/filters') is None:
            config['body/decoder/blocks/filters'] = list(reversed([64*2**i for i in range(num_stages)]))

        if config.get('body/decoder/upsample/filters') is None:
            config['body/decoder/upsample/filters'] = list(reversed([128*2**i for i in range(num_stages)]))

        return config
