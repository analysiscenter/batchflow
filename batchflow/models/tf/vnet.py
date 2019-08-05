"""  Milletari F. et al "`V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
<https://arxiv.org/abs/1606.04797>`_"
"""

from .encoder_decoder import EncoderDecoder
from .resnet import ResNet


class VNet(EncoderDecoder):
    """ VNet-like model

  **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
            num_stages : int
                number of downsampling blocks (default=4)

            blocks : dict
                Parameters for pre-processing blocks:

                filters : int, list of ints or list of lists of ints
                    The number of filters in the output tensor.
                    If int, same number of filters applies to all layers on all stages
                    If list of ints, specifies number of filters in each layer of different stages
                    If list of list of ints, specifies number of filters in different layers on different stages
                    If not given or None, filters parameters in encoder/blocks, decoder/blocks and decoder/upsample
                    default to values which make number of filters double
                    on each stage of encoding and halve on each stage of decoding,
                    provided that `decoder/skip` is `True`. Number of filters on the first stage of encoding will be
                    doubled number of filters in initial_block's output
                    When defining filters manually mind the filters defined in initial_block and head

            downsample : dict
                Parameters for downsampling (see :func:`~.layers.conv_block`)

        decoder : dict
            num_stages : int
                number of upsampling blocks. Defaults to the number of downsamplings.

            blocks : dict
                Parameters for post-processing blocks:

                filters : int, list of ints or list of lists of ints
                    same as encoder/blocks/filters but in reverse order

            upsample : dict
                Parameters for upsampling (see :func:`~.layers.upsample`).

    for more parameters see (see :class:`~.EncoderDecoder`)
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['initial_block'] = dict(layout='cna R cna +', kernel_size=5, filters=16)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/downsample'] = dict(layout='cna', kernel_size=2, strides=2)
        config['body/encoder/blocks'] = dict(base=ResNet.block, layout=None, filters=None, kernel_size=5)

        config['body/embedding'] = None

        config['body/decoder/upsample'] = dict(layout='tna', kernel_size=2, strides=2)
        config['body/decoder/blocks'] = dict(base=ResNet.block, layout=None, filters=None, kernel_size=5)

        config['head'] = dict(layout='R cna + cna', kernel_size=5, filters=[48, 32])

        config['loss'] = 'ce'
        return config

def build_config(self, names=None):
        config = super().build_config(names)

        num_stages = config.get('body/encoder/num_stages')

        initial_block_filters = config.get('initial_block/filters')
        f_0 = initial_block_filters[-1] if isinstance(initial_block_filters, (list, tuple)) else initial_block_filters

        encoder_filters = [f_0*2**(i+1) for i in range(num_stages)]
        decoder_filters = encoder_filters[::-1]

        if config.get('body/encoder/blocks/layout') is None:
            config['body/encoder/blocks/layout'] = ['cna'*2 if i<1 else 'cna'*3 for i in range(num_stages)]

        if config.get('body/encoder/blocks/filters') is None:
            config['body/encoder/blocks/filters'] = encoder_filters

        if config.get('body/encoder/downsample/filters') is None:
            config['body/encoder/downsample/filters'] = encoder_filters

        if config.get('body/decoder/blocks/layout') is None:
            config['body/decoder/blocks/layout'] = config.get('body/encoder/blocks/layout')[::-1]

        if config.get('body/decoder/blocks/filters') is None:
            config['body/decoder/blocks/filters'] = decoder_filters

        if config.get('body/decoder/upsample/filters') is None:
            config['body/decoder/upsample/filters'] = decoder_filters

        return config
