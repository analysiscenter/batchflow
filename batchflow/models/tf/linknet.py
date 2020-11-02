""" Chaurasia A., Culurciello E. "`LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
<https://arxiv.org/abs/1707.03718>`_"

NOTE: The fourth section of the article describes the method of weighing imbalance datasets,
      it does not exist in this implementation.

"""

from .encoder_decoder import EncoderDecoder
from .resnet import ResNet


class LinkNet(EncoderDecoder):
    """ LinkNet

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

        decoder : dict
            num_stages : int
                number of upsampling blocks. Defaults to the number of downsamplings.

            factor : None, int or list of ints
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage
                If not given or None, defaults to [2]*num_stages

            blocks : dict
                Parameters for post-processing blocks:

                filters : int, list of ints or list of lists of ints
                    same as encoder/blocks/filters

            upsample : dict
                Parameters for upsampling (see :func:`~.layers.upsample`).

                filters : int, list of ints or list of lists of ints
                    same as encoder/blocks/filters

    for more parameters see (see :class:`~.EncoderDecoder`)
    """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TFModel.default_config` """
        config = super().default_config()

        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/downsample'] += dict(layout=None)
        config['body/encoder/blocks'] += dict(base=ResNet.double_block,
                                              layout='cna cna',
                                              filters=[64, 128, 256, 512], kernel_size=3)

        config['body/embedding'] = None

        config['body/decoder/upsample'] += dict(layout=None)
        config['body/decoder/blocks'] += dict(layout='cna tna cna',
                                              filters=[[128, 128, 256],
                                                       [64, 64, 128],
                                                       [32, 32, 64],
                                                       [16, 16, 64]],
                                              kernel_size=[[1, 3, 1]]*4,
                                              strides=[[1, 2, 1]]*4)

        config['head'] += dict(layout='tna cna t', filters=32, kernel_size=[3, 3, 2], strides=[2, 1, 2])

        config['loss'] = 'ce'

        return config
