"""
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia.
"`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"
"""

from . import EncoderDecoder, ResNet50
from .layers import pyramid_pooling



class PSPNet(EncoderDecoder):
    """ PSPNet model archtecture.
    By default, input is compressed with encoder, then processed via Pyramid Pooling layer,
    and .................................

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
        Dictionary with parameters for entry encoding: downsampling of the inputs.
        See :meth:`~.EncoderDecoder.encoder` arguments.

        embedding : dict
        Same as :meth:`~.EncoderDecoder.embedding`. Default is to use ASPP.

        decoder : dict
        Dictionary with parameters for decoding of compressed representation.
        See :meth:`~.EncoderDecoder.decoder` arguments. Default is to use bilinear upsampling.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/embedding'] = dict(base=pyramid_pooling)
        config['body/decoder'] = None
        config['body/order'] = None
        return config

class PSPNet50(PSPNet):
    """
    PSPNet with ResNet50 encoder.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)
        config['body/encoder'] += dict(base=ResNet50,
                                       order=None,
                                       downsample=[[], [0], [], []],
                                       filters=[64, 128, 256, 512],
                                       bottlenck=True)
        return config
