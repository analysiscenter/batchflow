"""
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia.
"`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"
"""

from . import EncoderDecoder, ResNet18, ResNet34, ResNet50
from .layers import pyramid_pooling


class PSPNet(EncoderDecoder):
    """ PSPNet model archtecture.
    By default, input is compressed with encoder, then processed via pyramid pooling layer,
    and without further upsampling passed as output through 1x1 convolution.

    Parameters
    ----------
    inputs : dict
        Dictionary (see :meth:`~.TFModel._make_inputs`).

    body : dict
        encoder : dict
        Dictionary with parameters for entry encoding or dictionary with
        base model implementing ``make_encoder`` method.
        See :meth:`~.EncoderDecoder.encoder` arguments.

        decoder : dict
        Dictionary with parameters for decoding of compressed representation.
        See :meth:`~.EncoderDecoder.decoder` arguments. Default is None.
    """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TFModel.default_config` """
        config = super().default_config()
        config['body/embedding'] = dict(base=pyramid_pooling)
        config['body/decoder'] = None
        return config

class PSPNet18(PSPNet):
    """
    PSPNet with ResNet18 encoder.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)
        config['body/encoder'] += dict(base=ResNet18,
                                       downsample=[[], [0], [], []],
                                       filters=[64, 128, 256, 512])
        return config

class PSPNet34(PSPNet):
    """
    PSPNet with ResNet34 encoder.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)
        config['body/encoder'] += dict(base=ResNet34,
                                       downsample=[[], [0], [], []],
                                       filters=[64, 128, 256, 512])
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
                                       downsample=[[], [0], [], []],
                                       filters=[64, 128, 256, 512])
        return config
