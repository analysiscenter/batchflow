"""
Huang G. et al. "`Densely Connected Convolutional Networks
<https://arxiv.org/abs/1608.06993>`_"
Jegou S. et al "`The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation
<https://arxiv.org/abs/1611.09326>`_"
"""
from .encoder_decoder import Encoder, EncoderDecoder
from .blocks import DenseBlock



class DenseNet(Encoder):
    """ DenseNet architecture. """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TorchModel.default_config`"""
        config = super().default_config()
        config['common/conv/bias'] = False
        config['initial_block'] += dict(layout='cnap', filters=16,
                                        kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/blocks'] += dict(base=DenseBlock, layout='nacd',
                                              num_layers=None, growth_rate=None,
                                              skip=True, bottleneck=True,
                                              dropout_rate=0.2, filters=None)
        config['body/encoder/downsample'] += dict(layout='nacv', kernel_size=1, strides=1,
                                                  pool_size=2, pool_strides=2,
                                                  filters='same')
        config['head'] += dict(layout='Vf')
        return config


class DenseNetS(DenseNet):
    """ Small verison of DenseNet architecture. Intended to be used for testing purposes mainly. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/num_stages'] = 2
        config['body/encoder/blocks'] += dict(num_layers=[4, 4], growth_rate=32)
        return config

class DenseNet121(DenseNet):
    """ DenseNet-121 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=[6, 12, 24, 32],
                                              growth_rate=32)
        return config

class DenseNet169(DenseNet):
    """ DenseNet-169 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=[6, 12, 32, 16],
                                              growth_rate=32)
        return config

class DenseNet201(DenseNet):
    """ DenseNet-201 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=[6, 12, 48, 32],
                                              growth_rate=32)
        return config

class DenseNet264(DenseNet):
    """ DenseNet-264 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=[6, 12, 64, 48],
                                              growth_rate=32)
        return config



class SegmentationDenseNet(EncoderDecoder):
    """ FC DenseNet architecture for segmentation tasks. """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TorchModel.default_config`"""
        config = super().default_config()
        config['common/conv/bias'] = False
        config['initial_block'] += dict(layout='c', filters=48, kernel_size=3, strides=1)

        config['body/encoder/num_stages'] = 6
        config['body/encoder/blocks'] += dict(base=DenseBlock, layout='nacd',
                                              num_layers=None, growth_rate=None,
                                              skip=True, bottleneck=False,
                                              dropout_rate=0.2, filters=None)
        config['body/encoder/downsample'] += dict(layout='nacdp', kernel_size=1, strides=1,
                                                  pool_size=2, pool_strides=2, dropout_rate=.2,
                                                  filters='same')

        config['body/embedding'] += dict(base=DenseBlock, num_layers=None, growth_rate=None)

        config['body/decoder/blocks'] += dict(base=DenseBlock, layout='nacd',
                                              num_layers=None, growth_rate=None,
                                              skip=False, bottleneck=False,
                                              dropout_rate=0.2, filters=None)
        config['body/decoder/upsample'] += dict(layout='t')
        config['body/decoder/order'] = ['upsample', 'combine', 'block']
        return config


class DenseNetFC56(SegmentationDenseNet):
    """ FC DenseNet-56 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=4, growth_rate=12)
        config['body/embedding'] += dict(num_layers=4, growth_rate=12)
        config['body/decoder/blocks'] += dict(num_layers=4, growth_rate=12)
        return config

class DenseNetFC67(SegmentationDenseNet):
    """ FC DenseNet-67 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=5, growth_rate=16)
        config['body/embedding'] += dict(num_layers=5, growth_rate=16)
        config['body/decoder/blocks'] += dict(num_layers=5, growth_rate=16)
        return config

class DenseNetFC103(SegmentationDenseNet):
    """ FC DenseNet-103 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks'] += dict(num_layers=[4, 5, 7, 10, 12, 15],
                                              growth_rate=16)
        config['body/embedding'] += dict(num_layers=15, growth_rate=16)
        config['body/decoder/blocks'] += dict(num_layers=[4, 5, 7, 10, 12, 15],
                                              growth_rate=16)
        return config
