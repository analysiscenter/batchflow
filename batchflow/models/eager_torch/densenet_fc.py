""" Jegou S. et al "`The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation
<https://arxiv.org/abs/1611.09326>`_"
"""

from . import EncoderDecoder, DenseBlock

class DenseNetFC56(EncoderDecoder):
    """ FC DenseNet-56 architecture """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False
        config['initial_block'] += dict(layout='c', filters=48, kernel_size=3, strides=1)
        config['body/encoder/num_stages'] = 6
        config['body/encoder/downsample'] = dict(layout='nacdp', kernel_size=1, strides=1,
                                                 pool_size=2, pool_strides=2, dropout_rate=.2,
                                                 filters='same')
        config['body/encoder/blocks'] = dict(base=DenseBlock, layout='nacd', 
                                             num_layers=[4] * 6, growth_rate=12,
                                             dropout_rate=0.2, skip=True, filters=None)

        config['body/decoder/blocks'] = dict(base=DenseBlock, layout='nacd', 
                                             num_layers=[4] * 6, growth_rate=12,
                                             dropout_rate=0.2, skip=False, filters=None)
        config['body/decoder/upsample'] = dict(layout='t')
        config['body/decoder/order'] = 'ucb'

        config['head'] +=(dict(layout='c', kernel_size=1, filters=2))
        return config
