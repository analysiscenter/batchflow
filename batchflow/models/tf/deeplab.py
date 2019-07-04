"""
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
"`Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation
<https://arxiv.org/abs/1802.02611>`_"
"""
import tensorflow as tf

from .layers import aspp
from . import EncoderDecoder, Xception



class DeepLab(EncoderDecoder):
    """ DeepLab v3+ model archtecture.
    By default, inpu is encoded with Xception backbone (entry, middle and exit flows),
    then worked via Atrous Spatial Pyramid Pooling layer, and, finally, restored to original
    spatial size with simple yet effective decoder module.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        entry : dict
        Dictionary with parameters for entry encoding: downsampling of the inputs.
        See :meth:`~.EncoderDecoder.encoder` arguments.

        middle : dict
        Dictionary with parameters for middle encoding: thorough processing.
        See :meth:`~.EncoderDecoder.embedding` arguments.

        exit : dict
        Dictionary with parameters for exit encoding: final increase in number of features.
        See :meth:`~.EncoderDecoder.embedding` arguments.

        embedding : dict
        Same as :meth:`~.EncoderDecoder.embedding`. Default is to use ASPP.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] = dict(base=None, num_stages=None)
        config['body/entry/downsample'] = dict(layout=None)
        config['body/entry/blocks'] = dict(base=Xception.block, filters=None, strides=2, combine_type='conv')

        config['body/middle'] = dict(base=Xception.block, num_stages=None,
                                     filters=None, strides=1, combine_type='sum')

        config['body/exit'] = dict(base=Xception.block, num_stages=None,
                                   filters=None, strides=1, depth_activation=True, combine_type='conv')

        config['body/embedding'] = dict(base=aspp)

        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None)
        config['body/decoder/upsample'] = dict(layout='b')
        config['body/decoder/blocks'] = dict(base=cls.block)
        config['head'] = dict(layout='c', kernel_size=1)
        return config


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and decoder. """
        kwargs = cls.fill_params('body', **kwargs)
        entry = kwargs.pop('entry')
        middle = kwargs.pop('middle')
        exit = kwargs.pop('exit')
        embedding = kwargs.pop('embedding')
        decoder = kwargs.pop('decoder')

        with tf.variable_scope(name):
            # Entry flow: transition down
            encoder_outputs = cls.encoder(inputs, name='entry', **entry, **kwargs)

            # Middle flow: thorough processing without changing neither spatial resolution, nor number of filters
            middle = cls.embedding(encoder_outputs[-1], name='middle', **middle, **kwargs)

            # Exit flow: increase amount of features maps
            exit = cls.embedding(middle, name='exit', **exit, **kwargs)

            # Bottleneck: ASPP
            x = cls.embedding(exit, **embedding, **kwargs)
            encoder_outputs.append(x)

            # Decoder: transition up
            x = cls.decoder(encoder_outputs, **decoder, **kwargs)
        return x


class DeepLab8(DeepLab):
    """ DeepLab with output stride 8. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] = dict(layout='cnacna', filters=32, kernel_size=3, strides=1)

        config['body/entry/num_stages'] = 4
        config['body/entry/blocks/filters'] = [[64]*3,
                                               [128]*3,
                                               [256]*3,
                                               [728]*3]
        config['body/entry/blocks/strides'] = [2, 2, 2, 1]

        config['body/middle/num_stages'] = 8
        config['body/middle/rate'] = 2 # not sure if actually needed
        config['body/middle/filters'] = [[728]*3]*8

        config['body/exit/num_stages'] = 2
        config['body/exit/rate'] = [2, 4]
        config['body/exit/filters'] = [[728, 1024, 1024],
                                       [1536, 1536, 2048]]

        config['body/embedding/rates'] = (12, 24, 36)

        config['body/decoder'] = dict(skip=True, num_stages=4, factor=[1, 2, 1, 4])
        return config


class DeepLab16(DeepLab):
    """ DeepLab with output stride 16. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] = dict(layout='cnacna', filters=32, kernel_size=3, strides=1)

        config['body/entry/num_stages'] = 4
        config['body/entry/blocks/filters'] = [[64]*3,
                                               [128]*3,
                                               [256]*3,
                                               [728]*3]
        config['body/entry/blocks/strides'] = [2, 2, 2, 2]

        config['body/middle/num_stages'] = 16
        config['body/middle/rate'] = 2 # not sure if actually needed
        config['body/middle/filters'] = [[728]*3]*16

        config['body/exit/num_stages'] = 2
        config['body/exit/rate'] = [1, 2]
        config['body/exit/filters'] = [[728, 1024, 1024],
                                       [1536, 1536, 2048]]

        config['body/embedding/rates'] = (6, 12, 18)

        config['body/decoder'] = dict(skip=True, num_stages=4, factor=[1, 4, 1, 4])
        return config


class DeepLabS(DeepLab):
    """ DeepLab with output stride 16. Smaller than the others. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] = dict(layout='cna', filters=32, kernel_size=3, strides=1)

        config['body/entry/num_stages'] = 4
        config['body/entry/blocks/filters'] = [[6]*3,
                                               [1]*3,
                                               [2]*3,
                                               [7]*3]
        config['body/entry/blocks/strides'] = [2, 2, 2, 2]

        config['body/middle/num_stages'] = 4
        config['body/middle/rate'] = 2 # not sure if actually needed
        config['body/middle/filters'] = [[7]*3]*4

        config['body/exit/num_stages'] = 2
        config['body/exit/rate'] = [2, 4]
        config['body/exit/filters'] = [[7, 10, 10],
                                       [10, 10, 12]]

        config['body/embedding/rates'] = (6, 12, 18)

        config['body/decoder'] = dict(skip=True, num_stages=4, factor=[1, 4, 1, 4])
        return config
