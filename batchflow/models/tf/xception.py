"""
François Chollet. "`Xception: Deep Learning with Depthwise Separable Convolutions
<https://arxiv.org/abs/1610.02357>`_"
"""

import tensorflow.compat.v1 as tf

from . import TFModel
from .layers import conv_block, depthwise_conv, combine
from ..utils import unpack_args



class Xception(TFModel):
    """ Xception model architecture.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`).

    body : dict
        entry, middle, exit : dict
        Dictionary with parameters for entry encoding: downsampling of the inputs.

            num_stages : int
                Number of `block`'s in the respective flow.
            filters : list of sequences of 3 ints
                Number of filters inside for individual `block`.
            strides : int or list of ints
                Stride of the middle `separable_block` inside `block`.
            depth_activation : bool or list of bools
                Whether to use activation between depthwise and pointwise convolutions.
            combine_op : {'sum', 'softsum'}
                Whether to use convolution for skip-connections inside `separable_block` or
                just sum skip and output.
    """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TFModel.default_config` """
        config = TFModel.default_config()
        config['body/entry'] = dict(num_stages=None, filters=None, strides=2, combine_op='softsum')
        config['body/middle'] = dict(num_stages=None, filters=None, strides=1, combine_op='sum')
        config['body/exit'] = dict(num_stages=None, filters=None, strides=1,
                                   depth_activation=True, combine_op='softsum')
        config['body/order'] = ['entry', 'middle', 'exit']
        return config

    def build_config(self, names=None):
        """ Define model's architecture configuration. See :meth: `~.TFModel.build_config` """
        config = super().build_config(names)
        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Multiple consecutive blocks. """
        kwargs = cls.fill_params('body', **kwargs)

        with tf.variable_scope(name):
            x = inputs

            steps = kwargs.pop('num_stages', 0)
            for i in range(steps):
                with tf.variable_scope('group-'+str(i)):
                    args = {**kwargs, **unpack_args(kwargs, i, steps)}
                    x = cls.block(x, name='block-'+str(i), **args)
                    x = tf.identity(x, name='output')
        return x

    @classmethod
    def block(cls, inputs, filters, combine_op='softsum', name='block', **kwargs):
        """ Basic building block of the architecture.
        For details see figure 5 in the article.

        Parameters
        ----------
        filters : sequence of 3 ints
            Number of feature maps in each layer
        """
        strides = (1, kwargs.pop('strides', 1), 1)
        x = inputs

        with tf.variable_scope(name):
            # Three consecutive separable blocks
            for i, filter_ in enumerate(filters):
                x = cls.separable_block(x, filter_, strides=strides[i],
                                        name='separable_conv-{}'.format(i), **kwargs)

            outputs = combine([x, inputs], op=combine_op, strides=strides[1],
                              data_format=kwargs.get('data_format'))
        return outputs

    @classmethod
    def separable_block(cls, inputs, filters, kernel_size=3, strides=1, rate=1,
                        depth_activation=False, name='separable_block', **kwargs):
        """ Separable convolution, followed by pointwise convolution with batch-normalization in-between. """
        layout = 'nacna' if depth_activation else 'ncn'
        data_format = kwargs.get('data_format')

        with tf.variable_scope(name):
            x = depthwise_conv(inputs, kernel_size=kernel_size, strides=strides, dilation_rate=rate,
                               data_format=data_format, padding='same', name='depthwise')
            x = conv_block(x, layout=layout, filters=filters, kernel_size=1, **kwargs)
        return x



class Xception41(Xception):
    """ Xception-41 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] += dict(num_stages=3,
                                     filters=[[128]*3,
                                              [256]*3,
                                              [728]*3])

        config['body/middle'] += dict(num_stages=8,
                                      filters=[[728]*3]*8)

        config['body/exit'] += dict(num_stages=2, strides=[2, 1],
                                    filters=[[728, 1024, 1024],
                                             [1536, 1536, 2048]])
        return config


class Xception64(Xception):
    """ Xception-64 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] += dict(num_stages=3,
                                     filters=[[128]*3,
                                              [256]*3,
                                              [728]*3])

        config['body/middle'] += dict(num_stages=16,
                                      filters=[[728]*3]*16)

        config['body/exit'] += dict(num_stages=2, strides=[2, 1],
                                    filters=[[728, 1024, 1024],
                                             [1536, 1536, 2048]])
        return config


class XceptionS(Xception):
    """ Small version of Xception architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] += dict(num_stages=2,
                                     filters=[[6]*3,
                                              [12]*3,])
        config['body/middle'] += dict(num_stages=2,
                                      filters=[[12]*3]*2)
        config['body/exit'] += dict(num_stages=2, strides=[2, 1],
                                    filters=[[12]*3,
                                             [15]*3,])
        return config
