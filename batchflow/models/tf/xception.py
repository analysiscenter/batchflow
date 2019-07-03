"""
François Chollet. "`Xception: Deep Learning with Depthwise Separable Convolutions
<https://arxiv.org/abs/1610.02357>`_"
"""

import tensorflow as tf

from .layers import conv_block
from . import TFModel



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
            combine_type : {'sum', 'conv'}
                Whether to use convolution for skip-connections inside `separable_block` or
                just sum skip and output.
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body/entry'] = dict(num_stages=None, filters=None, strides=2, combine_type='conv')
        config['body/middle'] = dict(num_stages=None, filters=None, strides=1, combine_type='sum')
        config['body/exit'] = dict(num_stages=None, filters=None, strides=1, depth_activation=True, combine_type='conv')
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config


    @classmethod
    def slice_dict(cls, dictionary, idx):
        return {key: value[idx] for key, value in dictionary.items()
                if isinstance(value, list)}


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Entry, middle and exit flows consequently. """
        kwargs = cls.fill_params('body', **kwargs)
        entry = kwargs.pop('entry')
        middle = kwargs.pop('middle')
        exit = kwargs.pop('exit')

        with tf.variable_scope(name):
            x = inputs

            # Entry flow: downsample the inputs
            with tf.variable_scope('entry_flow'):
                entry_stages = entry.pop('num_stages', 0)
                for i in range(entry_stages):
                    args = {**kwargs, **entry, **cls.slice_dict(entry, i)}
                    x = cls.block(x, name='block-'+str(i), **args)

            # Middle flow: thorough processing
            with tf.variable_scope('middle_flow'):
                middle_stages = middle.pop('num_stages', 0)
                for i in range(middle_stages):
                    args = {**kwargs, **middle, **cls.slice_dict(middle, i)}
                    x = cls.block(x, name='block-'+str(i), **args)

            # Exit flow: final increase in number of feature maps
            with tf.variable_scope('exit_flow'):
                exit_stages = exit.pop('num_stages', 0)
                for i in range(exit_stages):
                    args = {**kwargs, **exit, **cls.slice_dict(exit, i)}
                    x = cls.block(x, name='block-'+str(i), **args)
        return x


    @classmethod
    def block(cls, inputs, filters, combine_type='sum', name='block', **kwargs):
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
            # Prepare skip-connection
            if combine_type == 'sum':
                shortcut = inputs
            elif combine_type == 'conv':
                shortcut = conv_block(inputs, 'cn', filters[-1], kernel_size=1,
                                      strides=strides[1], name='shortcut')

            # Three consecutive separable blocks
            for i, filter_ in enumerate(filters):
                x = cls.separable_block(x, filter_, strides=strides[i],
                                        name='separable_conv-{}'.format(i), **kwargs)

            outputs = tf.add_n([x, shortcut])
        return outputs


    @classmethod
    def separable_block(cls, inputs, filters, kernel_size=3, strides=1, rate=1,
                        depth_activation=False, name='separable_block', **kwargs):
        """ Separable convolution, followed by pointwise convolution with batch-normalization in-between. """
        layout = 'nacna' if depth_activation else 'ncn'
        data_format = kwargs.get('data_format')

        with tf.variable_scope(name):
            x = cls._depthwise_conv(inputs, kernel_size=kernel_size, strides=strides, dilation_rate=rate,
                                    data_format=data_format, padding='same', name='depthwise')

            x = conv_block(x, layout, filters=filters, kernel_size=1, **kwargs)
        return x


    @classmethod
    def _depthwise_conv(cls, inputs, depth_multiplier=1, name='depthwise_conv', **kwargs):
        data_format = kwargs.get('data_format')
        filters = cls.num_channels(inputs, data_format)
        depthwise_conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                                         depth_multiplier=depth_multiplier,
                                                         name='dwc', **kwargs)(inputs)
        return depthwise_conv


class Xception41(Xception):
    """ Xception-41 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] = dict(num_stages=3,
                                    filters=[[128]*3,
                                             [256]*3,
                                             [728]*3])

        config['body/middle'] = dict(num_stages=8,
                                     filters=[[728]*3]*8)

        config['body/exit'] = dict(num_stages=2, strides=[2, 1],
                                   filters=[[728, 1024, 1024],
                                            [1536, 1536, 2048]])
        return config


class Xception64(Xception):
    """ Xception-64 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] = dict(num_stages=3,
                                    filters=[[128]*3,
                                             [256]*3,
                                             [728]*3])

        config['body/middle'] = dict(num_stages=16,
                                     filters=[[728]*3]*16)

        config['body/exit'] = dict(num_stages=2, strides=[2, 1],
                                   filters=[[728, 1024, 1024],
                                            [1536, 1536, 2048]])
        return config


class XceptionS(Xception):
    """ Small version of Xception architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] = dict(num_stages=2,
                                    filters=[[6]*3,
                                             [12]*3,])
        config['body/middle'] = dict(num_stages=2,
                                     filters=[[12]*3]*2)
        config['body/exit'] = dict(num_stages=2, strides=[2, 1],
                                   filters=[[12]*3,
                                            [15]*3,])
        return config
