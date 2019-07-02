"""
François Chollet. "`Xception: Deep Learning with Depthwise Separable Convolutions
<https://arxiv.org/abs/1610.02357>`_"
"""

import tensorflow as tf

from .layers import conv_block
from . import TFModel



class Xception(TFModel):
    """ Non-empty. """

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
    def body(cls, inputs, name='body', **kwargs):
        kwargs = cls.fill_params('body', **kwargs)
        entry = kwargs.pop('entry')
        middle = kwargs.pop('middle')
        exit = kwargs.pop('exit')

        with tf.variable_scope(name):
            x = inputs

            # Entry flow: downsample the inputs
            with tf.variable_scope('entry_flow'):
                entry_stages = entry.pop('num_stages', 0)
                entry_filters = entry.pop('filters')
                entry_args = {**kwargs, **entry}
                for i in range(entry_stages):
                    x = cls.block(x, filters=entry_filters[i],
                                  name='block-'+str(i), **entry_args)
                print('DONE ENTRY')

            # Entry flow: downsample the inputs
            with tf.variable_scope('middle_flow'):
                middle_stages = middle.pop('num_stages', 0)
                middle_filters = middle.pop('filters')
                middle_args = {**kwargs, **middle}
                for i in range(middle_stages):
                    x = cls.block(x, filters=middle_filters[i],
                                  name='block-'+str(i), **middle_args)
                print('DONE MIDDLE')

            # Entry flow: downsample the inputs
            with tf.variable_scope('exit_flow'):
                exit_stages = exit.pop('num_stages', 0)
                exit_filters = exit.pop('filters')
                exit_args = {**kwargs, **exit}
                for i in range(exit_stages):
                    x = cls.block(x, filters=exit_filters[i],
                                  name='block-'+str(i), **exit_args)
                print('DONE EXIT')
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

        # Parse shapes
        inputs_shape = inputs.get_shape().as_list()
        dim = inputs.shape.ndims - 2
        axis = -1 if data_format == 'channels_last' else 1
        size = [-1] * (dim + 2)
        size[axis] = 1
        channels_in = inputs_shape[axis]

        # Loop through feature maps
        depthwise_layers = []
        for channel in range(channels_in):
            start = [0] * (dim + 2)
            start[axis] = channel

            input_slice = tf.slice(inputs, start, size)
            _kwargs = {**kwargs, 'inputs': input_slice, 'filters': depth_multiplier, 'name': 'slice-%d' % channel}

            slice_conv = tf.layers.conv2d(**_kwargs)
            depthwise_layers.append(slice_conv)

        # Concatenate the per-channel convolutions along the channel dimension.
        depthwise_conv = tf.concat(depthwise_layers, axis=axis, name=name)
        return depthwise_conv



class XceptionS(Xception):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/entry'] = dict(num_stages=2,
                                    filters=[[6]*3,
                                             [12]*3,])
        config['body/middle'] = dict(num_stages=2,
                                     filters=[[12]*3]*2)
        config['body/exit'] = dict(num_stages=2,
                                   filters=[[12]*3,
                                            [15]*3,])
        return config


