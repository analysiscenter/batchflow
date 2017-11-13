"""Contains class for LinkNet"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel


class LinkNet(TFModel):
    """ LinkNet
    https://arxiv.org/abs/1707.03718 (A.Chaurasia et al, 2017)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    in_filters : int
        number of filters in the first convolution block (64 by default)
    out_filters : int
        number of filters in the last convolution block (32 by default)
    num_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    """

    def _build_config(self, names=None):
        names = names if names else ['images', 'masks']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')

        in_filters = self.get_from_config('in_filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)
        out_filters = self.get_from_config('out_filters', 32)

        config['input_block']['filters'] = self.get_from_config('input_block/filters', in_filters)
        config['input_block']['inputs'] = self.inputs['images']

        layers_filters = 2 ** np.arange(num_blocks) * in_filters
        config['body']['filters'] = self.get_from_config('body/in_filters', layers_filters)

        config['head']['filters'] = self.get_from_config('head/filters', out_filters)
        config['head']['num_classes'] = self.num_classes('masks')

        return config


    @classmethod
    def input_block(cls, inputs, filters, name='input_block', **kwargs):
        """ 7x7 convolution with stride=2 and 3x3 max pooling with stride=2

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return conv_block(inputs, filters, 7, layout='cpna', name=name, strides=2, pool_size=3, **kwargs)

    @classmethod
    def body(cls, inputs, filters, name='body', **kwargs):
        """ LinkNet body

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = inputs
            encoder_outputs = []
            for i, ifilters in enumerate(filters):
                x = cls.downsampling_block(x, ifilters, 'downsampling-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1][1:]):
                x = cls.upsampling_block(x, ifilters, 'upsampling-'+str(i), **kwargs)
                x = cls.crop(x, encoder_outputs[-i-2], data_format=kwargs.get('data_format'))
                x = tf.add(x, encoder_outputs[-2-i])
            x = cls.upsampling_block(x, filters[0], 'upsampling-'+str(i+1), **kwargs)
            x = cls.crop(x, inputs, data_format=kwargs.get('data_format'))

        return x

    @classmethod
    def downsampling_block(cls, inputs, filters, name, **kwargs):
        """ Two ResNet blocks of two 3x3 convolution + shortcut

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        layout = 'cna'
        with tf.variable_scope(name):
            net = conv_block(inputs, filters, 3, 2*layout, 'conv-1', strides=[2, 1], **kwargs)
            shortcut = conv_block(inputs, filters, 1, layout, 'conv-2', strides=2, **kwargs)
            add = tf.add(net, shortcut, 'add-1')
            net = conv_block(add, filters, 3, 2*layout, 'conv-3', **kwargs)
            output = tf.add(net, add, 'add-2')
        return output

    @classmethod
    def upsampling_block(cls, inputs, filters, name, **kwargs):
        """ 1x1 convolution, 3x3 transposed convolution with stride=2 and 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        num_filters = inputs.get_shape()[-1].value // 4
        return conv_block(inputs, [num_filters, num_filters, filters], [1, 3, 1],
                          layout='cna tna cna', name=name, strides=[1, 2, 1], **kwargs)

    @classmethod
    def head(cls, inputs, filters, num_classes, name='head', **kwargs):
        """ 3x3 transposed convolution, 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in 3x3 convolutions
        num_classes : int
            number of classes (and number of filters in the last convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """

        x = conv_block(inputs, [filters, filters, num_classes], [3, 3, 2], layout='tna cna t',
                       strides=[2, 1, 2], name=name, **kwargs)
        return x
