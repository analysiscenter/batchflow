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
    batch_norm : None or dict
        parameters for batch normalization layers.
        If None, remove batch norm layers whatsoever.
        Default is ``{'momentum': 0.1}``.
    filters : int
        number of filters in the first convolution block (64 by default)
    num_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    """

    def _build(self):
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        num_classes = self.num_classes('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        batch_norm = self.get_from_config('batch_norm', {'momentum': 0.1})
        filters = self.get_from_config('filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)

        conv_block_config = self.get_from_config('conv_block', {})
        input_block_config = self.get_from_config('input_block', {'filters': filters})
        layers_filters = 2 ** np.arange(num_blocks) * filters
        body_config = self.get_from_config('body', {'filters': layers_filters})
        head_config = self.get_from_config('head', {'filters': 32})
        head_config['num_classes'] = num_classes

        kwargs = {'data_format': data_format, 'training': self.is_training, **conv_block_config}
        if batch_norm:
            kwargs['batch_norm'] = batch_norm

        with tf.variable_scope('LinkNet'):
            x = self.input_block(dim, inputs['images'], name='input', **{**kwargs, **input_block_config})
            x = self.body(dim, x, name='body', **{**kwargs, **body_config})
            output = self.head(dim, x, name='head', **{**kwargs, **head_config})

        logits = tf.identity(output, 'predictions')
        tf.nn.softmax(logits, name='predicted_proba')

    @classmethod
    def body(cls, dim, inputs, filters, name='body', **kwargs):
        """ LinkNet body

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in downsampling blocks
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
                x = cls.downsampling_block(dim, x, ifilters, 'downsampling-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1][1:]):
                x = cls.upsampling_block(dim, x, ifilters, 'upsampling-'+str(i), **kwargs)
                x = tf.add(x, encoder_outputs[-2-i])
            x = cls.upsampling_block(dim, x, filters[0], 'upsampling-'+str(i+1), **kwargs)

        return x

    @classmethod
    def head(cls, dim, inputs, filters, num_classes, name='head', **kwargs):
        """ 3x3 transposed convolution, 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
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
        layout = 'tna cna t' if 'batch_norm' in kwargs else 'ta ca t'
        with tf.variable_scope(name):
            x = conv_block(dim, inputs, [filters, filters, num_classes], [3, 3, 2], layout,
                           strides=[2, 1, 2], **kwargs)
        return x

    @staticmethod
    def input_block(dim, inputs, filters, name='input', **kwargs):
        """ 7x7 convolution with stride=2 and 3x3 max pooling with stride=2

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
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
        layout = 'cpna' if 'batch_norm' in kwargs else 'cpa'
        x = conv_block(dim, inputs, filters, 7, layout, name=name,
                       strides=2, pool_size=3, **kwargs)
        return x

    @staticmethod
    def downsampling_block(dim, inputs, filters, name, **kwargs):
        """ Two ResNet blocks of two 3x3 convolution + shortcut

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
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
        enable_batch_norm = 'batch_norm' in kwargs
        layout = 'cna' if enable_batch_norm else 'ca'
        with tf.variable_scope(name):
            net = conv_block(dim, inputs, filters, 3, 2*layout, 'conv-1', strides=[2, 1], **kwargs)
            shortcut = conv_block(dim, inputs, filters, 1, layout, 'conv-2', strides=2, **kwargs)
            add = tf.add(net, shortcut, 'add-1')

            net = conv_block(dim, add, filters, 3, 2*layout, 'conv-3', **kwargs)
            output = tf.add(net, add, 'add-2')
        return output

    @staticmethod
    def upsampling_block(dim, inputs, filters, name, **kwargs):
        """ 1x1 convolution, 3x3 transposed convolution with stride=2 and 1x1 convolution

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
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
        layout = 'cna tna cna' if 'batch_norm' in kwargs else 'ca ta ca'
        num_filters = inputs.get_shape()[-1].value // 4
        output = conv_block(dim, inputs, [num_filters, num_filters, filters], [1, 3, 1],
                            layout, name, strides=[1, 2, 1], **kwargs)
        return output
