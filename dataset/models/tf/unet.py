"""Contains class for UNet"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """ UNet

    References
    ----------
    .. Ronneberger O. et al ""
       Arxiv.org `<https://arxiv.org/abs/1505.04597>`_

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    filters : int
        number of filters in the first and the last convolution (64 by default)
    num_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    """

    def _build_config(self, names=None):
        names = names if names else ['images', 'masks']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')

        filters = self.get_from_config('filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)

        config['input_block']['filters'] = self.get_from_config('input_block/filters', filters)
        config['input_block']['inputs'] = self.inputs['images']

        layers_filters = 2 ** np.arange(num_blocks) * filters * 2
        config['body']['filters'] = self.get_from_config('body/filters', layers_filters)

        config['head']['filters'] = self.get_from_config('head/filters', filters)
        config['head']['num_classes'] = self.num_classes('masks')

        return config


    @classmethod
    def input_block(cls, inputs, filters, name='input_block', **kwargs):
        """ 3x3 convolution

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
        return conv_block(inputs, filters, 3, layout='cnacna', name=name, **kwargs)


    @classmethod
    def body(cls, inputs, filters, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
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
            encoder_outputs = [x]
            for i, ifilters in enumerate(filters):
                x = cls.downsampling_block(x, ifilters, name='downsampling-'+str(i), **kwargs)
                encoder_outputs.append(x)

            for i, ifilters in enumerate(filters[::-1]):
                x = cls.upsampling_block((x, encoder_outputs[-i-2]), ifilters//2, name='upsampling-'+str(i), **kwargs)

        return x

    @classmethod
    def downsampling_block(cls, inputs, filters, name, **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

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
        x = conv_block(inputs, filters, 3, layout='pcnacna', name=name, pool_size=2, pool_strides=2, **kwargs)
        return x

    @classmethod
    def upsampling_block(cls, inputs, filters, name, **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution

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
        with tf.variable_scope(name):
            x, skip = inputs
            x = conv_block(x, filters, 2, layout='t', name='upsample', strides=2, **kwargs)
            x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
            axis = -1 if kwargs.get('data_format') == 'channels_last' else 1
            x = tf.concat((skip, x), axis=axis)
            x = conv_block(x, filters, 3, layout='cnacna', name='conv', **kwargs)
        return x

    @classmethod
    def head(cls, inputs, filters, num_classes, name='head', **kwargs):
        """ Two 3x3 convolutions and 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in 3x3 convolutions
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return conv_block(inputs, [filters, filters, num_classes], [3, 3, 1], layout='cnacnac', name=name, **kwargs)
