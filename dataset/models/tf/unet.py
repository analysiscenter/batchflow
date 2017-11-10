"""Contains class for UNet"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """ UNet
    https://arxiv.org/abs/1505.04597 (O.Ronneberger et al, 2015)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    batch_norm : None or dict
        parameters for batch normalization layers.
        If None, remove batch norm layers whatsoever.
        Default is ``{'momentum': 0.1}``.
    filters : int
        number of filters after the first convolution (64 by default)
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

        kwargs = {'data_format': data_format, 'training': self.is_training}
        if batch_norm:
            kwargs['batch_norm'] = batch_norm

        x = self.input_block(dim, inputs['images'], filters, **kwargs)
        layers_filters = 2 ** np.arange(num_blocks) * filters * 2
        x = self.body(dim, x, layers_filters, **kwargs)
        output = self.head(dim, x, filters, num_classes, **kwargs)

        tf.nn.softmax(output, name='predicted_proba')

    @classmethod
    def body(cls, dim, inputs, filters, **kwargs):
        """ UNet body

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

        Return
        ------
        tf.Tensor
        """
        encoder_outputs = [inputs]
        for i, ifilters in enumerate(filters):
            x = cls.downsampling_block(dim, x, ifilters, 'downsampling-'+str(i), **kwargs)
            encoder_outputs.append(x)

        x = conv_block(dim, x, filters[-1]//2, 2, 't', 'middle', strides=2, **kwargs)

        axis = -1 if kwargs['data_format'] == 'channels_last' else 1
        for i, ifilters in enumerate(filters[::-1][1:]):
            x = tf.concat([encoder_outputs[-i-2], x], axis=axis)
            x = cls.upsampling_block(dim, x, ifilters, 'upsampling-'+str(i), **kwargs)

        return x

    @classmethod
    def head(_, dim, inputs, filters, num_classes, **kwargs):
        """ Two 3x3 convolutions and 1x1 convolution

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in 3x3 convolutions
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        name : str
            scope name

        Return
        ------
        tf.Tensor
        """
        layout = 'cnacna' if 'batch_norm' in kwargs else 'caca'
        x = conv_block(dim, inputs, [filters, filters, num_classes], [3, 3, 1], layout+'c', 'output', **kwargs)
        x = tf.identity(x, 'predictions')
        return x

    @staticmethod
    def input_block(dim, inputs, filters, **kwargs):
        """ 3x3 convolution

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

        Return
        ------
        tf.Tensor
        """
        layout = 'cnacna' if 'batch_norm' in kwargs else 'caca'
        x = conv_block(dim, inputs, filters, 3, layout, 'input', **kwargs)
        return x

    @staticmethod
    def downsampling_block(dim, inputs, filters, name, **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

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

        Return
        ------
        tf.Tensor
        """
        layout = 'pcnacna' if 'batch_norm' in kwargs else 'pcaca'
        with tf.variable_scope(name):
            x = conv_block(dim, inputs, filters, 3, layout, name, pool_size=2, **kwargs)
        return x

    @staticmethod
    def upsampling_block(dim, inputs, filters, name, **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution

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

        Return
        ------
        tf.Tensor
        """
        layout = 'cnacna' if 'batch_norm' in kwargs else 'caca'
        with tf.variable_scope(name):
            x = conv_block(dim, inputs, 2*filters, 3, layout, 'conv', **kwargs)
            x = conv_block(dim, x, filters, 2, 't', 'transposed', strides=2, **kwargs)
        return x
