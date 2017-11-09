"""Contains class for UNet"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """ UNet
    https://arxiv.org/abs/1505.04597 (O.Ronneberger et al, 2015)

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    batch_norm : bool
        if True enable batch normalization layers
    filters : int
        number of filters after the first convolution (64 by default)
    num_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    """

    def _build(self):
        """ Build a UNet model """
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        num_classes = self.num_classes('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        enable_batch_norm = self.get_from_config('batch_norm', True)
        filters = self.get_from_config('filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)

        conv = {'data_format': data_format}

        kwargs = {'conv': conv, 'training': self.is_training}
        if enable_batch_norm:
            kwargs['batch_norm'] = {'momentum': 0.1}

        unet_filters = 2 ** np.arange(num_blocks) * filters * 2
        axis = dim + 1 if data_format == 'channels_last' else 1

        layout = 'cnacna' if enable_batch_norm else 'caca'
        net = conv_block(dim, inputs['images'], filters, 3, layout, 'input-block', **kwargs)
        encoder_outputs = [net]

        for i, ifilters in enumerate(unet_filters):
            net = self.downsampling_block(dim, net, ifilters, 'downsampling-'+str(i), **kwargs)
            encoder_outputs.append(net)

        net = conv_block(dim, net, unet_filters[-1]//2, 2, 't', 'middle-block', 2, **kwargs)

        for i, ifilters in enumerate(unet_filters[::-1][1:]):
            net = tf.concat([encoder_outputs[-i-2], net], axis=axis)
            net = self.upsampling_block(dim, net, ifilters, 'upsampling-'+str(i), **kwargs)

        net = conv_block(dim, net, [filters, filters, num_classes], [3, 3, 1], layout+'c',
                         'output-block', **kwargs)
        logits = tf.identity(net, 'predictions')
        tf.nn.softmax(logits, name='predicted_proba')

    @staticmethod
    def downsampling_block(dim, inputs, filters, name, **kwargs):
        """LinkNet encoder block

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        filters : int
            number of output filters
        name : str
            tf.scope name

        Return
        ------
        outp : tf.Tensor
        """
        enable_batch_norm = 'batch_norm' in kwargs
        layout = 'pcnacna' if enable_batch_norm else 'pcaca'
        with tf.variable_scope(name):
            x = conv_block(dim, inputs, filters, 3, layout, name, pool_size=2, **kwargs)
        return x

    @staticmethod
    def upsampling_block(dim, inputs, filters, name, **kwargs):
        """LinkNet encoder block

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        filters : int
            number of output filters
        name : str
            tf.scope name

        Return
        ------
        outp : tf.Tensor
        """
        enable_batch_norm = 'batch_norm' in kwargs
        layout = 'cnacna' if enable_batch_norm else 'caca'
        with tf.variable_scope(name):
            x = conv_block(dim, inputs, 2*filters, 3, layout, name+'-1', **kwargs)
            x = conv_block(dim, x, filters, 2, 't', name+'-2', 2, **kwargs)
        return x
