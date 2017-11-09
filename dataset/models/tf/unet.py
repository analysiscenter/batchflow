"""Contains class for UNet"""
import tensorflow as tf
import numpy as np
from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """UNet as TFModel
    https://arxiv.org/abs/1505.04597 (O.Ronneberger et al, 2015)

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    batch_norm : bool
        if True enable batch normalization layers
    n_filters : int
        number of filters after the first convolution (64 by default)
    n_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    """

    def _build(self):
        """
        Builds a UNet model.
        """
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        enable_batch_norm = self.get_from_config('batch_norm', True)
        n_filters = self.get_from_config('n_filters', 64)
        n_blocks = self.get_from_config('n_blocks', 4)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'training': self.is_training}
        if enable_batch_norm:
            kwargs['batch_norm'] = batch_norm

        unet_filters = 2 ** np.arange(n_blocks) * n_filters * 2
        axis = dim + 1 if data_format == 'channels_last' else 1

        layout = 'cnacna' if enable_batch_norm else 'caca'
        net = conv_block(dim, inputs['images'], n_filters, 3, layout, 'input-block', **kwargs)
        encoder_outputs = [net]

        for i, filters in enumerate(unet_filters):
            net = self.downsampling_block(dim, net, filters, 'downsampling-'+str(i), **kwargs)
            encoder_outputs.append(net)

        net = conv_block(dim, net, unet_filters[-1]//2, 2, 't', 'middle-block', 2, **kwargs)

        for i, filters in enumerate(unet_filters[::-1][1:]):
            net = tf.concat([encoder_outputs[-i-2], net], axis=axis)
            net = self.upsampling_block(dim, net, filters, 'upsampling-'+str(i), **kwargs)

        net = conv_block(dim, net, [n_filters, n_filters, n_classes], [3, 3, 1], layout+'c',
                         'output-block', **kwargs)
        logits = tf.identity(net, 'predictions')
        tf.nn.softmax(logits, name='predicted_prob')

    @staticmethod
    def downsampling_block(dim, inputs, out_filters, name, **kwargs):
        """LinkNet encoder block.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        out_filters : int
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
            output = conv_block(dim, inputs, out_filters, 3, layout, name,
                                pool_size=2, **kwargs)
        return output

    @staticmethod
    def upsampling_block(dim, inputs, out_filters, name, **kwargs):
        """LinkNet encoder block.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        out_filters : int
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
            net = conv_block(dim, inputs, 2*out_filters, 3, layout, name+'-1', **kwargs)
            output = conv_block(dim, net, out_filters, 2, 't', name+'-2', 2, **kwargs)
        return output
