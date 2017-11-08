"""UNet"""
import tensorflow as tf
import numpy as np
from .layers import conv_block
from . import TFModel

class UNet(TFModel):
    """UNet as TFModel

    **Configuration**
    -----------------
    inputs : dict
        input config with keys 'images' and 'masks'
    b_norm : bool
        if True enable batch normalization layers
    """
    def _build(self):
        """Builds a UNet model."""
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        n_filters = self.get_from_config('n_filters', 64)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1,
                      'training': self.is_training}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        unet_filters = 2 ** np.arange(4) * n_filters * 2
        axis = dim + 1 if data_format == 'channels_last' else 1

        layout = 'cnacna' if b_norm else 'caca'
        net = conv_block(dim, inputs['images'], n_filters, 3, layout, 'input-block', **kwargs)
        encoder_outputs = [net]

        for i, filters in enumerate(unet_filters):
            net = self.downsampling_block(dim, net, filters, 'downsampling-'+str(i), b_norm, **kwargs)
            encoder_outputs.append(net)

        net = conv_block(dim, net, unet_filters[-1]//2, 2, 't', 'middle-block', 2, **kwargs)

        for i, filters in enumerate(unet_filters[-2:-5:-1]):
            net = tf.concat([encoder_outputs[-i-2], net], axis=axis)
            net = self.upsampling_block(dim, net, filters, 'upsampling-'+str(i), b_norm, **kwargs)


        net = conv_block(dim, net, unet_filters[0], 3, layout, 'output-block-1', **kwargs)
        net = conv_block(dim, net, n_classes, 1, layout, 'output-block-2', **kwargs)

        logits = tf.identity(net, 'predictions')
        tf.nn.softmax(logits, name='predicted_prob')

    @staticmethod
    def downsampling_block(dim, inputs, out_filters, name, b_norm, **kwargs):
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

        b_norm : bool
            if True enable batch normalization

        Return
        ------
        outp : tf.Tensor
        """
        layout = 'pcnacna' if b_norm else 'pcaca'
        with tf.variable_scope(name): # pylint: disable=not-context-manager
            outp = conv_block(dim, inputs, out_filters, 3, layout, name,
                              pool_size=2, **kwargs)
        return outp

    @staticmethod
    def upsampling_block(dim, inputs, out_filters, name, b_norm, **kwargs):
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

        b_norm : bool
            if True enable batch normalization

        Return
        ------
        outp : tf.Tensor
        """
        layout = 'cnacna' if b_norm else 'caca'
        with tf.variable_scope(name): # pylint: disable=not-context-manager
            net = conv_block(dim, inputs, 2*out_filters, 3, layout, name+'-1', **kwargs)
            outp = conv_block(dim, net, out_filters, 2, 't', name+'-2', 2, **kwargs)
        return outp
