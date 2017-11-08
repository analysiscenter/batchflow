"""LinkNet"""
import tensorflow as tf
from .layers import conv_block
from . import TFModel

class LinkNet(TFModel):
    """LinkNet as TFModel

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    b_norm : bool
        if True enable batch normalization layers
    """

    def _build(self):
        """Builds a LinkNet model."""
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1,
                      'training': self.is_training}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
            layout = 'cpna' if b_norm else 'cpa'

            net = conv_block(dim, inputs['images'], 64, 7, layout, 'input_conv', 2, pool_size=3, **kwargs)

            encoder_output = []

            for i, n_filters in enumerate([64, 128, 256, 512]):
                net = self.downsampling_block(dim, net, n_filters, 'downsampling-'+str(i), b_norm, **kwargs)
                encoder_output.append(net)

            for i, n_filters in enumerate([256, 128, 64]):
                net = self.upsampling_block(dim, net, n_filters, 'upsampling-'+str(i), b_norm, **kwargs)
                net = tf.add(net, encoder_output[-2-i])

            net = self.upsampling_block(dim, net, 64, 'upsampling-3', b_norm, **kwargs)

            layout = 'cna' if b_norm else 'ca'
            layout_transpose = 'tna' if b_norm else 'ta'

            net = conv_block(dim, net, 32, 3, layout_transpose, 'output_conv_1', 2, **kwargs)
            net = conv_block(dim, net, 32, 3, layout, 'output_conv_2', **kwargs)
            net = conv_block(dim, net, n_classes, 2, 't', 'output_conv_3', 2, **kwargs)

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
        with tf.variable_scope(name): # pylint: disable=not-context-manager
            layout = 'cna' if b_norm else 'ca'
            net = conv_block(dim, inputs, out_filters, 3, 2*layout, 'conv-1', strides=[2, 1], **kwargs)
            shortcut = conv_block(dim, inputs, out_filters, 1, layout, 'conv-2', 2, **kwargs)
            add = tf.add(net, shortcut, 'add-1')

            net = conv_block(dim, add, out_filters, 3, 2*layout, 'conv-3', **kwargs)
            outp = tf.add(net, add, 'add-2')
        return outp

    @staticmethod
    def upsampling_block(dim, inputs, out_filters, name, b_norm, **kwargs):
        """LinkNet decoder block.

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
        with tf.variable_scope(name): # pylint: disable=not-context-manager
            layout = 'cna' if b_norm else 'ca'
            layout_transpose = 'tna' if b_norm else 'ta'

            n_filters = inputs.get_shape()[-1].value // 4

            net = conv_block(dim, inputs, n_filters, 1, layout, 'conv-1', **kwargs)
            net = conv_block(dim, net, n_filters, 3, layout_transpose, 'conv-2', 2, **kwargs)
            outp = conv_block(dim, net, out_filters, 1, layout, 'conv-3', **kwargs)
            return outp
