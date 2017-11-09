"""Contains class for VGG"""
import tensorflow as tf

from . import TFModel
from .layers import conv_block, global_average_pooling

_ARCH = {'VGG16': [(2, 0, 64),
                   (2, 0, 128),
                   (2, 1, 256),
                   (2, 1, 512),
                   (2, 1, 512)],
         'VGG19': [(2, 0, 64),
                   (2, 0, 128),
                   (4, 0, 256),
                   (4, 0, 512),
                   (4, 0, 512)],
         'VGG7': [(2, 0, 64),
                  (2, 0, 128),
                  (2, 1, 256)]}

class VGG(TFModel):
    """VGG as TFModel
    https://arxiv.org/abs/1409.1556 (K.Simonyan et al, 2014)

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    batch_norm : bool
        if True enable batch normalization layers
    dilation_rate : int
        dilation rate for convolutional layers (1 by default)
    arch : str or list of tuples
        if str, it is 'VGG16' (by default), 'VGG19', 'VGG7'
        if list, each tuple must have the following components^
        tuple[0] : int
            number of convolution layers with 3x3 kernel
        tuple[1] : int
            number of convolution layers with 1x1 kernel
        tuple[2] : bool
            number of filters.
    """

    def _build(self):
        """
        Builds a VGG model.
        """
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        enable_batch_norm = self.get_from_config('batch_norm', True)
        arch = self.get_from_config('arch', 'VGG16')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'training': self.is_training}
        if enable_batch_norm:
            kwargs['batch_norm'] = batch_norm

        net = self.body(dim, inputs['images'], arch, **kwargs)
        net = self.head(dim, net, n_classes, data_format=data_format, is_training=self.is_training)

        logits = tf.identity(net, name='predictions')
        pred_proba = tf.nn.softmax(logits, name='predicted_prob')
        pred_labels = tf.argmax(pred_proba, axis=-1, name='predicted_labels')
        true_labels = tf.argmax(inputs['labels'], axis=-1, name='true_labels')
        equality = tf.equal(pred_labels, true_labels)
        equality = tf.cast(equality, dtype=tf.float32)
        tf.reduce_mean(equality, name='accuracy')

    @staticmethod
    def head(dim, input_tensor, num_classes, head_type='dense', data_format='channels_last', is_training=True,\
             dropout_rate=0):
        """Head for classification
        """
        with tf.variable_scope('head'):
            if head_type == 'dense':
                net = global_average_pooling(dim=dim, inputs=input_tensor, data_format=data_format)
                net = tf.layers.dropout(net, dropout_rate, training=is_training)
                net = tf.layers.dense(net, num_classes)
            elif head_type == 'conv':
                net = conv_block(dim=dim, inputs=input_tensor, filters=num_classes, kernel_size=1,\
                                     layout='c', name='conv_1', data_format=data_format)
                net = global_average_pooling(dim=dim, inputs=input_tensor, data_format=data_format)
            else:
                raise ValueError("Head_type should be dense or conv, but given %d" % head_type)
        return net

    @staticmethod
    def block(dim, inputs, depth_3, depth_1, filters, name='block', **kwargs):
        """VGG block.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        depth_3 : int
            the number of convolution layers with 3x3 kernel
        depth_1 : int
            the number of convolution layers with 1x1 kernel
        filters : int

        Return
        ------
        outp : tf.Tensor
        """
        enable_batch_norm = 'batch_norm' in kwargs
        net = inputs
        with tf.variable_scope(name):
            layout = 'cna' if enable_batch_norm else 'ca'
            layout = layout * (depth_3 + depth_1) + 'p'
            kernels = [3] * depth_3 + [1] * depth_1
            net = conv_block(dim, net, filters, kernels, layout, **kwargs)
            net = tf.identity(net, name='output')
        return net

    @staticmethod
    def body(dim, inputs, arch, **kwargs):
        """VGG body.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels
        inputs : tf.Tensor
        arch : str or list of tuples

        Return
        ------
        outp : tf.Tensor
        """
        if isinstance(arch, list):
            pass
        elif isinstance(arch, str):
            arch = _ARCH[arch]
        else:
            raise TypeError("arch must be str or list but {} was given.".format(type(arch)))
        net = inputs
        with tf.variable_scope('body'):
            for i, block_cfg in enumerate(arch):
                net = VGG.block(dim, net, *block_cfg, 'block-'+str(i), **kwargs)
        return net

class VGG16(VGG):
    """
    Builds a VGG16 model.
    """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'VGG16'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inputs, *args, **kwargs):
        """VGG16 body.
        """
        _ = args
        return VGG.body(dim, inputs, 'VGG16', **kwargs)


class VGG19(VGG):
    """
    Builds a VGG19 model.
    """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'VGG19'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inputs, *args, **kwargs):
        """VGG19 body.
        """
        _ = args
        return VGG.body(dim, inputs, 'VGG19', **kwargs)

class VGG7(VGG):
    """
    Builds a VGG7 model.
    """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'VGG7'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inputs, *args, **kwargs):
        """VGG7 body.
        """
        _ = args
        return VGG.body(dim, inputs, 'VGG7', **kwargs)
