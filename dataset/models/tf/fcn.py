"""Contains class for FCN"""
import tensorflow as tf

from .layers import conv_block
from . import TFModel, VGG16

class FCN(TFModel):
    """ Fully convolutional network (FCN)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016)

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    batch_norm : bool
        if True, enable batch normalization layers
    arch : str {'FCN32', 'FCN16', 'FCN8'}
        network architecture. Default is 'FCN32'.
    """

    def _build(self):
        """ Builds a FCN model """
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        num_classes = self.num_classes('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        enable_batch_norm = self.get_from_config('batch_norm', True)
        arch = self.get_from_config('arch', 'FCN32')

        conv = {'data_format': data_format, 'dilation_rate': self.get_from_config('dilation_rate', 1)}
        kwargs = {'conv': conv, 'training': self.is_training}
        if enable_batch_norm:
            kwargs['batch_norm'] = {'momentum': 0.1}

        x = VGG16.body(dim, inputs['images'], **kwargs)

        layout = 'cna' * 3 if enable_batch_norm else 'ca' * 3
        x = conv_block(dim, x, [100, 100, num_classes], [7, 1, 1], layout, 'conv-out', **kwargs)

        pool4 = tf.get_default_graph().get_tensor_by_name("body/block-3/output:0")
        pool3 = tf.get_default_graph().get_tensor_by_name("body/block-2/output:0")
        if arch == 'FCN32':
            x = conv_block(dim, x, num_classes, 64, 't', 'output', strides=32, **kwargs)
        else:
            conv7 = conv_block(dim, x, num_classes, 1, 't', 'conv7', strides=2, **kwargs)
            pool4 = conv_block(dim, pool4, num_classes, 1, 'c', 'pool4', strides=1, **kwargs)
            fcn16_sum = tf.add(conv7, pool4)
            if arch == 'FCN16':
                x = conv_block(dim, fcn16_sum, num_classes, 32, 't', 'output', strides=16, **kwargs)
            elif arch == 'FCN8':
                pool3 = conv_block(dim, pool3, num_classes, 1, 'c', 'pool3')
                fcn16_sum = conv_block(dim, fcn16_sum, num_classes, 1, 't', 'fcn16_sum', strides=2, **kwargs)
                fcn8_sum = tf.add(pool3, fcn16_sum)
                x = conv_block(dim, fcn8_sum, num_classes, 16, 't', 'output', strides=8, **kwargs)
            else:
                raise ValueError('Wrong value of arch, it must be "FCN32", "FCN16" or "FCN8".')

        tf.nn.softmax(tf.identity(x, 'predictions'), name='predicted_proba')

class FCN32(FCN):
    """  Fully convolutional network (FCN32)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016) """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'FCN32'
        super()._build(*args, **kwargs)

class FCN16(FCN):
    """  Fully convolutional network (FCN16)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016) """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'FCN16'
        super()._build(*args, **kwargs)

class FCN8(FCN):
    """  Fully convolutional network (FCN8)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016) """
    def _build(self, *args, **kwargs):
        self.config['arch'] = 'FCN8'
        super()._build(*args, **kwargs)
