""" Contains SqueezeNet """
import numpy as np
import tensorflow as tf

from . import TFModel
from .layers import conv_block




class SqueezeNet(TFModel):
    """ SqueezeNet neural network

    References
    ----------
    .. Iandola F. et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
       Arxiv.org, `<https://arxiv.org/abs/1602.07360>`_

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    layout : list of tuples
        A list should contain tuples of 4 ints:
        - number of convolution layers with 3x3 kernel
        - number of convolution layers with 1x1 kernel
        - number of filters in each layer
        - whether to downscale the image at the end of the block with max_pooling (2x2, stride=2)
    """

    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')

        config['input_block'] = {**dict(layout='cnap', filters=96, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2),
                                 **config['input_block']}
        config['input_block']['inputs'] = self.inputs['images']

        config['body']['layout'] = self.get_from_config('layout', 'fffmffffmf')

        num_blocks = len(config['body']['layout'])
        layers_filters = self.get_from_config('filters', 16) * 2 ** np.arange(num_blocks//2)
        layers_filters = np.repeat(layers_filters, 2)[:num_blocks].copy()
        print(layers_filters)
        config['body']['filters'] = self.get_from_config('body/filters', layers_filters)

        config['head'] = {**dict(layout='dcnaV', filters=self.num_classes('labels'),
                                 kernel_size=1, strides=1, dropout_rate=.5),
                          **config['head']}
        return config

    @classmethod
    def body(cls, inputs, filters, layout='', name='body', **kwargs):
        """ Create base VGG layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of block types
            - f : fire
            - m : max-pooling
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        x = inputs
        with tf.variable_scope(name):
            for i, block in enumerate(layout):
                if block == 'f':
                    x = cls.fire_block(x, filters=filters[i], name='fire-block-%d' % i, **kwargs)
                elif block == 'm':
                    x = conv_block(x, layout='p', name='max-pool-%d' % i, **kwargs)
        return x

    @classmethod
    def fire_block(cls, inputs, filters, layout='cna', name='fire-block', **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            the number of filters in each convolution layer

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = conv_block(inputs, filters, 1, layout=layout, name='squeeze-1x1', **kwargs)

            exp1 = conv_block(x, filters*4, 1, layout=layout, name='expand-1x1', **kwargs)
            exp3 = conv_block(x, filters*4, 3, layout=layout, name='expand-3x3', **kwargs)

            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat([exp1, exp3], axis=axis)
        return x
