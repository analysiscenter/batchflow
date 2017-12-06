""" Wang F. at all. "`Residual Attention Network for Image Classification
<https://arxiv.org/abs/1704.06904>`_"
"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel
from .resnet import ResNet


class ResNetAttention(TFModel):
    """ Residual Attention Network

    **Configuration**

    inputs : dict
        dict with images and labels (see :meth:`._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[64, 128, 256, 512])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block
        config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))

        config['body']['layout'] = 'r2r1r0rrr'
        config['body']['filters'] = 2 ** np.array([0, 0, 1, 1, 2, 2, 3, 3, 3]) * filters
        config['body']['trunk'] = dict(bottleneck=True)
        config['body']['mask'] = dict(bottleneck=True, pool_size=3, pool_strides=2)

        config['head']['layout'] = 'Vf'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = self.num_classes('targets')
        config['head']['filters'] = self.num_classes('targets')
        return config

    @classmethod
    def trunk(cls, inputs, name='trunk', **kwargs):
        """ Trunk branch

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/trunk', **kwargs)
        x = ResNet.double_block(inputs, name=name, **kwargs)
        return x

    @classmethod
    def mask(cls, inputs, level=0, name='mask', **kwargs):
        """ Mask branch

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        level : int
            nested mask level
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/mask', **kwargs)
        with tf.variable_scope(name):
            inputs, resize_to = inputs
            x = conv_block(inputs, layout='p', name='pool', **kwargs)
            b = ResNet.block(x, name='resblock_1', **kwargs)
            c = ResNet.block(b, name='resblock_2', **kwargs)

            if level > 0:
                i = cls.mask((b, b), level=level-1, name='submask-%d' % level, **kwargs)
                c = ResNet.block(c + i, name='resblock_3', **kwargs)

            size = cls.spatial_shape(resize_to, data_format=kwargs.get('data_format'))
            x = tf.image.resize_bilinear(c, size=size, name='interpolation')
        return x

    @classmethod
    def attention(cls, inputs, level=0, name='attention', **kwargs):
        """ Attention module

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        level : int
            nested mask level
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = ResNet.block(inputs, name='initial', **kwargs)

            t = cls.trunk(x, **kwargs)
            m = cls.mask((x, t), level=level, **kwargs)

            x = conv_block(m, layout='nac nac', kernel_size=1, name='scale',
                           **{**kwargs, 'filters': kwargs['filters']*4})
            x = tf.sigmoid(x)
            x = (1 + x) * t

            x = ResNet.block(x, name='last', **kwargs)
        return x

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base blocks

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        layout, filters = cls.pop(['layout', 'filters'], kwargs)

        x = inputs
        with tf.variable_scope(name):
            for i, b in enumerate(layout):
                if b == 'r':
                    x = ResNet.block(x, filters=filters[i], name='resblock-%d' % i, **kwargs)
                else:
                    x = cls.attention(x, level=int(b), filters=filters[i], name='attention-%d' % i, **kwargs)
        return x
