""" Contains class for ResNet """
import numpy as np
import tensorflow as tf

from . import TFModel
from .layers import conv_block


class ResNet(TFModel):
    """ The base ResNet model

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_

    ** Configuration **

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict

    """

    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        filters = self.get_from_config('filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)

        config['default']['data_format'] = self.data_format('images')

        config['input_block'] = {**dict(filters=64, kernel_size=7, layout='cnap',
                                        strides=2, pool_size=3, pool_strides=2),
                                 **config['input_block']}
        config['input_block']['inputs'] = self.inputs['images']

        body_filters = 2 ** np.arange(num_blocks) * filters
        config['body'] = {**dict(filters=body_filters, bottleneck_factor=4, se_block=False, ratio=16),
                          **config['body']}

        config['head']['num_classes'] = self.num_classes('labels')

        return config


    @classmethod
    def body(cls, inputs, filters, num_blocks, bottleneck, bottleneck_factor, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : list of int
            number of filters in each block group
        num_blocks : list of int
            number of blocks in each group
        bottleneck : bool
            whether to use a simple or bottleneck block
        bottleneck_factor : int
            filter number multiplier for a bottleneck block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = inputs
            for i, n_blocks in enumerate(num_blocks):
                with tf.variable_scope('block-%d' % i):
                    for block in range(n_blocks):
                        strides = 2 if i > 0 and block == 0 else 1
                        x = cls.block(x, filters=filters[i], bottleneck=bottleneck, bottleneck_factor=bottleneck_factor,
                                      name='layer-%d' % block, strides=strides, **kwargs)
            x = tf.identity(x, name='output')
        return x



    @classmethod
    def block(cls, inputs, bottleneck, name, **kwargs):
        """ A network building block

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor
        bottleneck : bool
            whether to use a simple or a bottleneck block
        name : str
            scope name

        Returns
        -------
        tf. tensor
            output tensor
        """
        if bottleneck:
            x = cls.bottleneck_block(inputs, name=name, **kwargs)
        else:
            x = cls.simple_block(inputs, name=name, **kwargs)
        return x

    @classmethod
    def simple_block(cls, inputs, filters, name, strides, se_block=False, ratio=16, **kwargs):
        """ A simple residual block

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters
        name : str
            scope name

        Returns
        -------
        tf. tensor
            output tensor
        """
        with tf.variable_scope(name):
            x = conv_block(inputs, filters, 3, layout='cnacn', name='conv', strides=[strides, 1], **kwargs)

            num_channels = cls.channels_shape(inputs, kwargs.get('data_format'))
            if num_channels != filters:
                shortcut = conv_block(inputs, filters, 1, 'c', name='shortcut', strides=strides, **kwargs)
            else:
                shortcut = inputs
            if se_block:
                x = cls.se_block(x, ratio, **kwargs)
            x = x + shortcut
        return x


    @classmethod
    def bottleneck_block(cls, inputs, filters, bottleneck_factor, name, strides, se_block=False, ratio=16, **kwargs):
        """ A stack of 1x1, 3x3, 1x1 convolutions

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in the first two convolutions
        out_filters : int
            number of filters in the last convolution

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            layout = kwargs.pop('layout', 'cna')
            out_filters = filters * bottleneck_factor
            x = conv_block(inputs, [filters, filters, out_filters], [1, 3, 1],
                           layout*3, name='conv', strides=[strides, 1, 1], **kwargs)

            num_channels = cls.channels_shape(inputs, kwargs.get('data_format'))
            if num_channels != out_filters:
                shortcut = conv_block(inputs, out_filters, 1, 'c', name='shortcut', strides=strides, **kwargs)
            else:
                shortcut = inputs
            if se_block:
                x = cls.se_block(x, ratio, **kwargs)
            x = x + shortcut

        return x


    @classmethod
    def se_block(cls, inputs, ratio, name='se', **kwargs):
        """
        Squeeze and excitation block

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor

        Returns
        -------
        tf. tensor
            output tensor
        """
        with tf.variable_scope(name):
            data_format = kwargs['data_format']
            in_filters = cls.channels_shape(inputs, data_format)
            x = conv_block(inputs, layout='Vfafa', units=[in_filters//ratio, in_filters],
                           activation=[tf.nn.relu, tf.nn.sigmoid], name='se', **kwargs)

            if data_format == 'channels_last':
                shape = [-1, 1, 1, in_filters]
            else:
                shape = [-1, in_filters, 1, 1]
            scale = tf.reshape(x, shape)
            x = inputs * scale
        return x

    @classmethod
    def head(cls, inputs, units=None, num_classes=None, name='head', **kwargs):
        """ A sequence of dense layers with the last one having ``num_classes`` units

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        units : int or tuple of int
            number of units in dense layers
        num_classes : int
            number of classes
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        if num_classes is None:
            raise ValueError('num_classes cannot be None')

        if units is None:
            units = []
        elif isinstance(units, int):
            units = [units]
        units = list(units) + [num_classes]

        layout = kwargs.pop('layout', 'f' * len(units))
        units = units[0] if len(units) == 1 else units

        x = conv_block(inputs, units=units, layout=layout, name=name, **kwargs)
        return x



class ResNet18(ResNet):
    """ The original ResNet-18 architecture

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body'].update({'num_blocks': [2, 2, 2, 2], 'bottleneck': False})
        return config


class ResNet34(ResNet):
    """ The original ResNet-34 architecture

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body'].update({'num_blocks': [3, 4, 6, 3], 'bottleneck': False})
        return config


class ResNet50(ResNet):
    """ The original ResNet-50 architecture

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body'].update({'num_blocks': [3, 4, 6, 3], 'bottleneck': True})
        return config


class ResNet101(ResNet):
    """ The original ResNet-101 architecture

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body'].update({'num_blocks': [3, 4, 23, 3], 'bottleneck': True})
        return config


class ResNet152(ResNet):
    """ The original ResNet-152 architecture

    References
    ----------
    .. Kaiming He et al. "Deep Residual Learning for Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1512.03385>`_
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body'].update({'num_blocks': [3, 8, 36, 3], 'bottleneck': True})
        return config
