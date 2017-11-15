"""Contains class for VGG"""
import tensorflow as tf

from . import TFModel
from .layers import conv_block


_VGG16_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (2, 1, 256, 1),
    (2, 1, 512, 1),
    (2, 1, 512, 1)
]

_VGG19_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (4, 0, 256, 1),
    (4, 0, 512, 1),
    (4, 0, 512, 1)
]

_VGG7_ARCH = [
    (2, 0, 64, 1),
    (2, 0, 128, 1),
    (2, 1, 256, 1)
]


class VGG(TFModel):
    """ Base VGG neural network

    References
    ----------
    .. Simonyan K., Zisserman A.. "Very Deep Convolutional Networks for Large-Scale Image Recognition"
       Arxiv.org, `<https://arxiv.org/abs/1409.1556>`_

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)

    body/arch : list of tuples
        A list should contain tuples of 4 ints:
        - number of convolution layers with 3x3 kernel
        - number of convolution layers with 1x1 kernel
        - number of filters in each layer
        - whether to downscale the image at the end of the block with max_pooling (2x2, stride=2)
    """
    @classmethod
    def _default_config(cls):
        config = TFModel._default_config()
        config['block'] = dict(layout='cna')
        config['head']['units'] = [4096, 4096]
        return config

    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['common']['data_format'] = self.data_format('images')
        config['input_block']['inputs'] = self.inputs['images']
        config['head']['units'] += [self.num_classes('labels')]
        if config['head'].get('layout') is None:
            config['head']['layout'] = 'f' * len(config['head']['units'])

        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create base VGG layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        arch : list of tuples
            (number of 3x3 conv, number of 1x1 conv, number of filters, whether to downscale)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        arch = kwargs.get('arch')
        if not isinstance(arch, (list, tuple)):
            raise TypeError("arch must be list or tuple, but {} was given.".format(type(arch)))

        x = inputs
        with tf.variable_scope(name):
            for i, block_cfg in enumerate(arch):
                x = cls.block(x, *block_cfg, name='block-%d' % i, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, depth3, depth1, filters, downscale, name='block', **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        depth3 : int
            the number of convolution layers with 3x3 kernel
        depth1 : int
            the number of convolution layers with 1x1 kernel
        filters : int
            the number of filters in each convolution layer
        downscale : bool
            whether to decrease spatial dimension at the end of the block

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('block', **kwargs)
        layout = kwargs.pop('layout') * (depth3 + depth1) + 'p' * downscale
        kernels = [3] * depth3 + [1] * depth1
        with tf.variable_scope(name):
            x = conv_block(inputs, filters, kernels, layout=layout, name='conv', **kwargs)
            x = tf.identity(x, name='output')
        return x


class VGG16(VGG):
    """ VGG16 network """
    @classmethod
    def _default_config(cls):
        config = VGG._default_config()
        config['body']['arch'] = _VGG16_ARCH
        return config


class VGG19(VGG):
    """ VGG19 network """
    @classmethod
    def _default_config(cls):
        config = VGG._default_config()
        config['body']['arch'] = _VGG19_ARCH
        return config


class VGG7(VGG):
    """ VGG7 network """
    @classmethod
    def _default_config(cls):
        config = VGG._default_config()
        config['body']['arch'] = _VGG7_ARCH
        return config
