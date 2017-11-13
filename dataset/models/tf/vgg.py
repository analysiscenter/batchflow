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
    https://arxiv.org/abs/1409.1556 (K.Simonyan et al, 2014)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    arch : list of tuples
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
        config['input_block']['inputs'] = self.inputs['images']
        config['body']['arch'] = self.get_from_config('arch')
        config['head']['units'] = self.get_from_config('head/units', [100, 100])
        config['head']['num_classes'] = self.num_classes('labels')

        return config


    @classmethod
    def body(cls, inputs, arch=None, name='body', **kwargs):
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
        if not isinstance(arch, (list, tuple)):
            raise TypeError("arch must be list or tuple, but {} was given.".format(type(arch)))

        x = inputs
        with tf.variable_scope(name):
            for i, block_cfg in enumerate(arch):
                x = cls.block(x, *block_cfg, name='block-%d' % i, **kwargs)
        return x

    @staticmethod
    def block(inputs, depth_3, depth_1, filters, downscale, name='block', **kwargs):
        """ A sequence of 3x3 and 1x1 convolutions followed by pooling

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        depth_3 : int
            the number of convolution layers with 3x3 kernel
        depth_1 : int
            the number of convolution layers with 1x1 kernel
        filters : int
            the number of filters in each convolution layer
        downscale : bool
            whether to decrease spatial dimension at the end of the block

        Returns
        -------
        tf.Tensor
        """
        layout = kwargs.pop('layout', 'cna')
        layout = layout * (depth_3 + depth_1) + 'p' * downscale
        kernels = [3] * depth_3 + [1] * depth_1
        with tf.variable_scope(name):
            x = conv_block(inputs, filters, kernels, layout=layout, name='conv', **kwargs)
            x = tf.identity(x, name='output')
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

        x = conv_block(inputs, units=units, layout=layout, **kwargs)
        return x


class VGG16(VGG):
    """ VGG16 network """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body']['arch'] = _VGG16_ARCH
        print(config)
        return config

    @classmethod
    def body(cls, inputs, arch=None, name='body', **kwargs):
        """ VGG16 body layers """
        # if body is called independently, then arch might not be set
        arch = _VGG16_ARCH if arch is None else arch
        return VGG.body(inputs, arch, name=name, **kwargs)

class VGG19(VGG):
    """ VGG19 network """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body']['arch'] = _VGG19_ARCH
        return config

    @classmethod
    def body(cls, inputs, arch=None, name='body', **kwargs):
        """ VGG19 body layers """
        # if body is called independently, then arch might not be set
        arch = _VGG19_ARCH if arch is None else arch
        return VGG.body(inputs, arch, name=name, **kwargs)


class VGG7(VGG):
    """ VGG7 network """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['body']['arch'] = _VGG7_ARCH
        return config

    @classmethod
    def body(cls, inputs, arch=None, name='body', **kwargs):
        """ VGG7 body layers """
        # if body is called independently, then arch might not be set
        arch = _VGG7_ARCH if arch is None else arch
        return VGG.body(inputs, arch, name=name, **kwargs)
