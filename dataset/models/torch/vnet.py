"""  Milletari F. et al "`V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
<https://arxiv.org/abs/1606.04797>`_"
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ... import is_best_practice
from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape, get_num_dims, get_num_channels
from .resnet import ResNet


class VNet(TorchModel):
    """ VNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling blocks (default=5)

        filters : list of int
            number of filters in each block (default=[16, 32, 64, 128, 256])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()

        filters = 16   # number of filters in the first block
        config['body/layout'] = ['cna', 'cna'*2] + ['cna'*3] * 3
        num_blocks = len(config['body/layout'])
        config['body/filters'] = (2 ** np.arange(num_blocks) * filters).tolist()
        config['body/kernel_size'] = 5
        config['body/upsample'] = dict(layout='tna', factor=2)
        config['head'] = dict(layout='c', kernel_size=1)

        config['loss'] = 'ce'
        if is_best_practice('optimizer'):
            config['optimizer'] = 'Adam'
        else:
            config['optimizer'] = ('SGD', dict(lr=1e-4, momentum=.99))
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, filters = cls.pop(['layout', 'filters'], kwargs)

        x, inputs = inputs, None
        encoder_outputs = []
        for i, ifilters in enumerate(filters):
            x = cls.encoder_block(x, layout=layout[i], filters=ifilters, downsample=i > 0, **kwargs)
            encoder_outputs.append(x)

        for i, ifilters in enumerate(filters[-2::-1]):
            x = cls.decoder_block((x, encoder_outputs[-i-2]), layout=layout[-i-1], filters=ifilters*2, **kwargs)

        return x

    @classmethod
    def encoder_block(cls, inputs, downsample=True, **kwargs):
        """ 5x5x5 convolutions and 2x2x2 max pooling with stride 2

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, kernel_size = cls.pop(['layout', 'kernel_size'], kwargs)

        x, inputs = inputs, None
        if downsample:
            x = ConvBlock(x, layout='cna', kernel_size=2, strides=2, **kwargs)
        x = ResNet.block(x, layout=layout, kernel_size=kernel_size, downsample=False, **kwargs)

        return x

    @classmethod
    def decoder_block(cls, inputs, **kwargs):
        """ 2x2x2 transposed convolution + 5x5x5 convolutions

        Parameters
        ----------
        inputs
            input tensor
        name : str
            scope name

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        layout, filters, kernel_size = cls.pop(['layout', 'filters', 'kernel_size'], kwargs)
        upsample_args = cls.pop('upsample', kwargs)

        x, skip = inputs
        inputs = None
        x = cls.upsample(x, filters=filters, name='upsample', **upsample_args, **kwargs)
        x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
        axis = cls.channels_axis(kwargs.get('data_format'))
        x = tf.concat((skip, x), axis=axis)
        x = ResNet.block(x, layout=layout, filters=filters, kernel_size=kernel_size, downsample=0, **kwargs)

        return x

    @classmethod
    def head(cls, inputs, num_classes, **kwargs):
        """ 1x1 convolution

        Parameters
        ----------
        inputs
            input tensor
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('head', kwargs)
        x = ConvBlock(inputs, filters=num_classes, **kwargs)
        return x
