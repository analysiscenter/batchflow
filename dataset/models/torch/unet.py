"""  Ronneberger O. et al "`U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>`_"
"""
import numpy as np
import torch
import torch.nn as nn

from ... import is_best_practice
from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape


class UNet(TorchModel):
    """ UNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'masks' (see :meth:`~.TorchModel._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[128, 256, 512, 1024])

        encoder : dict
            encoder block parameters (see :class:`.ConvBlock`)

        decoder : dict
            decoder block parameters (see :class:`.ConvBlock`)

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()

        config['common'] = dict(conv=dict(bias=False))
        config['body/num_blocks'] = 5
        config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        config['body/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder'] = dict(layout='cnacna', kernel_size=3)
        config['body/upsample'] = dict(layout='tna', kernel_size=2, strides=2)
        config['body/decoder'] = dict(layout='cnacna', kernel_size=3)
        config['head'] = dict(layout='c', kernel_size=1)

        config['loss'] = 'BCE'
        # The article does not specify the initial learning rate. 1e-4 was chosen arbitrarily.
        config['optimizer'] = ('SGD', dict(lr=1e-4, momentum=.99))

        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        if self.config.get('body/filters') is None:
            config['body/filters'] = (2 ** np.arange(config['body/num_blocks']) * 64).tolist()
        if config.get('head/num_classes') is None:
            config['head/num_classes'] = self.num_classes('targets')

        return config

    def body(self, inputs=None, **kwargs):
        """ A sequence of encoder and decoder blocks with skip connections

        Parameters
        ----------
        filters : tuple of int
            number of filters in downsampling blocks
        """
        kwargs = self.get_defaults('body', kwargs)
        filters = kwargs.pop('filters')
        downsample = kwargs.pop('downsample')
        encoder = kwargs.pop('encoder')
        upsample = kwargs.pop('upsample')
        decoder = kwargs.pop('decoder')

        encoders = []
        x = inputs
        for i, ifilters in enumerate(filters):
            down = downsample if i > 0 else None
            x = self.encoder_block(ifilters, down, encoder, inputs=x, **kwargs)
            encoders.append(x)

        decoders = []
        for i, ifilters in enumerate(filters[-2::-1]):
            skip = encoders[-i-2]
            x = self.decoder_block(ifilters, upsample, decoder, inputs=x, skip=skip, **kwargs)
            decoders.append(x)

        return UNetBody(encoders, decoders)

    @classmethod
    def encoder_block(cls, filters, downsample=None, encoder=None, inputs=None, **kwargs):
        """ 2x2 max pooling with stride 2 and two 3x3 convolutions

        Parameters
        ----------
        filters : int
            number of output filters
        downsample : bool
            whether to downsample the inputs (by default before convolutions)
        """
        if downsample:
            downsample = cls.get_defaults('body/downsample', downsample)
            down_block = ConvBlock(filters=filters, inputs=inputs, **{**kwargs, **downsample})
            inputs = down_block
        encoder = cls.get_defaults('body/encoder', encoder)
        enc_block = ConvBlock(filters=filters, inputs=inputs, **{**kwargs, **encoder})
        return nn.Sequential(down_block, enc_block) if downsample else enc_block

    @classmethod
    def decoder_block(cls, filters, upsample=None, decoder=None, inputs=None, skip=None, **kwargs):
        """ Takes inputs from a previous block and a skip connection

        Parameters
        ----------
        filters : int
            number of output filters
        upsample : dict
            parameters for upsample block
        decoder : dict
            parameters for decoder block
        inputs
            previous decoder block
        skip
            skip connection
        """
        upsample = cls.get_defaults('body/upsample', upsample)
        upsample = {**kwargs, **upsample}
        decoder = cls.get_defaults('body/decoder', decoder)
        decoder = {**kwargs, **decoder}
        return DecoderBlock(filters, upsample, decoder, inputs=inputs, skip=skip)

    @classmethod
    def head(cls, num_classes, inputs=None, **kwargs):
        """ Conv block with 1x1 convolution

        Parameters
        ----------
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        """
        kwargs = cls.get_defaults('head', kwargs)
        return ConvBlock(filters=num_classes, inputs=inputs, **kwargs)


class UNetBody(nn.Module):
    """ A sequence of encoder and decoder blocks with skip connections """
    def __init__(self, encoders, decoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.output_shape = self.decoders[-1].output_shape

    def forward(self, x):
        skip = []
        for encoder in self.encoders:
            x = encoder(x)
            skip.append(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip=skip[-i-2])

        return x

class DecoderBlock(nn.Module):
    """ An upsampling block aggregating a skip connection """
    def __init__(self, filters, upsample, decoder, inputs=None, **kwargs):
        super().__init__()
        self.upsample = ConvBlock(filters=filters, inputs=inputs, **upsample)
        shape = list(get_shape(self.upsample))
        shape[1] *= 2
        shape = tuple(shape)
        self.decoder = ConvBlock(filters=filters, inputs=shape, **decoder)
        self.output_shape = self.decoder.output_shape

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size() > skip.size():
            shape = [slice(None, c) for c in skip.size()[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            x = x[shape]

        x = torch.cat([skip, x], dim=1)
        x = self.decoder(x)
        return x
