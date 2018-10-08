"""  Encoder-decoder """
import torch
import torch.nn as nn

from .layers import ConvBlock
from . import TorchModel
from .utils import get_shape
from .resnet import ResNet18


class EncoderDecoder(TorchModel):
    """ Encoder-decoder architecture

    **Configuration**

    inputs : dict
        dict with 'images' (see :meth:`~.TorchModel._make_inputs`)

    body : dict
        encoder : dict
            base_class : TorchModel
                a model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs
            other args
                parameters for base class ``make_encoder`` method

        embedding : dict
            :class:`~.ConvBlock` parameters for the bottom block

        decoder : dict
            num_stages : int
                number of upsampling blocks
            factor : int or list of int
                if int, the total upsampling factor for all blocks combined.
                if list, upsampling factors for each block.
            layout : str
                upsampling method (see :func:`~.layers.upsample`)

            other :func:`~.layers.upsample` parameters.

    Examples
    --------
    Use ResNet18 as an encoder (which by default downsamples the image with a factor of 8),
    create an embedding that contains 16 channels,
    and build a decoder with 3 upsampling stages to scale the embedding 8 times with transposed convolutions::

        config = {
            'inputs': dict(images={'shape': B('image_shape'), 'name': 'targets'}),
            'initial_block/inputs': 'images',
            'encoder/base_class': ResNet18,
            'embedding/filters': 16,
            'decoder': dict(num_stages=3, factor=8, layout='tna')
        }
    """
    @classmethod
    def default_config(cls):
        config = TorchModel.default_config()
        config['body/encoder'] = dict(base_class=ResNet18)
        config['body/decoder'] = dict(layout='tna', factor=8, num_stages=3)
        config['body/embedding'] = dict(layout='cna', filters=8)
        config['loss'] = 'mse'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head/targets'] = self.targets
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and decoder

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in decoder blocks
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        decoder = kwargs.pop('decoder')
        embedding = kwargs.pop('embedding')

        with tf.variable_scope(name):
            encoder_outputs = cls.encoder(inputs, **encoder, **kwargs)

            x = cls.embedding(encoder_outputs[-1], **embedding, **kwargs)
            if x != encoder_outputs[-1]:
                encoder_outputs += [x]

            x = cls.decoder(encoder_outputs, **decoder, **kwargs)
        return x

    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        """ Linear convolutions with kernel 1 """
        with tf.variable_scope(name):
            x = cls.crop(inputs, targets, kwargs['data_format'])
            channels = cls.num_channels(targets)
            x = conv_block(x, layout='c', filters=channels, kernel_size=1, **kwargs)
        return x

    @classmethod
    def encoder(cls, inputs, base_class, name='encoder', **kwargs):
        """ Create encoder from a base_class model

        Parameters
        ----------
        inputs
            input tensor
        base_class : TorchModel
            a model class (default=ResNet101).
            Should implement ``make_encoder`` method.
        name : str
            scope name
        kwargs : dict
            parameters for ``make_encoder`` method

        Returns
        -------
        list of tf.Tensor
        """
        x = base_class.make_encoder(inputs, name=name, **kwargs)
        return x

    @classmethod
    def embedding(cls, inputs, name='embedding', **kwargs):
        """ Create embedding from inputs tensor

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        if kwargs.get('layout') is not None:
            x = ConvBlock(inputs, name=name, **kwargs)
        else:
            x = inputs
        return x

    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        """ Create decoder with a given number of upsampling stages

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        steps = kwargs.pop('num_stages', len(inputs)-1)
        factor = kwargs.pop('factor')

        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor] * steps
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        with tf.variable_scope(name):
            x = inputs[-1]
            for i in range(steps):
                x = cls.upsample(x, factor=factor[i], name='decoder-'+str(i), **kwargs)
        return x


class EncoderDecoderBody(nn.Module):
    """ A sequence of encoder and decoder blocks with skip connections """
    def __init__(self, encoders, decoders, skip=True):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.skip = skip
        self.output_shape = self.decoders[-1].output_shape

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-i-2] if self.skip else None
            x = decoder(x, skip=skip)

        return x
