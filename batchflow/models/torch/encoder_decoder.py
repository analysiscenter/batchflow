"""  Encoder-decoder """
import torch
import torch.nn as nn

from .layers import ConvBlock
from .layers import Upsample
from . import TorchModel
from .resnet import ResNet18
from .utils import get_shape
from ..utils import unpack_args


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
        config['common/conv/padding'] = 'same'
        config['body/encoder'] = dict(base=None, num_stages=None)
        config['body/encoder/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder/blocks'] = dict(base=cls.block)

        config['body/embedding'] = dict(base=cls.block)

        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None)
        config['body/decoder/upsample'] = dict(layout='tna')
        config['body/decoder/blocks'] = dict(base=cls.block, combine_op='softsum')
        config['head'] = dict(layout='c', kernel_size=1)
        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Create encoder, embedding and decoder

        Parameters
        ----------
        inputs
            input tensor
        filters : tuple of int
            number of filters in decoder blocks

        Returns
        -------
        nn.Module
        """
        kwargs = cls.get_defaults('body', kwargs)
        encoder = kwargs.pop('encoder')
        embeddings = kwargs.get('embedding')
        decoder = kwargs.pop('decoder')

        # print('BODY INPUTS', get_shape(inputs))
        # Encoder: transition down
        encoder_args = {**kwargs, **encoder}
        encoders = cls.encoder(inputs, **encoder_args)
        x = encoders[-1]

        # Bottleneck: working with compressed representation via multiple steps of processing
        embeddings = embeddings if isinstance(embeddings, (tuple, list)) else [embeddings]

        for i, embedding in enumerate(embeddings):
            embedding_args = {**kwargs, **embedding}
            x = cls.embedding(x, **embedding_args)
        encoders.append(x)

        # Decoder: transition up
        decoder_args = {**kwargs, **decoder}
        decoders = cls.decoder(encoders, **decoder_args)

        return EncoderDecoderBody(encoders, decoders, skip=True)


    @classmethod
    def head(cls, inputs, filters, **kwargs):
        """ Linear convolutions with kernel 1 """
        x = ConvBlock(inputs, filters=filters, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, name='block', **kwargs):
        """ Default conv block for processing tensors in encoder and decoder.
        By default makes 3x3 convolutions followed by batch-norm and activation.
        Does not change tensor shapes.
        """
        layout = kwargs.pop('layout', None) or 'cna'
        filters = kwargs.pop('filters', None) or cls.num_channels(inputs)
        return ConvBlock(inputs, layout=layout, filters=filters, name=name, **kwargs)


    @classmethod
    def encoder(cls, inputs, base_class=None, **kwargs):
        base_class = kwargs.pop('base')
        steps, downsample, block_args = cls.pop(['num_stages', 'downsample', 'blocks'], kwargs)

        if base_class is not None:
            encoder_outputs = base_class.make_encoder(inputs, **kwargs)

        else:
            base_block = block_args.get('base')
            x = inputs

            encoders = [x]
            for i in range(steps):
                # Preprocess tensor with given block
                x = EncoderBlock(x, i, steps, downsample, block_args, **kwargs)
                encoders.append(x)
        return encoders

    @classmethod
    def embedding(cls, inputs, **kwargs):
        """ Create embedding from inputs tensor

        Parameters
        ----------
        inputs
            input tensor

        Returns
        -------
        nn.Module
        """
        x = cls.block(inputs, **kwargs)
        return x

    @classmethod
    def decoder(cls, inputs, **kwargs):
        steps = kwargs.pop('num_stages') or len(inputs)-2
        skip, upsample, block_args = cls.pop(['skip', 'upsample', 'blocks'], kwargs)
        base_block = block_args.get('base')

        x = inputs[-1]
        decoders = []
        for i in range(steps):
            x = DecoderBlock(x, inputs[-i-3], i, steps, upsample, block_args, **kwargs)
            decoders.append(x)
        return decoders


class EncoderBlock(nn.Module):
    def __init__(self, inputs, i, steps, downsample, block_args, **kwargs):
        super().__init__()
        ifilters = list(get_shape(inputs))[1]
        self.downsample = ConvBlock(inputs, filters=ifilters, **{**kwargs, **downsample})
        shape = list(get_shape(self.downsample))
        shape = tuple(shape)

        base_block = block_args.get('base')
        args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}
        self.encoder = base_block(shape, **args)
        self.output_shape = self.encoder.output_shape


    def forward(self, x):
        x = self.downsample(x)
        x = self.encoder(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, inputs, skip, i, steps, upsample, block_args, **kwargs):
        super().__init__()

        # Upsample: keep the same amount of filters
        ifilters = upsample.get('filters') or get_shape(inputs)[1]
        self.upsample = ConvBlock(inputs, filters=ifilters, **{**kwargs, **upsample})

        base_block = block_args.get('base')
        args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}

        shape = list(get_shape(self.upsample))
        shape[1] += get_shape(skip)[1]
        self.decoder = base_block(tuple(shape), **args)
        self.output_shape = self.decoder.output_shape


    def forward(self, x, skip):
        x = self.upsample(x)

        # Move to separate method `crop`
        x_shape = list(get_shape(x))
        skip_shape = list(get_shape(skip))
        if x_shape[2] > skip_shape[2]:
            shape = [slice(None, c) for c in skip.size()[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            x = x[shape]
        elif x_shape[2] < skip_shape[2]:
            background = torch.zeros(*x_shape[:2], *skip_shape[2:])
            background[:, :, :x_shape[2], :x_shape[3]] = x
            x = background
        x = torch.cat([skip, x], dim=1)

        x = self.decoder(x)
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
        skips = [x]
        for encoder in self.encoders[1:]:
            x = encoder(x)
            skips.append(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-i-3] if self.skip else None
            x = decoder(x, skip=skip)
        return x
