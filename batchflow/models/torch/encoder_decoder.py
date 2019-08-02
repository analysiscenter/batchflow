"""  Encoder-decoder """
import torch.nn as nn

from .layers import ConvBlock
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

        # Encoder: transition down
        encoder_args = {**kwargs, **encoder}
        encoders = cls.encoder(inputs, **encoder_args)
        x = encoders[-1]

        # Bottleneck: working with compressed representation via multiple steps of processing
        embeddings = embeddings if isinstance(embeddings, (tuple, list)) else [embeddings]

        embeddings = []
        for i, embedding in enumerate(embeddings):
            embedding_args = {**kwargs, **embedding}
            x = cls.embedding(x, **embedding_args)
            embeddings.append(x)

        encoders.append(x)

        # Decoder: transition up
        decoder_args = {**kwargs, **decoder}
        decoders = cls.decoder(encoders, **decoder_args)
        return EncoderDecoderBody(encoders, embeddings, decoders, skip=True)


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
        if kwargs.get('layout') is not None:
            x = ConvBlock(inputs, **kwargs)
        else:
            x = inputs
        return x

    @classmethod
    def decoder(cls, inputs, **kwargs):
        steps = kwargs.pop('num_stages') or len(inputs)-2
        skip, upsample, block_args = cls.pop(['skip', 'upsample', 'blocks'], kwargs)
        base_block = block_args.get('base')


        x = inputs[-1]
        decoders = []
        for i in range(steps):
            x = DecoderBlock(x, inputs[-i-3], i, upsample, block_args, **kwargs)
            decoders.append(x)
        return decoders


class EncoderBlock(nn.Module):
    def __init__(self, inputs, i, steps, downsample, block_args, **kwargs):
        super().__init__()
        dfilters = list(get_shape(inputs))[1]
        self.downsample = ConvBlock(inputs, filters=dfilters, **{**kwargs, **downsample})
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
    def __init__(self, inputs, skip, i, upsample, block_args, **kwargs):
        super().__init__()
        _ = skip
        ufilters = list(get_shape(inputs))[1]
        self.upsample = ConvBlock(inputs, filters=ufilters, **{**kwargs, **upsample})
        shape = list(get_shape(self.upsample))
        shape[1] = 10
        shape = tuple(shape)

        base_block = block_args.get('base')
        args = {**kwargs, **block_args, **unpack_args(block_args, i, 4)}
        self.decoder = base_block(shape, **args)
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


class EncoderDecoderBody(nn.Module):
    """ A sequence of encoder and decoder blocks with skip connections """
    def __init__(self, encoders, embeddings, decoders, skip=True):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.embeddings = nn.ModuleList(embeddings)
        self.decoders = nn.ModuleList(decoders)
        self.skip = skip
        self.output_shape = self.decoders[-1].output_shape

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        for embedding in self.embeddings:
            x = embedding(x)
        skips.append(x)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-i-3] if self.skip else None
            x = decoder(x, skip=skip)

        return x
