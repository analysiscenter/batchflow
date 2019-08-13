"""  Encoder-decoder """
import torch.nn as nn

from . import TorchModel
from .layers import ConvBlock, Upsample, Crop, Combine
from .utils import get_shape
from ..utils import unpack_args


class EncoderDecoder(TorchModel):
    """ Encoder-decoder architecture. Allows to combine features of different models,
    e.g. ResNet, in order to create new ones with just a few lines of code.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TorchModel._make_inputs`)

    body : dict
        encoder : dict
            base : TorchModel, optional
                Model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs.

            num_stages : int
                Number of downsampling stages.

            downsample : dict
                Parameters for downsampling (see :func:`~.layers.conv_block`)

            blocks : dict
                Parameters for pre-processing blocks:

                base : callable, optional
                    Tensor processing function. Default is :func:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.

            other args : dict
                Parameters for ``make_encoder`` method.

        embedding : dict or sequence of dicts
            base : callable, optional
                Tensor processing function. Default is :func:`~.layers.ConvBlock`.
            other args
                Parameters for the base block.

        decoder : dict
            num_stages : int
                Number of upsampling blocks.

            factor : int or list of int
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage.

            skip : bool
                Whether to concatenate upsampled tensor with stored pre-downsample encoding.
            upsample : dict
                Parameters for upsampling (see :func:`~.layers.Upsample`).

            blocks : dict
                Parameters for post-processing blocks:

                base : callable
                    Tensor processing function. Default is :func:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.

    Examples
    --------
    Preprocess input image with 7x7 convolutions, downsample it 4 times with ResNet blocks in between,
    use convolution block in the bottleneck, then restore original image size with subpixel convolutions and
    ResNeXt blocks in between:

    >>> config = {
            'inputs/images/shape': B('image_shape'),
            'inputs/masks/shape': B('mask_shape'),
            'initial_block': {'inputs': 'images',
                              'layout': 'cna', 'filters': 4, 'kernel_size': 7},
            'body/encoder': {'num_stages': 4,
                             'blocks': {'base': ResNet.block,
                                        'resnext': False,
                                        'filters': [8, 16, 32, 64]}},
            'body/embedding': {'filters': 128},
            'body/decoder': {'upsample': {'layout': 'X'},
                             'blocks': {'base': ResNet.block,
                                        'filters': [128, 64, 32, 16],
                                        'resnext': True}},
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
        config['body/decoder/blocks'] = dict(base=cls.block, combine_op='concat')
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
        """ Create encoder, embedding and decoder.

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

        for embedding in embeddings:
            embedding_args = {**kwargs, **embedding}
            x = cls.embedding(x, **embedding_args)
        encoders.append(x)

        # Decoder: transition up
        decoder_args = {**kwargs, **decoder}
        decoders = cls.decoder(encoders, **decoder_args)
        return EncoderDecoderBody(encoders, decoders, skip=decoder_args.get('skip'))

    @classmethod
    def head(cls, inputs, filters, **kwargs):
        """ Linear convolutions. """
        kwargs = cls.get_defaults('head', kwargs)
        x = super().head(inputs=inputs, filters=filters, **kwargs)

        if get_shape(x)[1] != filters:
            args = {**kwargs, **dict(layout='c', filters=filters, kernel_size=1)}
            x = ConvBlock(inputs, **args)
        return x

    @classmethod
    def block(cls, inputs, **kwargs):
        """ Default conv block for processing tensors. Makes 3x3 convolutions followed by batch-norm and activation.
        Does not change tensor shapes.
        """
        layout = kwargs.pop('layout', None) or 'cna'
        filters = kwargs.pop('filters', None) or get_shape(inputs)[1]
        return ConvBlock(inputs, layout=layout, filters=filters, **kwargs)


    @classmethod
    def encoder(cls, inputs, base_class=None, **kwargs):
        """ Create encoder either by using ``make_encoder`` of passed `base` model,
        or by combining building blocks, specified in `blocks/base`.

        Parameters
        ----------
        inputs
            Input tensor.

        base : TorchModel
            Model class. Should implement ``make_encoder`` method.

        name : str
            Scope name.

        num_stages : int
            Number of downsampling stages.

        blocks : dict
            Parameters for tensor processing before downsampling.

        downsample : dict
            Parameters for downsampling.

        kwargs : dict
            Parameters for ``make_encoder`` method.

        Returns
        -------
        list of nn.Modules
        """
        base_class = kwargs.pop('base')
        steps, downsample, block_args = cls.pop(['num_stages', 'downsample', 'blocks'], kwargs)

        if base_class is not None:
            encoder_outputs = base_class.make_encoder(inputs, **kwargs)
        else:
            x = inputs
            encoder_outputs = [x]

            for i in range(steps):
                d_args = {**kwargs, **downsample, **unpack_args(downsample, i, steps)}
                d_args['filters'] = d_args.get('filters') or get_shape(x)[1]
                b_args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}
                x = EncoderBlock(x, d_args, b_args, **kwargs)
                encoder_outputs.append(x)
        return encoder_outputs

    @classmethod
    def embedding(cls, inputs, **kwargs):
        """ Create embedding from inputs tensor.

        Parameters
        ----------
        inputs
            Input tensor.

        name : str
            Scope name.

        base : callable
            Tensor processing function. Default is :func:`~.layers.ConvBlock`.

        kwargs : dict
            Parameters for `base` block.

        Returns
        -------
        torch.nn.Module
        """
        base_block = kwargs.get('base', cls.block)
        return base_block(inputs, **kwargs)

    @classmethod
    def decoder(cls, inputs, **kwargs):
        """ Create decoder with a given number of upsampling stages.

        Parameters
        ----------
        inputs
            Input tensor.

        name : str
            Scope name.

        steps : int
            Number of upsampling stages. Defaults to the number of downsamplings.

        factor : int or list of ints
            If int, the total upsampling factor for all stages combined.
            If list, upsampling factors for each stage.s, then each entry is increase of size on i-th upsampling stage.

        skip : bool
            Whether to concatenate upsampled tensor with stored pre-downsample encoding.

        upsample : dict
            Parameters for upsampling.

        blocks : dict
            Parameters for post-processing blocks.

        kwargs : dict
            Parameters for :func:`~.layers.Upsample` method.

        Returns
        -------
        torch.nn.Module

        Raises
        ------
        TypeError
            If passed `factor` is not integer or list.
        """
        steps = kwargs.pop('num_stages') or len(inputs)-2
        factor = kwargs.pop('factor') or [2]*steps
        skip, upsample, block_args = cls.pop(['skip', 'upsample', 'blocks'], kwargs)

        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor] * steps
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        x = inputs[-1]
        decoders = []
        for i in range(steps):
            if factor[i] == 1:
                continue
            # Make upsample/block args, as well as prepare the skip connection if needed
            u_args = {**kwargs, **upsample, **unpack_args(upsample, i, steps)}
            u_args['filters'] = u_args.get('filters') or get_shape(x)[1]
            u_args['factor'] = u_args.get('factor') or factor[i]
            b_args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}
            skip_ = inputs[-i-3] if (skip and (i < len(inputs)-2)) else None

            x = DecoderBlock(x, skip_, u_args, b_args, **kwargs)
            decoders.append(x)
        return decoders



class EncoderDecoderBody(nn.Module):
    """ A sequence of encoder and decoder blocks with optional skip connections """
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
            skip = skips[-i-3] if (self.skip and (i < len(skips)-2)) else None
            x = decoder(x, skip=skip)
        return x


class EncoderBlock(nn.Module):
    """ Pass tensor through complex block, then downsample. """
    def __init__(self, inputs, d_args, b_args, **kwargs):
        _ = kwargs
        super().__init__()

        # Preprocess tensor with given block
        base_block = b_args.get('base')
        self.encoder = base_block(inputs, **b_args)

        # Downsampling
        if d_args.get('layout'):
            self.downsample = ConvBlock(get_shape(self.encoder), **d_args)
        else:
            self.downsample = self.encoder
        self.output_shape = self.downsample.output_shape

    def forward(self, x):
        x = self.encoder(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    """ Upsample tensor, then pass it through complex block and combine with skip if needed. """
    def __init__(self, inputs, skip, u_args, b_args, **kwargs):
        _ = kwargs
        super().__init__()

        # Upsample by a desired factor
        if u_args.get('layout'):
            self.upsample = Upsample(inputs=inputs, **u_args)
        else:
            self.upsample = inputs

        # Process tensor with block
        base_block = b_args.get('base')
        self.decoder = base_block(get_shape(self.upsample), **b_args)

        # Output shape: take skip into account
        shape = get_shape(self.decoder)
        if skip is not None:
            self.crop = Crop(shape, skip)
            self.combine = Combine([skip, get_shape(self.crop)], b_args.get('combine_op'))
            shape = self.combine.output_shape
        self.output_shape = shape

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.decoder(x)

        if skip is not None:
            x = self.crop(x, skip)
            x = self.combine([skip, x])
        return x
