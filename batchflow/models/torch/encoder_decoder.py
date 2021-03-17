""" Encoder, decoder, encoder-decoder architectures. """
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import TorchModel
from .utils import get_shape
from .layers import ConvBlock, Upsample, Combine
from .blocks import DefaultBlock
from ..utils import unpack_args



class EncoderModule(nn.ModuleDict):
    """ Encoder: create compressed representation of an input by reducing its spatial dimensions. """
    def __init__(self, inputs=None, return_all=True, **kwargs):
        super().__init__()
        self.return_all = return_all
        self._make_modules(inputs, **kwargs)

    def forward(self, x):
        outputs = []

        for letter, layer in zip(self.layout, self.values()):
            if letter in ['b', 'd', 'p']:
                x = layer(x)
            elif letter in ['s']:
                outputs.append(x)
        outputs.append(x)

        if self.return_all:
            return outputs
        return outputs[-1]


    def _make_modules(self, inputs, **kwargs):
        num_stages = kwargs.pop('num_stages')
        encoder_layout = ''.join([item[0] for item in kwargs.pop('order')])

        block_args = kwargs.pop('blocks')
        downsample_args = kwargs.pop('downsample')
        self.layout = ''

        for i in range(num_stages):
            for letter in encoder_layout:

                if letter in ['b']:
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, num_stages)}

                    layer = ConvBlock(inputs=inputs, **args)
                    inputs = layer(inputs)
                    layer_desc = 'block-{}'.format(i)

                elif letter in ['d', 'p']:
                    args = {**kwargs, **downsample_args, **unpack_args(downsample_args, i, num_stages)}

                    layer = ConvBlock(inputs=inputs, **args)
                    inputs = layer(inputs)
                    layer_desc = 'downsample-{}'.format(i)

                elif letter in ['s']:
                    layer = nn.Identity()
                    layer_desc = 'skip-{}'.format(i)
                else:
                    raise ValueError('Unknown letter in order {}, use one of "b", "d", "p", "s"'
                                     .format(letter))

                self.update([(layer_desc, layer)])
                self.layout += letter



class EmbeddingModule(nn.Module):
    """ Embedding: thorough processing of an input tensor. """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        inputs = inputs[-1] if isinstance(inputs, list) else inputs
        self.embedding = ConvBlock(inputs=inputs, **kwargs)

    def forward(self, x):
        inputs = x if isinstance(x, list) else [x]
        x = inputs[-1]
        inputs.append(self.embedding(x))
        return inputs



class DecoderModule(nn.ModuleDict):
    """ Decoder: increasing spatial dimensions. """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        self._make_modules(inputs, **kwargs)

    def forward(self, x):
        inputs = x if isinstance(x, list) else [x]
        x = inputs[-1]
        i = 0

        for letter, layer in zip(self.layout, self.values()):
            if letter in ['b', 'u']:
                x = layer(x)
            elif letter in ['c'] and self.skip and (i < len(inputs) - 2):
                x = layer([x, inputs[-i - 3]])
                i += 1
        return x


    def _make_modules(self, inputs, **kwargs):
        inputs = inputs if isinstance(inputs, list) else [inputs]
        x = inputs[-1]

        num_stages = kwargs.pop('num_stages') or len(inputs) - 2
        decoder_layout = ''.join([item[0] for item in kwargs.pop('order')])
        self.skip = kwargs.pop('skip')

        factor = kwargs.pop('factor') or [2]*num_stages
        if isinstance(factor, int):
            factor = int(factor ** (1/num_stages))
            factor = [factor] * num_stages
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        block_args = kwargs.pop('blocks')
        upsample_args = kwargs.pop('upsample')
        combine_args = kwargs.pop('combine')
        self.layout = ''

        for i in range(num_stages):
            for letter in decoder_layout:

                if letter in ['b']:
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, num_stages)}

                    layer = ConvBlock(inputs=x, **args)
                    x = layer(x)
                    layer_desc = 'block-{}'.format(i)

                elif letter in ['u']:
                    args = {'factor': factor[i],
                            **kwargs, **upsample_args, **unpack_args(upsample_args, i, num_stages)}

                    layer = Upsample(inputs=x, **args)
                    x = layer(x)
                    layer_desc = 'upsample-{}'.format(i)

                elif letter in ['c']:
                    if self.skip and (i < len(inputs) - 2):
                        args = {'factor': factor[i],
                                **kwargs, **combine_args, **unpack_args(combine_args, i, num_stages)}

                        layer = Combine(inputs=[x, inputs[-i - 3]], **args)
                        x = layer([x, inputs[-i - 3]])
                        layer_desc = 'combine-{}'.format(i)
                else:
                    raise ValueError('Unknown letter in order {}, use one of ("b", "u", "c")'.format(letter))

                self.update([(layer_desc, layer)])
                self.layout += letter


class Encoder(TorchModel):
    """ Encoder architecture. Allows to combine blocks from different models,
    e.g. ResNet and DenseNet, in order to create new ones with just a few lines of code.
    Intended to be used for classification tasks.

    Parameters
    ----------
    body : dict
        encoder : dict, optional
            num_stages : int
                Number of downsampling stages.

            order : str, sequence of str
                Determines order of applying layers.
                If str, then each letter stands for operation:
                'b' for 'block', 'd'/'p' for 'downsampling', 's' for 'skip'.
                If sequence, than the first letter of each item stands for operation:
                For example, `'sbd'` allows to use throw skip connection -> block -> downsampling.

            downsample : dict, optional
                Parameters for downsampling (see :class:`~.layers.ConvBlock`)

            blocks : dict, optional
                Parameters for pre-processing blocks.

                base : callable
                    Tensor processing function. Default is :class:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.
    """
    @classmethod
    def default_config(cls):
        """ Encoder's defaults: use max pooling after each `cna` stage. """
        config = super().default_config()

        config['body/encoder'] = dict(num_stages=None,
                                      order=['skip', 'block', 'downsampling'])
        config['body/encoder/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder/blocks'] = dict(base=DefaultBlock)
        return config

    @classmethod
    def body(cls, inputs, return_all=False, **kwargs):
        """ Make a sequential list of encoder modules. """
        kwargs = cls.get_defaults('body', kwargs)
        encoder = kwargs.pop('encoder')
        layers = [('encoder', EncoderModule(inputs=inputs, return_all=return_all, **{**kwargs, **encoder}))]
        return nn.Sequential(OrderedDict(layers))



class Decoder(TorchModel):
    """ Decoder architecture. Allows to combine blocks from different models,
    e.g. ResNet and DenseNet, in order to create new ones with just a few lines of code.
    Intended to be used for increasing spatial dimensionality of inputs.

    Parameters
    ----------
    body : dict
        decoder : dict, optional
            num_stages : int
                Number of upsampling blocks.

            factor : int or list of int
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage.

            skip : bool, dict
                If bool, then whether to combine upsampled tensor with stored pre-downsample encoding by
                using `combine` parameters that can be specified for each of blocks separately.

            order : str, sequence of str
                Determines order of applying layers.
                If str, then each letter stands for operation:
                'b' for 'block', 'u' for 'upsampling', 'c' for 'combine'
                If sequence, than the first letter of each item stands for operation.
                For example, `'ucb'` allows to use upsampling -> combine -> block.

            upsample : dict
                Parameters for upsampling (see :class:`~.layers.Upsample`).

            blocks : dict
                Parameters for post-processing blocks:

                base : callable
                    Tensor processing function. Default is :class:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.

            combine : dict
                If dict, then parameters for combining tensors, see :class:`~.layers.Combine`.

    head : dict, optional
        Parameters for the head layers, usually :class:`~.layers.ConvBlock` parameters. Note that an extra 1x1
        convolution may be applied in order to make predictions compatible with the shape of the targets.
    """
    @classmethod
    def default_config(cls):
        """ Decoder's defaults: use deconvolution, followed by a `cna` stage. """
        config = super().default_config()

        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None,
                                      order=['upsampling', 'block', 'combine'])
        config['body/decoder/upsample'] = dict(layout='tna')
        config['body/decoder/blocks'] = dict(base=DefaultBlock)
        config['body/decoder/combine'] = dict(op='concat', leading_index=1)
        return config


    @classmethod
    def body(cls, inputs, **kwargs):
        """ Make a sequential list of decoder modules. """
        kwargs = cls.get_defaults('body', kwargs)
        decoder = kwargs.pop('decoder')
        layers = [('decoder', DecoderModule(inputs=inputs, **{**kwargs, **decoder}))]
        return nn.Sequential(OrderedDict(layers))

    @classmethod
    def head(cls, inputs, target_shape, classes, **kwargs):
        """ Make network's head. If needed, apply 1x1 convolution to obtain correct output shape. """
        kwargs = cls.get_defaults('head', kwargs)
        layers = []
        layer = super().head(inputs, target_shape, classes, **kwargs)
        if layer is not None:
            inputs = layer(inputs)
            layers.append(layer)

        if classes:
            if get_shape(inputs)[1] != classes:
                layer = ConvBlock(inputs=inputs, layout='c', filters=classes, kernel_size=1)
                layers.append(layer)
        return nn.Sequential(*layers)



class EncoderDecoder(Decoder):
    """ Encoder-decoder architecture. Allows to combine blocks from different models,
    e.g. ResNet and DenseNet, in order to create new ones with just a few lines of code.
    Intended to be used for segmentation tasks.

    Parameters
    ----------
    body : dict
        encoder : dict, optional
            num_stages : int
                Number of downsampling stages.

            order : str, sequence of str
                Determines order of applying layers.
                If str, then each letter stands for operation:
                'b' for 'block', 'd'/'p' for 'downsampling', 's' for 'skip'.
                If sequence, than the first letter of each item stands for operation:
                For example, `'sbd'` allows to use throw skip connection -> block -> downsampling.

            downsample : dict, optional
                Parameters for downsampling (see :class:`~.layers.ConvBlock`)

            blocks : dict, optional
                Parameters for pre-processing blocks.

                base : callable
                    Tensor processing function. Default is :class:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.

        embedding : dict or None, optional
            If None no embedding block is created.
            If dict, then parameters for tensor processing function.

            base : callable
                Tensor processing function. Default is :class:`~.layers.ConvBlock`.
            other args
                Parameters for the base block.

        decoder : dict, optional
            num_stages : int
                Number of upsampling blocks.

            factor : int or list of int
                If int, the total upsampling factor for all stages combined.
                If list, upsampling factors for each stage.

            skip : bool, dict
                If bool, then whether to combine upsampled tensor with stored pre-downsample encoding by
                using `combine` parameters that can be specified for each of blocks separately.

            order : str, sequence of str
                Determines order of applying layers.
                If str, then each letter stands for operation:
                'b' for 'block', 'u' for 'upsampling', 'c' for 'combine'
                If sequence, than the first letter of each item stands for operation.
                For example, `'ucb'` allows to use upsampling -> combine -> block.

            upsample : dict
                Parameters for upsampling (see :class:`~.layers.Upsample`).

            blocks : dict
                Parameters for post-processing blocks:

                base : callable
                    Tensor processing function. Default is :class:`~.layers.ConvBlock`.
                other args : dict
                    Parameters for the base block.

            combine : dict
                If dict, then parameters for combining tensors, see :class:`~.layers.Combine`.

    head : dict, optional
        Parameters for the head layers, usually :class:`~.layers.ConvBlock` parameters. Note that an extra 1x1
        convolution may be applied in order to make predictions compatible with the shape of the targets.

    Examples
    --------
    Use ResNet as an encoder with desired number of blocks and filters in them (total downsampling factor is 4),
    create an embedding that contains 256 channels, then upsample it to get 8 times the size of initial image.

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape')},
                           masks={'name': 'targets', 'shape': B('mask_shape')}),
            'initial_block/inputs': 'images',
            'body/encoder': {'base': ResNet,
                             'num_blocks': [2, 3, 4]
                             'filters': [16, 32, 128]},
            'body/embedding': {'layout': 'cna', 'filters': 256},
            'body/decoder': {'num_stages': 5, 'factor': 32},
        }

    Preprocess input image with 7x7 convolutions, downsample it 5 times with DenseNet blocks in between,
    use MobileNet block in the bottom, then restore original image size with subpixel convolutions and
    ResNeXt blocks in between:

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape')},
                           masks={'name': 'targets', 'shape': B('mask_shape')}),
            'initial_block': {'inputs': 'images',
                              'layout': 'cna', 'filters': 4, 'kernel_size': 7},
            'body/encoder': {'num_stages': 5,
                             'blocks': {'base': DenseNet.block,
                                        'num_layers': [2, 2, 3, 4, 5],
                                        'growth_rate': 6, 'skip': True}},
            'body/embedding': {'base': MobileNet.block,
                               'width_factor': 2},
            'body/decoder': {'upsample': {'layout': 'X'},
                             'blocks': {'base': ResNet.block,
                                        'filters': [256, 128, 64, 32, 16],
                                        'resnext': True}},
        }
    """
    @classmethod
    def default_config(cls):
        """ Encoder, followed by a decoder, with skips between stages. """
        config = super().default_config()

        config['body/encoder'] = dict(num_stages=None,
                                      order=['skip', 'block', 'downsampling'])
        config['body/encoder/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder/blocks'] = dict(base=DefaultBlock)

        config['body/embedding'] = dict(base=DefaultBlock)

        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None,
                                      order=['upsampling', 'block', 'combine'])
        config['body/decoder/upsample'] = dict(layout='tna')
        config['body/decoder/blocks'] = dict(base=DefaultBlock)
        config['body/decoder/combine'] = dict(op='concat', leading_index=1)
        return config


    @classmethod
    def body(cls, inputs, **kwargs):
        """ Sequence of encoder, embedding and decoder. """
        kwargs = cls.get_defaults('body', kwargs)
        encoder = kwargs.pop('encoder')
        embedding = kwargs.pop('embedding')
        decoder = kwargs.pop('decoder')

        layers = []
        encoder = cls.encoder(inputs=inputs, **{**kwargs, **encoder})
        encoder_outputs = encoder(inputs)
        layers.append(('encoder', encoder))

        if embedding is not None:
            embedding = cls.embedding(inputs=encoder_outputs, **{**kwargs, **embedding})
        else:
            embedding = nn.Identity()
        encoder_outputs = embedding(encoder_outputs)
        layers.append(('embedding', embedding))

        decoder = cls.decoder(inputs=encoder_outputs, **{**kwargs, **decoder})
        layers.append(('decoder', decoder))

        return nn.Sequential(OrderedDict(layers))

    @classmethod
    def encoder(cls, inputs, **kwargs):
        """ Create encoder either from base model or block args. """
        if 'base_model' in kwargs:
            base_model = kwargs['base_model']
            base_model_kwargs = kwargs.get('base_model_kwargs', {})
            return base_model.body(inputs=inputs, return_all=True, encoder=base_model_kwargs).encoder
        return EncoderModule(inputs=inputs, **kwargs)

    @classmethod
    def embedding(cls, inputs, **kwargs):
        return EmbeddingModule(inputs=inputs, **kwargs)

    @classmethod
    def decoder(cls, inputs, **kwargs):
        return DecoderModule(inputs=inputs, **kwargs)



class AutoEncoder(EncoderDecoder):
    """ Model without skip-connections between corresponding stages of encoder and decoder. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/decoder'] += dict(skip=False)
        return config



class VariationalBlock(nn.Module):
    """ Reparametrization trick block. """
    def __init__(self, inputs=None, base_mu=None, base_std=None, **kwargs):
        super().__init__()
        self.mean = base_mu(inputs=inputs, **kwargs)
        self.std = base_std(inputs=inputs, **kwargs)

    def forward(self, x):
        mean = self.mean(x)
        std = self.std(x)
        return mean + std * torch.randn_like(std)


class VariationalAutoEncoder(AutoEncoder):
    """ Autoencoder that maps input into distribution. Based on
    Kingma, Diederik P; Welling, Max "`Auto-Encoding Variational Bayes
    <https://arxiv.org/abs/1312.6114>`_"

    Notes
    -----
    Distribution that is learned is always normal.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/embedding'] += dict(base=VariationalBlock,
                                         base_mu=DefaultBlock, base_std=DefaultBlock)
        return config
