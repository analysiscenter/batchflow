""" Encoder, decoder, encoder-decoder architectures. """
import numpy as np
import torch
import torch.nn as nn

from .base import EagerTorch
from .utils import get_shape
from .layers import ConvBlock, Upsample, Combine, Crop
from ..utils import unpack_args



class DefaultBlock(nn.Module):
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        layout = kwargs.pop('layout', 'cna')
        filters = kwargs.pop('filters', 'same*2')
        self.layer = ConvBlock(inputs=inputs, layout=layout, filters=filters, **kwargs)

    def forward(self, x):
        return self.layer(x)



class EncoderModule(nn.Module):
    def __init__(self, inputs=None, return_all=True, **kwargs):
        super().__init__()
        self.return_all = return_all
        self.make_modules(inputs, **kwargs)

    def forward(self, x):
        b_counter, d_counter = 0, 0
        outputs = []

        for i in range(self.num_stages):
            for letter in self.encoder_layout:
                if letter in ['b']:
                    x = self.encoder_b[b_counter](x)
                    b_counter += 1
                elif letter in ['d', 'p']:
                    x = self.encoder_d[d_counter](x)
                    d_counter += 1
                elif letter in ['s']:
                    outputs.append(x)
        outputs.append(x)

        if self.return_all:
            return outputs
        return outputs[-1]


    def make_modules(self, inputs, **kwargs):
        num_stages = kwargs.pop('num_stages')
        encoder_layout = ''.join([item[0] for item in kwargs.pop('order')])
        self.num_stages, self.encoder_layout = num_stages, encoder_layout

        block_args = kwargs.pop('blocks')
        downsample_args = kwargs.pop('downsample')

        self.encoder_b, self.encoder_d = nn.ModuleList(), nn.ModuleList()

        for i in range(num_stages):
            for letter in encoder_layout:
                if letter in ['b']:
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, num_stages)}
                    base_block = args.get('base')

                    layer = base_block(inputs=inputs, **args)
                    inputs = layer(inputs)
                    self.encoder_b.append(layer)
                elif letter in ['d', 'p']:
                    args = {**kwargs, **downsample_args, **unpack_args(downsample_args, i, num_stages)}

                    layer = ConvBlock(inputs=inputs, **args)
                    inputs = layer(inputs)
                    self.encoder_d.append(layer)
                elif letter in ['s']:
                    pass
                else:
                    raise ValueError('BAD', letter)



class EmbeddingModule(nn.Module):
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        base_block = kwargs.get('base')
        if base_block is not None:
            inputs = inputs[-1] if isinstance(inputs, list) else inputs
            self.embedding = base_block(inputs=inputs, **kwargs)
        else:
            self.embedding = nn.Identity()

    def forward(self, x):
        inputs = x if isinstance(x, list) else [x]
        x = inputs[-1]
        inputs.append(self.embedding(x))
        return inputs



class DecoderModule(nn.Module):
    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        self.make_modules(inputs, **kwargs)


    def forward(self, x):
        inputs = x if isinstance(x, list) else [x]
        x = inputs[-1]

        b_counter, u_counter, c_counter = 0, 0, 0

        for i in range(self.num_stages):
            for letter in self.decoder_layout:
                if letter in ['b']:
                    x = self.decoder_b[b_counter](x)
                    b_counter += 1
                elif letter in ['u']:
                    x = self.decoder_u[u_counter](x)
                    u_counter += 1
                elif letter in ['c']:
                    if self.skip and (i < len(inputs) - 2):
                        x = self.decoder_c[c_counter]([inputs[-i - 3], x])
                        c_counter += 1
        return x


    def make_modules(self, inputs, **kwargs):
        inputs = inputs if isinstance(inputs, list) else [inputs]
        x = inputs[-1]

        num_stages = kwargs.pop('num_stages') or len(inputs) - 2
        decoder_layout = ''.join([item[0] for item in kwargs.pop('order')])
        self.num_stages, self.decoder_layout = num_stages, decoder_layout

        skip = kwargs.pop('skip')
        self.skip = skip

        factor = kwargs.pop('factor') or [2]*num_stages
        if isinstance(factor, int):
            factor = int(factor ** (1/num_stages))
            factor = [factor] * num_stages
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        block_args = kwargs.pop('blocks')
        upsample_args = kwargs.pop('upsample')
        combine_args = kwargs.pop('combine')

        self.decoder_b, self.decoder_u, self.decoder_c = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for i in range(num_stages):
            for letter in decoder_layout:
                if letter in ['b']:
                    args = {'filters': 'same // 2',
                            **kwargs, **block_args, **unpack_args(block_args, i, num_stages)}
                    base_block = args.get('base')

                    layer = base_block(inputs=x, **args)
                    x = layer(x)
                    self.decoder_b.append(layer)
                elif letter in ['u']:
                    args = {'factor': factor[i],
                            **kwargs, **upsample_args, **unpack_args(upsample_args, i, num_stages)}

                    layer = Upsample(inputs=x, **args)
                    x = layer(x)
                    self.decoder_u.append(layer)
                elif letter in ['c']:
                    args = {**kwargs, **combine_args, **unpack_args(combine_args, i, num_stages)}

                    if skip and (i < len(inputs) - 2):
                        layer = Combine(inputs=[inputs[-i - 3], x])
                        x = layer([inputs[-i - 3], x])
                        self.decoder_c.append(layer)
                else:
                    raise ValueError('BAD')



class Encoder(EagerTorch):
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['body/encoder'] = dict(num_stages=None,
                                      order=['skip', 'block', 'downsampling'])
        config['body/encoder/downsample'] = dict(layout='p', pool_size=2, pool_strides=2)
        config['body/encoder/blocks'] = dict(base=DefaultBlock)
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('body', kwargs)
        encoder = kwargs.pop('encoder')
        return EncoderModule(inputs=inputs, return_all=False, **{**kwargs, **encoder})




class Decoder(EagerTorch):
    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['body/decoder'] = dict(skip=True, num_stages=None, factor=None,
                                      order=['upsampling', 'block', 'combine'])
        config['body/decoder/upsample'] = dict(layout='tna')
        config['body/decoder/blocks'] = dict(base=DefaultBlock)
        config['body/decoder/combine'] = dict(op='concat')
        return config


    @classmethod
    def body(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('body', kwargs)
        decoder = kwargs.pop('decoder')
        return DecoderModule(inputs=inputs, **{**kwargs, **decoder})

    @classmethod
    def head(cls, inputs, target_shape, classes, **kwargs):
        kwargs = cls.get_defaults('head', kwargs)
        layers = []
        layer = super().head(inputs, target_shape, classes, **kwargs)
        if layer is not None:
            inputs = layer(inputs)
            layers.append(layer)

        if get_shape(inputs) != target_shape:
            layer = Crop(resize_to=target_shape)
            inputs = layer(inputs)
            layers.append(layer)

            if get_shape(inputs)[1] != classes:
                layer = ConvBlock(inputs=inputs, layout='c', filters=classes, kernel_size=1)
                layers.append(layer)
        return nn.Sequential(*layers)



class EncoderDecoder(Decoder):
    @classmethod
    def default_config(cls):
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
        config['body/decoder/combine'] = dict(op='concat')
        return config


    @classmethod
    def body(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('body', kwargs)
        encoder = kwargs.pop('encoder')
        embedding = kwargs.pop('embedding')
        decoder = kwargs.pop('decoder')

        encoder = EncoderModule(inputs=inputs, **{**kwargs, **encoder})
        encoder_outputs = encoder(inputs)

        embedding = EmbeddingModule(inputs=encoder_outputs, **{**kwargs, **embedding})
        encoder_outputs = embedding(encoder_outputs)

        decoder = DecoderModule(inputs=encoder_outputs, **{**kwargs, **decoder})

        return nn.Sequential(encoder, embedding, decoder)



class AutoEncoder(EncoderDecoder):
    """ Model without skip-connections between corresponding stages of encoder and decoder. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/decoder'] += dict(skip=False)
        return config



class VariationalBlock(nn.Module):
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
        config['body/embedding'] += dict(base=VariationalBlock, base_mu=DefaultBlock, base_std=DefaultBlock)
        return config