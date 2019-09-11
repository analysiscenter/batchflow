""" EfficientNet """
# pylint: disable=missing-docstring

import numpy as np
import tensorflow as tf

from . import TFModel, MobileNet_v2
from .layers import conv_block


class ScalableModel(TFModel):

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['body/blocks'] = [
            dict(repeats=1, scalable=True, layout='cna cna', kernel_size=3, filters=8)
        ]
        config['head'] = dict(layout='Pf')

        config['common/width_factor'] = 1.0
        config['common/depth_factor'] = 1.0

        config['loss'] = 'ce'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)

        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')

        return config

    @classmethod
    def round_filters(cls, filters, factor):
        return int(filters * factor)

    @classmethod
    def round_repeats(cls, repeats, factor):
        return int(np.ceil(repeats * factor))

    @classmethod
    def non_repeated_block(cls, inputs, name, **kwargs):
        kwargs = cls.fill_params(name, **kwargs)
        if kwargs.get('layout'):
            return cls.block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        return cls.non_repeated_block(inputs, name, **kwargs)

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        kwargs = cls.fill_params('body', **kwargs)

        with tf.variable_scope(name):
            blocks = kwargs.pop('blocks')
            x = inputs
            for i, block_args in enumerate(blocks):
                x = cls.block(x, name='block-%d' % i, **{**kwargs, **block_args})

            return x

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        return cls.non_repeated_block(inputs, name, **kwargs)

    @classmethod
    def block(cls, inputs, name, **block_args):
        base_block = block_args.pop('base', None)
        if base_block is None:
            base_block = cls.conv_block

        new_layer_args = cls.update_layer_params(block_args)
        return base_block(inputs, name=name, **new_layer_args)

    @classmethod
    def conv_block(cls, inputs, name, **kwargs):
        with tf.variable_scope(name):
            repeats = kwargs.get('repeats', 1)
            x = inputs
            for i in range(repeats):
                x = conv_block(x, name='layer-%d' % i, **kwargs)

        return x

    @classmethod
    def update_layer_params(cls, kwargs):

        new_kwargs = dict(**kwargs)
        scalable = new_kwargs.get('scalable', False)

        if not scalable:
            return new_kwargs

        w_factor = new_kwargs.get('width_factor')
        filters = new_kwargs.pop('filters', None)
        if filters:
            if isinstance(filters, int):
                new_kwargs['filters'] = cls.round_filters(filters, w_factor)
            elif isinstance(filters, list):
                new_kwargs['filters'] = [cls.round_filters(f, w_factor) for f in filters]
            else:
                raise ValueError("filters should be int or list, {} given".format(type(filters)))

        d_factor = kwargs.get('depth_factor')
        repeats = kwargs.get('repeats')
        if repeats:
            new_kwargs['repeats'] = cls.round_repeats(repeats, d_factor)

        return new_kwargs


class EfficientNetB0(ScalableModel):
    """

    phi - scaling parameter
    alpha - depth scaling base, depth factor `d=alpha^phi`
    beta - width (number of channels) scaling base, width factor `w=beta^phi`
    resolution is set explicitly via inputs resolution.
    helper function `get_resolution_factor` is provided to calculate resolution factor `r`
    by given `alpha`, `beta`, `phi`, so that :math: `r^2 * w^2 * d \approx 2^phi`

    """

    resolution = 224

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['initial_block'] = dict(scalable=True, layout='cna', kernel_size=3, strides=2, filters=32,
                                       activation=tf.nn.swish)

        config['body/base'] = MobileNet_v2.block
        config['body/blocks'] = [
            # NB! when expansion_factor=1 original implementaion skips expansion convolution
            dict(repeats=1, scalable=True, kernel_size=3, strides=1, filters=16,
                 expansion_factor=1,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=2, scalable=True, kernel_size=3, strides=2, filters=24,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=2, scalable=True, kernel_size=5, strides=2, filters=40,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=3, scalable=True, kernel_size=3, strides=2, filters=80,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=3, scalable=True, kernel_size=5, strides=1, filters=112,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=4, scalable=True, kernel_size=5, strides=2, filters=192,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
            dict(repeats=1, scalable=True, kernel_size=3, strides=1, filters=320,
                 expansion_factor=6,
                 se_block=dict(ratio=0.25), activation=tf.nn.swish),
        ]

        config['head'] = dict(scalable=True, layout='cna V df', kernel_size=1, strides=1, filters=1280,
                              activation=tf.nn.swish, dropout_rate=0.2)

        return config

    @classmethod
    def round_filters(cls, filters, factor):
        """Round number of filters based on depth multiplier."""
        divisor = 8
        min_depth = 8
        if not factor:
            return filters

        filters *= factor
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)


class EfficientNetB1(EfficientNetB0):

    resolution = 240

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.0
        config['common/depth_factor'] = 1.1

        config['head/dropout_rate'] = 0.2


class EfficientNetB2(EfficientNetB0):

    resolution = 260

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.1
        config['common/depth_factor'] = 1.2

        config['head/dropout_rate'] = 0.3


class EfficientNetB3(EfficientNetB0):

    resolution = 300

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.2
        config['common/depth_factor'] = 1.4

        config['head/dropout_rate'] = 0.3


class EfficientNetB4(EfficientNetB0):

    resolution = 380

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.4
        config['common/depth_factor'] = 1.8

        config['head/dropout_rate'] = 0.4


class EfficientNetB5(EfficientNetB0):

    resolution = 456

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.6
        config['common/depth_factor'] = 2.2

        config['head/dropout_rate'] = 0.4


class EfficientNetB6(EfficientNetB0):

    resolution = 528

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.8
        config['common/depth_factor'] = 2.6

        config['head/dropout_rate'] = 0.5


class EfficientNetB7(EfficientNetB0):

    resolution = 600

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 2.0
        config['common/depth_factor'] = 3.1

        config['head/dropout_rate'] = 0.5
