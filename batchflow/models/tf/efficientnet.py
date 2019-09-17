""" Base class for scalable models and EfficientNet

Mingxing Tan, Quoc V. Le "`EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
<https://arxiv.org/abs/1905.11946>`_"

"""
import numpy as np
import tensorflow as tf

from . import TFModel, MobileNet_v2
from .layers import conv_block


class ScalableModel(TFModel):
    """ Base class for scalable models.
    Scaling can be done by depth and/or by width

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block
        base :
            Base block. default is :func:`~.layers.conv_block`
        repeats : int
            number of repeats of the block
        scalable : bool
            indicates whether the block can be scaled
        other parameters :
            see :func:`~.layers.conv_block` or base block documentation

    body : dict
        base : function
            Base block. default is :func:`~.layers.conv_block`
        blocks : list of dict
            parameters for each block
            repeats : int
                number of repeats of the block
            scalable : bool
                indicates whether the block can be scaled
            other parameters :
                see :func:`~.layers.conv_block` or base block documentation

    head : dict
        parameters for head

    common : dict
        common parameters
        width_factor :
        depth_factor : float
            scaling factors
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['body'].update(dict(base=None,
                                   blocks=[dict(repeats=1, scalable=True, layout='cna cna', kernel_size=3, filters=8)]))
        config['head'].update(dict(layout='Pf'))

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
        """ Round number of filters based on width multiplier. """
        return int(filters * factor)

    @classmethod
    def round_repeats(cls, repeats, factor):
        """ Round number of layers based on depth multiplier. """
        return int(np.ceil(repeats * factor))

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        kwargs = cls.fill_params(name, **kwargs)
        if kwargs.get('layout'):
            return cls.block(inputs, name=name, **kwargs)
        return inputs

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
        kwargs = cls.fill_params(name, **kwargs)
        if kwargs.get('layout'):
            return cls.block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def block(cls, inputs, name, **block_args):
        """ block of repeated layers of same structure"""
        base_block = block_args.pop('base', None)

        new_layer_args = cls.update_layer_params(block_args)
        if base_block is None:
            with tf.variable_scope(name):
                repeats = new_layer_args.get('repeats', 1)
                x = inputs
                for i in range(repeats):
                    x = conv_block(x, name='layer-%d' % i, **new_layer_args)

                return x
        else:
            return base_block(inputs, name=name, **new_layer_args)

    @classmethod
    def update_layer_params(cls, kwargs):
        """ calculate filters and number of repeats depending on scaling factors """
        new_kwargs = dict(**kwargs)
        scalable = new_kwargs.get('scalable', False)

        if not scalable:
            return new_kwargs

        w_factor = new_kwargs.get('width_factor')
        filters = new_kwargs.get('filters')
        if filters and w_factor > 1:
            if isinstance(filters, int):
                new_kwargs['filters'] = cls.round_filters(filters, w_factor)
            elif isinstance(filters, list):
                new_kwargs['filters'] = [cls.round_filters(f, w_factor) for f in filters]
            else:
                raise ValueError("filters should be int or list, {} given".format(type(filters)))

        d_factor = kwargs.get('depth_factor')
        repeats = kwargs.get('repeats')
        if repeats and d_factor > 1:
            new_kwargs['repeats'] = cls.round_repeats(repeats, d_factor)

        return new_kwargs


class EfficientNetB0(ScalableModel):
    """
    EfficientNetB0

    **Configuration**
        see :class:`ScalableModel`
        Base block is :meth:`.MobileNet_v2.block`

    """

    resolution = 224  # image resolution used in original paper

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['initial_block'].update(dict(scalable=True, layout='cna', kernel_size=3, strides=2, filters=32))

        config['body'].update(dict(base=MobileNet_v2.block, blocks=[
            # NB! when expansion_factor=1 original efficientnet implementation skips expansion convolution
            dict(repeats=1, scalable=True, kernel_size=3, strides=1, filters=16,
                 expansion_factor=1, se_block=dict(ratio=4)),
            dict(repeats=2, scalable=True, kernel_size=3, strides=2, filters=24,
                 expansion_factor=6, se_block=dict(ratio=4)),
            dict(repeats=2, scalable=True, kernel_size=5, strides=2, filters=40,
                 expansion_factor=6, se_block=dict(ratio=4)),
            dict(repeats=3, scalable=True, kernel_size=3, strides=2, filters=80,
                 expansion_factor=6, se_block=dict(ratio=4)),
            dict(repeats=3, scalable=True, kernel_size=5, strides=1, filters=112,
                 expansion_factor=6, se_block=dict(ratio=4)),
            dict(repeats=4, scalable=True, kernel_size=5, strides=2, filters=192,
                 expansion_factor=6, se_block=dict(ratio=4)),
            dict(repeats=1, scalable=True, kernel_size=3, strides=1, filters=320,
                 expansion_factor=6, se_block=dict(ratio=4)),
        ]))

        config['head'].update(dict(scalable=True, layout='cna V df', kernel_size=1, strides=1, filters=1280,
                                   dropout_rate=0.2))

        config['common'].update(activation=tf.nn.swish)

        return config


class EfficientNetB1(EfficientNetB0):
    """ EfficientNetB1 """

    resolution = 240

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.0
        config['common/depth_factor'] = 1.1

        config['head/dropout_rate'] = 0.2

        return config


class EfficientNetB2(EfficientNetB0):
    """ EfficientNetB2 """

    resolution = 260

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.1
        config['common/depth_factor'] = 1.2

        config['head/dropout_rate'] = 0.3

        return config


class EfficientNetB3(EfficientNetB0):
    """ EfficientNetB3 """

    resolution = 300

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.2
        config['common/depth_factor'] = 1.4

        config['head/dropout_rate'] = 0.3

        return config


class EfficientNetB4(EfficientNetB0):
    """ EfficientNetB4 """

    resolution = 380

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.4
        config['common/depth_factor'] = 1.8

        config['head/dropout_rate'] = 0.4

        return config


class EfficientNetB5(EfficientNetB0):
    """ EfficientNetB5 """

    resolution = 456

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.6
        config['common/depth_factor'] = 2.2

        config['head/dropout_rate'] = 0.4

        return config


class EfficientNetB6(EfficientNetB0):
    """ EfficientNetB6 """

    resolution = 528

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 1.8
        config['common/depth_factor'] = 2.6

        config['head/dropout_rate'] = 0.5

        return config


class EfficientNetB7(EfficientNetB0):
    """ EfficientNetB7 """

    resolution = 600

    @classmethod
    def default_config(cls):
        config = super().default_config()

        config['common/width_factor'] = 2.0
        config['common/depth_factor'] = 3.1

        config['head/dropout_rate'] = 0.5

        return config
