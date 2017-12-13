"""  Lin G. et al "`RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
<https://arxiv.org/abs/1611.06612>`_"
"""
import tensorflow as tf
import numpy as np

from .layers import conv_block
from . import TFModel
from .resnet import ResNet, ResNet101


class RefineNet(TFModel):
    """ RefineNet

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        num_blocks : int
            number of downsampling/upsampling blocks (default=4)

        filters : list of int
            number of filters in each block (default=[128, 256, 512, 1024])

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        filters = 64   # number of filters in the first block
        config['input_block'].update(dict(layout='cna cna', filters=filters, kernel_size=3, strides=1))
        config['body']['encoder'] = dict(base_class=ResNet101)
        config['body']['num_blocks'] = 4
        config['body']['filters'] = 2 ** np.arange(config['body']['num_blocks']) * filters * 2
        config['body']['upsample'] = dict(layout='tna', factor=2)
        config['head'].update(dict(layout='cna cna', filters=filters, kernel_size=3, strides=1))
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['num_classes'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of int
            number of filters in downsampling blocks
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        filters = kwargs.pop('filters')

        with tf.variable_scope(name):
            encoder_outputs = cls.make_encoder(inputs, **encoder, **kwargs)

            x = None
            for i, tensor in enumerate(encoder_outputs):
                decoder_inputs = encoder_outputs[-i-1] if x is None else (encoder_outputs[-i-1], x)
                x = cls.decoder_block(decoder_inputs, filters=filters[i], name='decoder-'+str(i), **kwargs)

        return x

    @classmethod
    def encoder(cls, inputs, base_class, name, **kwargs):
        """ Create encoder from a base_class model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        base_class : TFModel
            a model class (default=ResNet101).
            Should implement ``make_encoder`` method.
        name : str
            scope name
        kwargs : dict
            input_block : dict
                input_block parameters for ``base_class`` model
            body : dict
                body parameters for ``base_class`` model
            and any other ``conv_block`` params.

        Returns
        -------
        tf.Tensor
        """
        x = base_class.make_encoder(inputs, name=name, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, name='block', **kwargs):
        """ RefineNet block with Residual Conv Unit, Multi-resolution fusion and Chained-residual pooling.

        Parameters
        ----------
        inputs : tuple of tf.Tensor
            input tensors (the first should have the largest spatial dimension)
        name : str
            scope name
        kwargs : dict
            upsample : dict
                upsample params

        Returns
        -------
        tf.Tensor
        """
        upsample_args = cls.pop('upsample', kwargs)
        upsample_args = {**kwargs, **upsample_args}

        with tf.variable_scope(name):
            filters = min([cls.num_channels(t, data_format=kwargs['data_format']) for t in inputs])
            # Residual Conv Unit
            after_rcu = []
            for i, tensor in enumerate(inputs):
                x = ResNet.double_block(tensor, filters=filters, layout='acac',
                                        bottleneck=False, downsample=False,
                                        name='rcu-%d' % i, **kwargs)
                after_rcu.append(x)

            # Multi-resolution fusion
            with tf.variable_scope('mrf'):
                after_mrf = 0
                for i, tensor in enumerate(after_rcu):
                    x = conv_block(tensor, layout='ac', filters=filters, kernel_size=3,
                                   name='conv-%d' % i, **kwargs)
                    x = cls.upsample((tensor, after_rcu[0]), layout='b', name='upsample-%d' % i, **upsample_args)
                    after_mrf += x
            # free memory
            x, after_mrf = after_mrf, None
            after_rcu = None

            # Chained-residual pooling
            x = tf.relu(x)
            after_crp = x
            num_pools = 4
            for i in range(num_pools):
                x = conv_block(x, layout='pc', filters=filters, kernel_size=3, strides=1,
                               pool_size=5, pool_strides=1, name='rcp-%d' % i, **kwargs)
                after_crp += x

            x, after_crp = after_crp, None
            x = ResNet.double_block(x, layout='acac', filters=filters, bottleneck=False, downsample=False,
                                    name='rcu-last', **kwargs)
            x = tf.identity(x, name='output')
        return x

    @classmethod
    def decoder_block(cls, inputs, filters, name, **kwargs):
        """ 3x3 convolution and 2x2 transposed convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        config = cls.fill_params('body', **kwargs)
        upsample_args = cls.pop('upsample', config)

        with tf.variable_scope(name):
            x, skip = inputs
            x = cls.upsample(x, filters=filters, name='upsample', **upsample_args)
            x = cls.crop(x, skip, data_format=kwargs.get('data_format'))
            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat((skip, x), axis=axis)
            x = conv_block(x, 'cnacna', filters, kernel_size=3, name='conv', **kwargs)
        return x

    @classmethod
    def head(cls, inputs, num_classes, name='head', **kwargs):
        """ Conv block followed by 1x1 convolution

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        num_classes : int
            number of classes (and number of filters in the last 1x1 convolution)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('head', **kwargs)
        with tf.variable_scope(name):
            x = conv_block(inputs, name='conv', **kwargs)
            x = conv_block(inputs, name='last', **{**kwargs, **dict(filters=num_classes, kernel_size=1, layout='c')})
        return x
