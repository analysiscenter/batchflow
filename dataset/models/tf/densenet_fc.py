""" Jegou S. et al "`The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation
<https://arxiv.org/abs/1611.09326>`_"
"""
import tensorflow as tf

from . import TFModel
from .densenet import DenseNet


class DenseNetFC(TFModel):
    """ DenseNet for semantic segmentation

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        num_blocks : list of int
            number of layers in downsampling/upsampling blocks

        block : dict
            dense block parameters

        transition_down : dict
            downsampling transition layer parameters

        transition_up : dict
            upsampling transition layer parameters
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['input_block'].update(dict(layout='c', filters=48, kernel_size=3, strides=1,
                                          pool_size=3, pool_strides=2))

        config['body']['block'] = dict(layout='nacd', dropout_rate=.2, growth_rate=12, bottleneck=False)
        config['body']['upsample'] = dict(layout='nat', factor=2, kernel_size=3)
        config['body']['transition_down'] = dict(layout='nacdp', kernel_size=1, strides=1,
                                                 pool_size=2, pool_strides=2, dropout_rate=.2,
                                                 reduction_factor=1)

        config['head'].update(dict(layout='c', kernel_size=1))
        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ FC DenseNet body

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body', **kwargs)
        num_blocks, block = cls.pop(['num_blocks', 'block'], kwargs)
        trans_up, trans_down = cls.pop(['upsample', 'transition_down'], kwargs)
        block = {**kwargs, **block}
        trans_up = {**kwargs, **trans_up}
        trans_down = {**kwargs, **trans_down}

        with tf.variable_scope(name):
            x, inputs = inputs, None
            encoder_outputs = []
            for i, num_layers in enumerate(num_blocks[:-1]):
                x = cls.encoder_block(x, num_layers=num_layers, name='encoder-'+str(i), **block)
                encoder_outputs.append(x)
                x = cls.transition_down(x, name='transition_down-%d' % i, **trans_down)

            axis = cls.channels_axis(kwargs.get('data_format'))
            for i, num_layers in enumerate(num_blocks[::-1][:-1]):
                x = cls.decoder_block(x, num_layers=num_layers, name='decoder-'+str(i), **block)
                x = cls.transition_up(x, name='transition_up-%d' % i, **trans_up)
                x = cls.crop(x, encoder_outputs[-i-1], data_format=kwargs.get('data_format'))
                x = tf.concat((x, encoder_outputs[-i-1]), axis=axis)
            x = cls.decoder_block(x, num_layers=num_blocks[0], name='decoder-'+str(i+1), **block)
        return x

    @classmethod
    def encoder_block(cls, inputs, name, **kwargs):
        """ DenseNet block + shortcut

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        with tf.variable_scope(name):
            x = DenseNet.block(inputs, name='dense-block', **kwargs)
            axis = cls.channels_axis(kwargs.get('data_format'))
            x = tf.concat((inputs, x), axis=axis)
        return x

    @classmethod
    def decoder_block(cls, inputs, name, **kwargs):
        """ DenseNet block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)
        x = DenseNet.block(inputs, name=name, **kwargs)
        return x

    @classmethod
    def transition_down(cls, inputs, name='transition_down', **kwargs):
        """ A downsampling interconnect layer between two dense blocks

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/transition_down', **kwargs)
        return DenseNet.transition_layer(inputs, name=name, **kwargs)

    @classmethod
    def transition_up(cls, inputs, name='transition_up', **kwargs):
        """ An upsampling interconnect layer between two dense blocks

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/upsample', **kwargs)
        num_filters = cls.num_channels(inputs, kwargs.get('data_format'))
        return cls.upsample(inputs, filters=num_filters, name=name, **kwargs)


class DenseNetFC56(DenseNetFC):
    """ FC DenseNet-56 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_blocks'] = [4] * 6
        config['body']['block']['growth_rate'] = 12
        return config

class DenseNetFC67(DenseNetFC):
    """ FC DenseNet-67 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_blocks'] = [5] * 6
        config['body']['block']['growth_rate'] = 16
        return config

class DenseNetFC103(DenseNetFC):
    """ FC DenseNet-103 architecture """
    @classmethod
    def default_config(cls):
        config = DenseNetFC.default_config()
        config['body']['num_blocks'] = [4, 5, 7, 10, 12, 15]
        config['body']['block']['growth_rate'] = 16
        return config

