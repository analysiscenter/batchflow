"""  Encoder-decoder """

import tensorflow as tf

from .layers import conv_block
from . import TFModel
from .resnet import ResNet18


class EncoderDecoder(TFModel):
    """ AutoEncoder

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    body : dict
        encoder : dict
            base_class : TFModel
                a model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs
            other args
                parameters for base class ``make_encoder`` method

        filters : list of int
            number of filters in each decoder block (default=[512, 256, 256, 256])

        upsample : dict
            :meth:`~.TFModel.upsample` parameters to use in each decoder block

    head : dict
        num_classes : int
            number of semantic classes
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body']['encoder'] = dict(base_class=ResNet18)
        config['body']['decoder'] = dict(layout='tna', factor=2)
        config['body']['embedding'] = dict(layout='cna', filters=1)
        config['loss'] = 'mse'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['targets'] = self.targets
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

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

            x = cls.decoder(encoder_outputs + [x], **decoder, **kwargs)
        return x

    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
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
        inputs : tf.Tensor
            input tensor
        base_class : TFModel
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
        inputs : tf.Tensor
            input tensor
        name : str
            scope name
        kwargs : dict
            parameters for :func:`~.tf.layers.conv_block`

        Returns
        -------
        tf.Tensor
        """
        x = conv_block(inputs, name=name, **kwargs)
        return x

    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        """ Create decoder from a base_class model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name
        kwargs : dict
            parameters for ``upsample`` method

        Returns
        -------
        list of tf.Tensor
        """
        steps = kwargs.get('num_stages', len(inputs)-1)
        factor = kwargs.pop('factor')
        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor]
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        with tf.variable_scope(name):
            x = inputs[-1]
            for i in range(steps):
                x = cls.upsample(x, factor=factor[i], name='decoder-'+str(i), **kwargs)
        return x
