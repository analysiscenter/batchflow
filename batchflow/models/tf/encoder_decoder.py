"""  Encoder-decoder """

import tensorflow as tf

from .layers import conv_block
from . import TFModel
from .resnet import ResNet18


class EncoderDecoder(TFModel):
    """ Encoder-decoder architecture. Allows to combine features of different models,
    e.g. ResNet and DenseNet, in order to create new ones with just a few lines of code.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
            base_class : TFModel
                Model implementing ``make_encoder`` method which returns tensors
                with encoded representation of the inputs. Defaults to ResNet18.
            other args
                Parameters for base class ``make_encoder`` method.

        embedding : dict
            :func:`~.layers.conv_block` parameters for the bottom block.

        decoder : dict
            num_stages : int
                Number of upsampling blocks.
            factor : int or list of int
                If int, the total upsampling factor for all blocks combined.
                If list, upsampling factors for each stage.
            bridges : bool
                Whether to concatenate upsampled tensor with stored pre-downsample encoding.
            use_post : bool
                Whether to post-process tensors after upsampling.
            layout : str
                Upsampling method (see :func:`~.layers.upsample`).
            block : dict
                Parameters for post-processing blocks.
            other args
                Parameters for :func:`~.layers.upsample`.

    Examples
    --------
    Use ResNet18 as an encoder (which by default downsamples the image with a factor of 8),
    create an embedding that contains 16 channels,
    and build a decoder with 3 upsampling stages to scale the embedding 8 times with transposed convolutions::

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape'), 'name': 'targets'}),
            'initial_block/inputs': 'images',
            'encoder/base_class': ResNet18,
            'embedding/filters': 16,
            'decoder': dict(num_stages=3, factor=8, layout='tna')
        }


    Preprocess input image with 7x7 convolutions, use DenseNet as encoder to downsample the image
    with a factor of 16 with every DenseBlock concatenating its input to the output, don't change
    shape of resulted tensor in embedding, decode compressed image to initial shape by alternating
    ResNeXt and DenseNet blocks with desired parameters:

    >>> config = {
            'inputs': dict(images={'shape': B('image_shape')},
                           masks={'name': 'targets', 'shape': B('image_shape')}),
            'initial_block': {'inputs': 'images',
                              'layout': 'cna', 'filters': 4, 'kernel_size': 7},
            'body/encoder': {'base_class': DenseNet,
                             'num_layers': [2, 2, 3, 3, 4],
                             'block/growth_rate': 6, 'block/skip': True},
            'body/decoder': {'num_stages': 4, 'factor': 16,
                             'bridges': True,
                             'block': {'base_block':[ResNet.block, DenseNet.block, ResNet.block, DenseNet.block],
                                       'layout': ['cnacna', 'nacd', 'cnacna', 'nacd']
                                       'filters': [256, None, 64, None], 'resnext': True,
                                       'num_layers': [None, 4, None, 2], 'growth_rate': 6, 'skip': False}},
        }

    Notes
    -----
    Downsampling is done one less time than the length of `filters` (or other size-defining parameter) list in the
    `encoder` configuration. That is due to the fact that the first block is used as preprocessing of input tensors.
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body']['encoder'] = dict(base_class=ResNet18)
        config['body']['embedding'] = dict(layout='cna', kernel_size=1)
        config['body']['decoder'] = dict(layout='tna', factor=8, num_stages=3, bridges=False, use_post=False)
        config['body']['decoder']['block'] = dict(base_block=conv_block, layout='cna', activation=tf.nn.relu)
        config['head'] = dict(layout='c', kernel_size=1)
        return config


    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['targets'] = self.targets
        return config


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Create encoder, embedding and decoder. """
        kwargs = cls.fill_params('body', **kwargs)
        encoder = kwargs.pop('encoder')
        decoder = kwargs.pop('decoder')
        embedding = kwargs.pop('embedding')

        with tf.variable_scope(name):
            # Encoder: transition down
            encoder_outputs = cls.encoder(inputs, **encoder, **kwargs)

            # Bottleneck: working with compressed representation
            x = cls.embedding(encoder_outputs[-1], **embedding, **kwargs)
            encoder_outputs.append(x)

            #Decoder: transition up
            x = cls.decoder(encoder_outputs, **decoder, **kwargs)
        return x


    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        """ Linear convolutions with kernel size of 1. """
        kwargs = cls.fill_params('head', **kwargs)
        with tf.variable_scope(name):
            x = cls.crop(inputs, targets, kwargs['data_format'])
            channels = cls.num_channels(targets)
            x = conv_block(x, filters=channels, **kwargs)
        return x


    @classmethod
    def encoder(cls, inputs, name='encoder', **kwargs):
        """ Create encoder from a `base_class` model

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        base_class : TFModel
            Model class (default is ResNet18). Should implement ``make_encoder`` method.

        name : str
            Scope name.

        kwargs : dict
            Parameters for ``make_encoder`` method.

        Returns
        -------
        list of tf.Tensors
        """
        base_class = kwargs.pop('base_class')
        x = base_class.make_encoder(inputs, name=name, **kwargs)
        return x


    @classmethod
    def embedding(cls, inputs, name='embedding', **kwargs):
        """ Create embedding from inputs tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        name : str
            Scope name.

        kwargs : dict
            Parameters for :func:`~.tf.layers.conv_block`.

        Returns
        -------
        tf.Tensor
        """
        kwargs['filters'] = kwargs.get('filters') or cls.num_channels(inputs)
        if kwargs.get('layout') is not None:
            x = conv_block(inputs, name=name, **kwargs)
        else:
            x = inputs
        return x


    @classmethod
    def decoder(cls, inputs, name='decoder', **kwargs):
        """ Create decoder with a given number of upsampling stages.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        name : str
            Scope name.

        bridges : bool
            Whether to concatenate upsampled tensor with stored pre-downsample encoding.

        use_post : bool
            Whether to post-process tensors after upsampling.

        block : dict
            Parameters for post-processing blocks.

        kwargs : dict
            Parameters for ``upsample`` method.

        Returns
        -------
        tf.Tensor
        """
        steps = kwargs.pop('num_stages', len(inputs)-2)
        factor, bridges, use_post = cls.pop(['factor', 'bridges', 'use_post'], kwargs)

        block_args = kwargs.pop('block') if use_post else None

        if isinstance(factor, int):
            factor = int(factor ** (1/steps))
            factor = [factor] * steps
        elif not isinstance(factor, list):
            raise TypeError('factor should be int or list of int, but %s was given' % type(factor))

        with tf.variable_scope(name):
            x = inputs[-1]

            axis = cls.channels_axis(kwargs.get('data_format'))
            for i in range(steps):
                with tf.variable_scope('decoder-'+str(i)):
                    # Upsample by a desired factor
                    x = cls.upsample(x, factor=factor[i], name='upsample', **kwargs)

                    # Post-process resulting tensor
                    if block_args is not None:
                        args = {key: value[i] for key, value in block_args.items()
                                if isinstance(value, list)}
                        args = {**kwargs, **block_args, **args} # enforce priority of subkeys
                        args = {key: value for key, value in args.items() if value is not None}
                        base_block = args.get('base_block')
                        x = base_block(x, name='post', **args)

                    # Concatenate it with stored encoding of the ~same shape
                    if bridges and (i < len(inputs)-3):
                        x = cls.crop(x, inputs[-i-3], data_format=kwargs.get('data_format'))
                        x = tf.concat((x, inputs[-i-3]), axis=axis, name='bridges-concat')
        return x
