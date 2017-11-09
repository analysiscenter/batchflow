""" Contains inception_v1 network: https://arxiv.org/abs/1409.4842 """
import tensorflow as tf


from . import TFModel
from .layers import conv_block
from .layers.pooling import max_pooling

BLOCK_FILTERS = [
    [[64, 96, 128, 16, 32, 32],
     [128, 128, 192, 32, 96, 64]],
    [[192, 96, 208, 16, 48, 64],
     [160, 112, 224, 24, 64, 64],
     [128, 128, 256, 24, 64, 64],
     [112, 144, 288, 32, 64, 64],
     [256, 160, 320, 32, 128, 128]],
    [[256, 160, 320, 32, 128, 128],
     [384, 192, 384, 48, 128, 128]]]

BLOCK_NAMES = ['a', 'b', 'c', 'd', 'e']

class InceptionV1(TFModel):
    """ implementation of inception_v1 model

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)

    dim : int {1, 2, 3}
        spatial dimension of input without the number of channels

    batch_norm : bool
        if True enable batch normalization layers

    data_format : str {'channels_last', 'channels_first'}

    dropout_rate : float

    head_style: str {'dense', 'conv'}
         determine head of model

         'dense' - consist of GAP and dense layers

         'conv' - consist of conv 1x1 and GAP layers

    """
    def _build(self):
        dim = self.get_from_config('dim', 2)
        batch_norm = self.get_from_config('batch_norm', True)
        dropout_rate = self.get_from_config('dropout_rate', 0)
        style = self.get_from_config('head_style', 'dense')

        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        data_format = self.data_format('images')
        num_classes = self.num_classes('labels')

        max_pool = {'pool_size': 3,
                    'strides': 2,
                    'padding': 'same',
                    'data_format': data_format}

        conv = {'data_format': data_format}

        kwargs = {'max_pooling': max_pool,
                  'conv': conv,
                  'data_format': data_format,
                  'is_training': self.is_training}

        if batch_norm:
            batch_norm = {'training': self.is_training}
            kwargs['batch_norm'] = batch_norm

        with tf.variable_scope('inception'):
            net = self.body(dim, inputs['images'], **kwargs)
            net = self.head(dim, net, style, 'Vdf', num_classes, dropout_rate=dropout_rate)

        self.metrics(tf.identity(net, name='predictions'), inputs['labels'])

    @staticmethod
    def body(dim, inputs, **kwargs):
        """ Building block for inception_v1 network

        Parameters
        ----------
        dim : int

        inputs : tf.Tensor

        Returns
        -------
        net : tf.Tensor
        """
        layout = 'cnpcncnp' if 'batch_norm' in kwargs else 'cpccp'
        with tf.variable_scope('body'):
            net = conv_block(dim, inputs, [64, 64, 192], [7, 3, 3], strides=[2, 1, 1],\
                             layout=layout, name='conv', **kwargs)

            for i, filters in enumerate(BLOCK_FILTERS):
                length = len(filters)
                for name, filt in zip(BLOCK_NAMES[:length], filters):
                    net = InceptionV1.block(dim, net, filt, name=str(i+3)+name, **kwargs)
                if i != 2:
                    net = max_pooling(dim=dim, inputs=net, **kwargs['max_pooling'])
        return net

    def metrics(self, net, targets):
        """ Added to graph some useful funstion like accuracy or preidctions
        Parameters
        ----------
        net : tf.Tensor
            Network output

        targets : tf.Tensor
            Answers on the data
        """
        prob = tf.nn.softmax(net, name='predicted_proba')

        labels_hat = tf.cast(tf.argmax(prob, axis=1), tf.float32, name='predicted_labels')
        labels = tf.cast(tf.argmax(targets, axis=1), tf.float32, name='target_labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

    @staticmethod
    def block(dim, inputs, filters, data_format='channels_last', name=None, is_training=True, **kwargs):
        """ Function contains building block from inception_v1 achitecture

        Parameters
        ----------
        dim : int
            spacial dimension of input without the number of channels

        inputs : tf.Tensor

        filters : list with 6 items:

            - number of filters in one con

            - number of filters in conv 1x1 going before conv 3x3

            - number of filters in conv 3x3

            - number of filters in conv 1x1 going before conv 5x5,

            - number of filters in conv 5x5,

            - number of filters in conv 1x1 going

        data_format : str {'channels_last', 'channels_first'}

        name : str
            name of block

        is_training : bool

        Returns
        -------
            tf.Tensor - output tf.Tensor
        """
        layout = 'cn' if 'batch_norm' in kwargs else 'c'
        with tf.variable_scope("block-" + name):
            block_1 = conv_block(dim, inputs, filters[0], 1, layout=layout, name='conv_1', **kwargs)

            block_3 = conv_block(dim, inputs, [filters[1], filters[2]], [1, 3], layout=layout*2,\
                                 name='conv_3', **kwargs)

            block_5 = conv_block(dim, inputs, [filters[3], filters[4]], [1, 5], layout=layout*2,\
                                 name='conv_5', **kwargs)

            conv_pool = conv_block(dim, inputs, filters[5], 1, 'p'+layout, 'c_pool', pool_size=3,\
                                   pool_strides=1, is_training=is_training, data_format=data_format)

            axis = -1 if data_format == 'channels_last' else 1
        return tf.concat([block_1, block_3, block_5, conv_pool], axis, name='output')
