""" Contains inception_v1 network: https://arxiv.org/abs/1409.4842 """
import tensorflow as tf

from .tf_model import TFModel
from .layers import conv_block
from .layers.pooling import max_pooling, global_average_pooling

FILTERS = [
    [[64, 96, 128, 16, 32, 32],
     [128, 128, 192, 32, 96, 64]],
    [[192, 96, 208, 16, 48, 64],
     [160, 112, 224, 24, 64, 64],
     [128, 128, 256, 24, 64, 64],
     [112, 144, 288, 32, 64, 64],
     [256, 160, 320, 32, 128, 128]],
    [[256, 160, 320, 32, 128, 128],
     [384, 192, 384, 48, 128, 128]]]

NAMES = ['a', 'b', 'c', 'd', 'e']

class InceptionV1(TFModel):
    """ implementation of inception_v1 model

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)

    dim: int {1, 2, 3}
        spatial dimension of input without the number of channels

    batch_norm : bool
        if True enable batch normalization layers

    data_format: str {'channels_last', 'channels_first'}

    dropout_rate: float

    head_type: str {'dense', 'conv'}
         determine head of model

         'dense' - consist of GAP and dense layers

         'conv' - consist of conv 1x1 and GAP layers

    """
    def _build(self, *args, **kwargs):
        _ = args

        dim = self.get_from_config('dim', 2)
        b_norm = self.get_from_config('batch_norm', True)
        dropout_rate = self.get_from_config('dropout_rate', 0)
        head_type = self.get_from_config('head_type', 'dense')

        names = ['images', 'labels']
        _, transformed_placeholders = self._make_inputs(names)

        data_format = self.data_format('images')
        num_classes = self.num_classes('labels')

        max_pool = {'pool_size': 3,
                    'strides': 2,
                    'padding': 'same',
                    'data_format': data_format}

        batch_norm = {'training': self.is_training}
        conv = {'data_format': data_format}

        kwargs = {'max_pooling': max_pool,
                  'batch_norm': batch_norm,
                  'conv': conv,
                  'data_format': data_format,
                  'is_training': self.is_training}

        with tf.variable_scope('inception'):
            net = self.body(dim, transformed_placeholders['images'], b_norm, **kwargs)
            net = self.head(dim, net, num_classes, head_type, data_format, self.is_training, dropout_rate)

        self.statistic(tf.identity(net, name='predictions'), transformed_placeholders['labels'])

    @staticmethod
    def body(dim, inputs, b_norm, **kwargs):
        """ Building block for inception_v1 network

        Parameters
        ----------
        dim: int

        inputs: tf.Tensor

        b_norm: bool:
            if True enable batch normalization

        Returns
        -------
        net: tf.Tensor
        """
        layout = 'cnpcncnp' if b_norm else 'cpccp'
        with tf.variable_scope('body'):
            net = conv_block(dim=dim, inputs=inputs, filters=[64, 64, 192], kernel_size=[7, 3, 3], strides=[2, 1, 1],\
                             layout=layout, name='conv', **kwargs)
            for i, filters in enumerate(FILTERS):
                length = len(filters)
                for name, filt in zip(NAMES[:length], filters):
                    net = InceptionV1.block(dim=dim, inputs=net, filters=filt, name=str(i+3)+name,\
                                            b_norm=b_norm, **kwargs)
                if i != 2:
                    net = max_pooling(dim=dim, inputs=net, **kwargs['max_pooling'])
        return net

    @staticmethod
    def head(dim, inputs, n_outputs, head_type='dense', data_format='channels_last', is_training=True,\
             dropout_rate=0):
        """ Head of network.
        Consist of two kinds 'dense' and 'conv'.

        Parameters
        ----------
        dim: int

        input: tf.Tensor

        n_outputs: int
            number of outputs parameters

        head_type: str {'dense', 'conv'}

        data_format: str {'channels_last', 'channels_first'}

        is_training: bool

        dropout_rate: float

        Returns
        -------
        net: tf.Tensor
        """
        with tf.variable_scope('head'):
            if head_type == 'dense':
                net = global_average_pooling(dim=dim, inputs=inputs, data_format=data_format)
                if dropout_rate:
                    net = tf.layers.dropout(net, dropout_rate, training=is_training)
                net = tf.layers.dense(net, n_outputs)

            elif head_type == 'conv':
                net = conv_block(dim=dim, inputs=inputs, filters=n_outputs, kernel_size=1,\
                                     layout='c', name='conv_1', data_format=data_format)
                net = global_average_pooling(dim=dim, inputs=net, data_format=data_format)
            else:
                raise ValueError("Head_type should be dense or conv, but given %d" % head_type)
        return net

    def statistic(self, net, targets):
        """ Added to graph some useful funstion like accuracy or preidctions
        Parameters
        ----------
        net: tf.Tensor
        Network output

        targets: tf.Tensor
        Answers on the data
        """
        prob = tf.nn.softmax(net, name='prob_predictions')

        labels_hat = tf.cast(tf.argmax(prob, axis=1), tf.float32, name='predicted_labels')
        labels = tf.cast(tf.argmax(targets, axis=1), tf.float32, name='target_labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

    @staticmethod
    def block(dim, inputs, filters, data_format='channels_last', name=None, b_norm=True, is_training=True, **kwargs):
        """ Function contains building block from inception_v1 achitecture

        Parameters
        ----------
        dim: int
        spacial dimension of input without the number of channels

        inputs: tf.Tensor

        filters: list with 6 items:

        - number of filters in one conv 1x1

        - number of filters in conv 1x1 going before conv 3x3

        - number of filters in conv 3x3

        - number of filters in conv 1x1 going before conv 5x5,

        - number of filters in conv 5x5,

        - number of filters in conv 1x1 going after max_pool

        data_format: str {'channels_last', 'channels_first'}

        name: str
        name of block

        batch_norm: bool
        Use batch norm or not

        Returns
        -------
        tf.Tensor - output tf.Tensor
        """
        layout = 'cn' if b_norm else 'c'
        with tf.variable_scope("block-" + name):
            block_1 = conv_block(dim=dim, inputs=inputs, filters=filters[0], kernel_size=1,\
                                 layout=layout, name='conv_1', **kwargs)

            block_3 = conv_block(dim=dim, inputs=inputs, filters=[filters[1], filters[2]], \
                                 kernel_size=[1, 3], layout=layout*2, name='conv_3', **kwargs)

            block_5 = conv_block(dim=dim, inputs=inputs, filters=[filters[3], filters[4]], \
                                 kernel_size=[1, 5], layout=layout*2, name='conv_5', **kwargs)

            conv_pool = conv_block(dim=dim, inputs=inputs, filters=filters[5], kernel_size=1,\
                                   layout='p'+layout, name='c_pool', data_format=data_format, pool_size=3,\
                                   pool_strides=1, is_training=is_training)
            axis = -1 if data_format == 'channels_last' else 1
        return tf.concat([block_1, block_3, block_5, conv_pool], axis, name='output')
