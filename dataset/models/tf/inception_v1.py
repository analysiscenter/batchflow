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
    """ Inception_v1
    https://arxiv.org/abs/1409.4842


    **Configuration**


    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    batch_norm : None or dict
        parameters for batch normalization layers.
        If None, remove batch norm layers whatsoever.
        Default is ``{'momentum': 0.1}``.
    data_format : str {'channels_last', 'channels_first'}
    dropout_rate : float
        parameter for dropout in head of model.
        if 0. dropout off.
    conv_block : dict
        parameters to convolutional layers in all network.
        As `kernel_size` or `filters` and etc.
    input_block : dict
        parameters to input block.
    body : dict
        config with parameters to body of network.
    head : dict
        config with parameters to head of network.
    """
    def _build(self):
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)


        num_classes = self.num_classes('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        batch_norm = self.get_from_config('batch_norm', {'momentum': 0.1})
        filters = self.get_from_config('filters', 64)
        dropout_rate = self.get_from_config('dropout_rate', 0)

        conv_block_config = self.get_from_config('conv_block', {})
        input_block_config = self.get_from_config('input_block', {'filters': [filters]*2+[filters*3],
                                                                  'kernel_size': [7, 3, 3],
                                                                  'strides': [2, 1, 1]})
        body_config = self.get_from_config('body', {})
        head_config = self.get_from_config('head', {'layout': 'Vdf',
                                                    'units': num_classes,
                                                    'dropout_rate': dropout_rate})
        head_config['num_classes'] = num_classes

        kwargs = {'pool_size': 3,
                  'pool_strides': 2,
                  'padding': 'same',
                  'data_format': data_format,
                  'is_training': self.is_training,
                  **conv_block_config}

        if batch_norm:
            kwargs['batch_norm'] = batch_norm

        with tf.variable_scope('inception'):
            layout = 'cnp cn cnp' if 'batch_norm' in kwargs else 'cp c cp'
            net = inputs['images']
            net = self.input_block(dim, net, name='input', layout=layout, **{**kwargs, **input_block_config})


            net = self.body(dim, net, **{**kwargs, **body_config})
            output = self.head(dim, net, **{**kwargs, **head_config})
        self.metrics(tf.identity(output, name='predictions'), inputs['labels'])

    @classmethod
    def body(cls, dim, inputs, **kwargs):
        """ Inception_v1 body


        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor


        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(kwargs.get('name', 'body')):
            net = inputs
            for i, filters in enumerate(BLOCK_FILTERS):
                length = len(filters)
                for name, filt in zip(BLOCK_NAMES[:length], filters):
                    net = cls.block(dim, net, filt, name='block-'+str(i+3)+name, **kwargs)
                if i != 2:
                    net = conv_block(dim, net, layout='p', name=str(i)+'_pooling', **kwargs)
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
    def block(dim, inputs, filters, name=None, **kwargs):
        """ Function contains building block from inception_v1 achitecture


        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : list with 6 items:

            - number of filters in one con

            - number of filters in conv 1x1 going before conv 3x3

            - number of filters in conv 3x3

            - number of filters in conv 1x1 going before conv 5x5,

            - number of filters in conv 5x5,

            - number of filters in conv 1x1 going
        name : str
            scope name


        Returns
        -------
        tf.Tensor
        """
        layout = 'cn' if 'batch_norm' in kwargs else 'c'
        with tf.variable_scope(name):
            block_1 = conv_block(dim, inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            block_3 = conv_block(dim, inputs, [filters[1], filters[2]], [1, 3], layout*2, name='conv_3', **kwargs)

            block_5 = conv_block(dim, inputs, [filters[3], filters[4]], [1, 5], layout*2, name='conv_5', **kwargs)

            conv_pool = conv_block(dim, inputs, filters[5], 1, 'p'+layout, 'c_pool', **{**kwargs, 'pool_strides': 1})

            axis = -1 if kwargs['data_format'] == 'channels_last' else 1
            concat = tf.concat([block_1, block_3, block_5, conv_pool], axis, name='output')
        return concat
