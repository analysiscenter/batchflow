""" Contains inception_v1 network: https://arxiv.org/abs/1409.4842 """
import tensorflow as tf


from . import TFModel
from .layers import conv_block


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
    """ Base Inception v1 neural network
    https://arxiv.org/abs/1409.4842 (C.Szegedy at al, 2014)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    use_batch_norm : bool or dict
        parameters for batch normalization layers.
        If False, remove batch norm layers whatsoever.
        Default is ``{'momentum': 0.1}``.
    data_format : str {'channels_last', 'channels_first'}
    dropout_rate : float
        parameter for dropout in head of model.
        if 0. dropout off.
    """
    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')
        if not self.get_from_config('use_batch_norm', True):
            config['default'].pop('batch_norm')

        config['input_block']['inputs'] = self.inputs['images']
        config['input_block']['filters'] = self.get_from_config('input_block/filters', [64, 64, 192])
        config['input_block']['kernel_size'] = self.get_from_config('input_block/kernel_size', [7, 3, 3])
        config['input_block']['strides'] = self.get_from_config('input_block/strides', [2, 1, 1])

        config['body']['pool_size'] = self.get_from_config('body/pool_size', 3)
        config['body']['pool_strides'] = self.get_from_config('body/pool_strides', 2)
        config['body']['padding'] = self.get_from_config('body/padding', 'same')

        config['head']['layout'] = self.get_from_config('head/layout', 'Vdf')
        config['head']['units'] = self.num_classes('labels')
        config['head']['dropout_rate'] = self.get_from_config('head/dropout_rate', 0)

        return config

    @classmethod
    def input_block(cls, inputs, name='head', **kwargs):
        """ Three 3x3 convolution and two 3x3 max pooling with stride 2

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name: str
            scope name
        """
        layout = 'cnp cn cnp' if 'batch_norm' in kwargs else 'cp c cp'
        x = conv_block(inputs, name=name, **kwargs)
        return x

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Inception_v1 body

        Parameters
        ----------
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
                    net = cls.block(net, filt, name='block-'+str(i+3)+name, **kwargs)
                if i != 2:
                    net = conv_block(net, layout='p', name=str(i)+'_pooling', **kwargs)
        return net

    @classmethod
    def block(cls, inputs, filters, name=None, **kwargs):
        """ Function contains building block from inception_v1 achitecture

        Parameters
        ----------
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
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            branch_3 = conv_block(inputs, [filters[1], filters[2]], [1, 3], layout*2, name='conv_3', **kwargs)

            branch_5 = conv_block(inputs, [filters[3], filters[4]], [1, 5], layout*2, name='conv_5', **kwargs)

            branch_pool = conv_block(inputs, filters[5], 1, 'p'+layout, 'c_pool', **{**kwargs, 'pool_strides': 1})

            axis = -1 if kwargs['data_format'] == 'channels_last' else 1
            concat = tf.concat([branch_1, branch_3, branch_5, branch_pool], axis, name='output')
        return concat
