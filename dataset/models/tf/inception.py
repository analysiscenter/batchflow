""" Contains inception_v1 network: https://arxiv.org/abs/1409.4842 """
import tensorflow as tf


from . import TFModel
from .layers import conv_block


_DEFAULT_BODY_ARCH = [
    {'conv': [[64, 96, 128, 16, 32, 32],
              [128, 128, 192, 32, 96, 64]],
     'max_pooling': dict(pool_size=3, pool_strides=2)},

    {'conv': [[192, 96, 208, 16, 48, 64],
              [160, 112, 224, 24, 64, 64],
              [128, 128, 256, 24, 64, 64],
              [112, 144, 288, 32, 64, 64],
              [256, 160, 320, 32, 128, 128]],
     'max_pooling': dict(pool_size=3, pool_strides=2)},

    {'conv': [[256, 160, 320, 32, 128, 128],
              [384, 192, 384, 48, 128, 128]]}
]


class Inception_v1(TFModel):
    """ Inception network, version 1

    References
    ----------
    .. Szegedy C. et al "Going Deeper with Convolutions"
       Arxiv.org `<https://arxiv.org/abs/1409.4842>`_

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    """
    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')

        config['input_block'] = {**dict(layout='cnp cn cn p', filters=[64, 64, 192],
                                        kernel_size=[7, 3, 3], strides=[2, 1, 1], pool_size=3),
                                 **config['input_block']}
        config['input_block']['inputs'] = self.inputs['images']

        config['body']['arch'] = self.get_from_config('body/arch', _DEFAULT_BODY_ARCH)

        config['head'] = {**dict(layout='Vdf', units=self.num_classes('labels'), dropout_rate=.4),
                          **config['head']}
        return config

    @classmethod
    def body(cls, inputs, arch, **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        arch : list of dict
            network architecture

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(kwargs.get('name', 'body')):
            x = inputs
            for i, group_cfg in enumerate(arch):
                with tf.variable_scope('block-%d' % i):
                    for j, filters in enumerate(group_cfg['conv']):
                        x = cls.block(x, filters, name='module-%d' % j, **kwargs)
                    if 'max_pooling' in group_cfg:
                        x = conv_block(x, layout='p', name='max-pooling', **kwargs)
        return x

    @classmethod
    def block(cls, inputs, filters, layout='cn', name=None, **kwargs):
        """ Function contains building block from inception_v1 achitecture

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : list with 6 items:
            - number of filters in 1x1 conv
            - number of filters in 1x1 conv going before conv 3x3
            - number of filters in 3x3 conv
            - number of filters in 1x1 conv going before conv 5x5,
            - number of filters in 5x5 conv,
            - number of filters in 1x1 conv going before max-pooling
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            branch_3 = conv_block(inputs, [filters[1], filters[2]], [1, 3], layout*2, name='conv_3', **kwargs)

            branch_5 = conv_block(inputs, [filters[3], filters[4]], [1, 5], layout*2, name='conv_5', **kwargs)

            branch_pool = conv_block(inputs, filters[5], 1, 'p'+layout, 'conv_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_3, branch_5, branch_pool], axis, name='output')
        return output
