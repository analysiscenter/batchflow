""" Hieu Pham et al. "`Efficient Neural Architecture Search via Parameter Sharing
<https://arxiv.org/abs/1802.03268>`_"
"""

import tensorflow as tf

from . import TFModel
from .layers import conv_block


class NASNet_A(TFModel):
    """ NASNet-A

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`.TFModel._make_inputs`)

    body : dict
        parameters for all conv_blocks
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body'].update(dict(filters=16, pool_size=3, strides=1, pool_strides=1, depth_multiplier=1))
        config['head'].update(dict(layout='Vdf', dropout_rate=.5))
        config['loss'] = 'ce'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = self.num_classes('targets')
        config['head']['filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

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

    @classmethod
    def block(cls, inputs, name='block', **kwargs):
        """ Convolution cell

        Parameters
        ----------

        inputs : tuple of two tf.Tensor
            input tensors
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/block', **kwargs)

        with tf.variable_scope(name):
            h_prev, h = inputs

            s1_3 = conv_block(h, 'Cna', kernel_size=3, name='sep1_3', **kwargs)
            a = s1_3 + h

            s2_3 = conv_block(h_prev, 'Cna', kernel_size=3, name='sep2_3', **kwargs)
            s2_5 = conv_block(h, 'Cna', kernel_size=5, name='sep2_5', **kwargs)
            b = s2_3 + p2_5


            p2_3 = conv_block(h, 'v', pool_size=3, name='avg3_3', **kwargs)
            c = p2_3 + h_prev

            p3_3_1 = conv_block(h_prev, 'v', pool_size=3, name='avg3_3_1', **kwargs)
            p3_3_2 = conv_block(h_prev, 'v', pool_size=3, name='avg3_3_2', **kwargs)
            d = p3_3_1 + p3_3_2

            s4_5 = conv_block(h_prev, 'Cna', kernel_size=5, name='sep4_5', **kwargs)
            s4_3 = conv_block(h_prev, 'Cna', kernel_size=3, name='sep4_3', **kwargs)
            e = s4_5 + s4_3

            axis = cls.channels_axis(kwargs['data_format'])
            c = tf.concat([a, b, c, d, e], axis=axis)

        return c

    @classmethod
    def reduction(cls, inputs, name='reduction', **kwargs):
        """ Reduction cell

        Parameters
        ----------

        inputs : tuple of two tf.Tensor
            input tensors
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('body/reduction', **kwargs)

        with tf.variable_scope(name):
            h_prev, h = inputs

            s1_5 = conv_block(h_prev, 'Cna', kernel_size=5, name='sep1_5', **kwargs)
            p1_5 = conv_block(h, 'v', pool_size=3, name='avg1_3', **kwargs)
            a = s1_5 + p1_5

            s2_3 = conv_block(h, 'Cna', kernel_size=3, name='sep2_3', **kwargs)
            p2_3 = conv_block(h, 'v', pool_size=3, name='avg2_3', **kwargs)
            b = s2_3 + p2_3

            s3_3 = conv_block(h, 'Cna', kernel_size=3, name='sep3_3', **kwargs)
            p3_3 = conv_block(h, 'v', pool_size=3, name='avg3_3', **kwargs)
            c = s3_3 + p3_3

            s4_5 = conv_block(c, 'Cna', kernel_size=5, name='sep4_5', **kwargs)
            p4_3 = conv_block(h, 'v', pool_size=3, name='avg4_3', **kwargs)
            c = s4_5 + p4_5

            s5_3 = conv_block(c, 'Cna', kernel_size=3, name='sep5_3', **kwargs)
            s5_5 = conv_block(h_prev, 'Cna', kernel_size=5, name='sep5_5', **kwargs)
            c = s5_3 + s5_5

            axis = cls.channels_axis(kwargs['data_format'])
            c = tf.concat([a, b, c], axis=axis)

        return c
