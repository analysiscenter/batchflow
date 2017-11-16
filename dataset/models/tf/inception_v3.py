""" Contains class for Inception_v3 """
import tensorflow as tf

from . import TFModel
from .layers import conv_block


_DEFAULT_V3_ARCH = {
    'b': {'filters': [[64, 48, 96, 32], [64, 48, 96, 64], [64, 48, 96, 64]]},
    'r': dict(pool_size=3, pool_strides=2, padding='valid', filters=(384, 64, 96)),
    'f': {'filters': [[192, 128],
                      [192, 160],
                      [192, 160],
                      [192, 192]]},
    'm': {'filters': (192, 320)},
    'e': {'filters': [320, 384, 448, 192]}
}

class Inception_v3(TFModel):
    """ The base Inception_v3 model

    References
    ----------
    .. Christian Szegedy et al. "Rethinking the Inception Architecture for Computer Vision"
       Argxiv.org `<https://arxiv.org/abs/1512.00567>`_

    ** Configuration **

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict

    body : dict
        arch : dict
            parameters for each block.
            key - short name block
            item - list or dict with parameters

    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['input_block'].update(dict(layout='cna cna cnap cna cnap', filters=[32, 32, 64, 80, 192],
                                          kernel_size=[3, 3, 3, 1, 3], strides=[2, 1, 1, 1, 1],
                                          pool_size=3, pool_strides=2, padding='valid'))
        config['body']['layout'] = 'bbbrffffmee'
        config['body']['arch'] = _DEFAULT_V3_ARCH
        config['head'].update(dict(layout='Vdf', dropout_rate=.8))

        return config

    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['common']['data_format'] = self.data_format('images')
        config['input_block']['inputs'] = self.inputs['images']
        config['head']['units'] = self.num_classes('labels')
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of blocks:
            b - block (figure 5 in paper)
            r - reduction_block (figure 6 in paper)
            f - factor_block (figure 10 in paper)
            m - mixed_block (not figure in paper, but describe in 6 section)
            e - expanded_block (figure 7 in paper)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
             """
        kwargs = cls.fill_params('body', **kwargs)
        arch = kwargs.pop('arch')
        layout = kwargs.pop('layout')

        with tf.variable_scope(name):
            x = inputs

            layout_dict = {}
            for layer in layout:
                if layer not in layout_dict:
                    layout_dict[layer] = [-1, 0]
                layout_dict[layer][1] += 1

            for i, block in enumerate(layout):
                layout_dict[block][0] += 1
                block_no = layout_dict[block][0]

                filters = arch[block].get('filters')
                if isinstance(filters, list):
                    filters = filters[block_no]

                if block == 'b':
                    x = cls.block(x, filters, name='block-%d'%i, **kwargs)
                elif block == 'r':
                    x = cls.reduction_block(x, filters, name='reduction_block-%d'%i, **kwargs)
                elif block == 'f':
                    x = cls.factor_block(x, filters, name='factor_block-%d'%i, **kwargs)
                elif block == 'm':
                    x = cls.mixed_block(x, filters, name='mixed_block-%d'%i, **kwargs)
                elif block == 'e':
                    x = cls.expanded_block(x, filters, name='expanded_block-%d'%i, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, filters, layout='cna', name='block', **kwargs):
        """ Network building block
        For details see figure 5 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 4 ints
            number of output filters
        layout : str
            a sequence of layers (see :meth:'.conv_block`)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            branch_1_3 = conv_block(inputs, [filters[1], filters[0]], [1, 3], layout*2, name='conv_1_3', **kwargs)

            branch_1_3_3 = conv_block(inputs, [filters[0]]+[filters[2]]*2, [1, 3, 3], layout*3, name='conv_1_3_3',
                                      **kwargs)

            branch_pool = conv_block(inputs, filters[3], 1, 'p'+layout, 'c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_3, branch_1_3_3, branch_pool], axis=axis, name='output')
        return output

    @classmethod
    def reduction_block(cls, inputs, filters, layout='cna', name='reduction_block', **kwargs):
        """ Reduction block
        For details see figure 6 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 3 ints
            number of output filters
        layout : str
            a sequence of layers (see :meth:'.conv_block`)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_3 = conv_block(inputs, filters[0], 3, layout, name='conv_3', strides=2, padding='valid', **kwargs)

            branch_1_3 = conv_block(inputs, [filters[1]]+[filters[2]], [1, 3], layout*2, name='conv_1_3', **kwargs)
            branch_1_3_3 = conv_block(branch_1_3, filters[2], 3, layout, name='conv_1_3_3', strides=2,
                                      padding='valid', **kwargs)

            branch_pool = conv_block(inputs, layout='p', pool_size=3, pool_strides=2, name='max_pooling',
                                     padding='valid', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])

            output = tf.concat([branch_3, branch_1_3_3, branch_pool], axis, name='output')
        return output

    @classmethod
    def mixed_block(cls, inputs, filters, layout='cna', name='mixed_block', **kwargs):
        """ Mixed block
        For details see section 6 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 2 ints
            number of output filters
        layout : str
            a sequence of layers (see :meth:'.conv_block`)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)
            branch_1_3 = conv_block(branch_1, filters[1], 3, layout, name='conv_1_3', strides=2,
                                    padding='valid', **kwargs)

            branch_1_7 = conv_block(inputs, filters[0], [1, [1, 7], [7, 1]], layout*3, name='conv_1_7', **kwargs)
            branch_1_7_3 = conv_block(branch_1_7, filters[0], 3, layout, name='conv_1_7_3', strides=2,
                                      padding='valid', **kwargs)

            branch_pool = conv_block(inputs, layout='p', name='c_pool', padding='valid', **kwargs)

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1_3, branch_1_7_3, branch_pool], axis, name='output')
        return output

    @classmethod
    def factor_block(cls, inputs, filters, layout='cna', name='factor_block', **kwargs):
        """ 7x7 factorization block
        For details see figure 10 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuple of 3 int
            number of output filters
        layout : str
            a sequence of layers (see :meth:'.conv_block`)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            factor = [1, 7], [7, 1]
            kernel_size = [1, *factor]
            branch_1_3 = conv_block(inputs, [filters[1]] * 2 + [filters[0]], kernel_size, layout * 3,
                                    name='conv_1_3', **kwargs)

            kernel_size = [1, *factor * 2]
            branch_1_7 = conv_block(inputs, [filters[1]] * 4 + [filters[0]], kernel_size, layout * 5,
                                    name='conv_1_7', **kwargs)

            branch_pool = conv_block(inputs, filters[0], 1, 'p'+layout, name='c_pool', **{**kwargs, 'pool_strides': 1})

            axis = cls.channels_axis(kwargs['data_format'])
            output = tf.concat([branch_1, branch_1_3, branch_1_7, branch_pool], axis=axis, name='output')
        return output

    @classmethod
    def expanded_block(cls, inputs, filters, layout='cna', name='expanded_block', **kwargs):
        """ Network building block
        For details see figure 7 in the article.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : tuole of 4 ints
            number of output filters
        layout : str
            a sequence of layers (see :meth:'.conv_block`)
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            axis = cls.channels_axis(kwargs['data_format'])
            branch_1 = conv_block(inputs, filters[0], 1, layout, name='conv_1', **kwargs)

            branch_pool = conv_block(inputs, filters[3], 1, layout='p'+layout, name='c_pool',
                                     **{**kwargs, 'pool_strides': 1})

            branch_a1 = conv_block(inputs, filters[1], 1, layout, name='conv_a1', **kwargs)
            branch_a1_31 = conv_block(branch_a1, filters[1], [3, 1], layout, name='conv_1_31', **kwargs)
            branch_a1_13 = conv_block(branch_a1, filters[1], [1, 3], layout, name='conv_1_13', **kwargs)
            branch_a = tf.concat([branch_a1_31, branch_a1_13], axis=axis)

            branch_b13 = conv_block(inputs, [filters[2], filters[1]], [1, 3], layout*2, name='conv_b13', **kwargs)
            branch_b13_31 = conv_block(branch_b13, filters[1], [3, 1], layout, name='conv_b13_31', **kwargs)
            branch_b13_13 = conv_block(branch_b13, filters[1], [1, 3], layout, name='conv_b13_13', **kwargs)
            branch_b = tf.concat([branch_b13_31, branch_b13_13], axis=axis)

            output = tf.concat([branch_1, branch_pool, branch_a, branch_b], axis=axis, name='output')
        return output
