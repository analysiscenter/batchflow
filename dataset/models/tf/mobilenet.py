''' Contains class for MobileNet '''

import tensorflow as tf

from . import TFModel
from .layers import conv_block


_DEFAULT_BODY = {'strides': [1, 2, 1, 2, 1, 2,
                             1, 1, 1, 1, 1,
                             2, 2],
                 'double_filters': [True, True, False, True, False, True,
                                    False, False, False, False, False,
                                    True, False]}


class MobileNet(TFModel):
    """ MobileNet

    References
    ----------
        Howard A. G. et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
        Arxiv.org `<https://arxiv.org/abs/1704.04861>`_

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)

    input_block : dict

    strides : list of int
        strides in separable convolutions

    double_filters : list of bool
        if True number of filters in 1x1 covolution will be doubled

    width_factor : float
        multiplier for number of channels

    resolution_factor : float
        multiplier for spatial resolution

    head : dict

    """


    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['input_block']['inputs'] = self.inputs['images']

        input_block = self.get_from_config('input_block', {'layout': 'cna', 'filters': 32,
                                                           'kernel_size' : 3, 'strides': 2})

        config['input_block']['width_factor'] = self.get_from_config('width_factor', 1.0)
        config['input_block']['resolution_factor'] = self.get_from_config('resolution_factor', 1.0)


        config['input_block'] = {**input_block,
                                 **config['input_block']}

        config['default']['data_format'] = self.data_format('images')

        config['body']['strides'] = self.get_from_config('strides', _DEFAULT_BODY['strides'])
        config['body']['double_filters'] = self.get_from_config('double_filters',
                                                                _DEFAULT_BODY['double_filters'])

        config['head'] = {**dict(layout='Vf', units=self.num_classes('labels')),
                          **config['head']}
        return config


    @classmethod
    def body(cls, inputs, strides, double_filters, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor
        strides : list of int
            strides in separable convolutions
        double_filters : list of bool
            if True number of filters in 1x1 covolution will be doubled
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """

        with tf.variable_scope(name):
            x = inputs
            for index in range(len(strides)):
                x = cls.block(x, strides[index], double_filters[index], 'block-'+str(index), **kwargs)
            return x


    @classmethod
    def block(cls, inputs, strides, double_filters=False, name=None, **kwargs):
        """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

        Parameters
        ----------

        inputs : tf.Tensor
            input tensor
        strides : int
            strides in separable convolution
        double_filters : bool
            if True number of filters in 1x1 covolution will be doubled
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """

        data_format = kwargs.get('data_format')
        num_channels = cls.channels_shape(inputs, data_format)
        filters = [num_channels, num_channels*2] if double_filters else [num_channels]*2

        x = conv_block(inputs, filters, [3, 1], 'sna cna', name, [strides, 1], **kwargs)
        return x


    @classmethod
    def input_block(cls, inputs, filters=32, width_factor=1, resolution_factor=1, name='input_block', **kwargs):
        """ Transform inputs with a convolution block

        Parameters
        ----------
        filters : int
            number of filters in convolutional layer
        width_factor : float
            multiplier for number of channels
        resolution_factor : float
            multiplier for spatial resolution

        kwargs : dict
            See :func:`.layers.conv_block`.

        Returns
        -------
        tf.Tensor

        """

        filters = filters*width_factor

        data_format = kwargs.get('data_format')

        initial_shape = cls.spatial_shape(inputs, data_format)
        new_shape = [int(size*resolution_factor) for size in initial_shape]

        x = tf.image.resize_images(inputs, new_shape)
        return conv_block(x, filters, name=name, **kwargs)
