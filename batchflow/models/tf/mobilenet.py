""" Howard A. et al. "`MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
<https://arxiv.org/abs/1704.04861>`_"

Sandler M. et al. "`Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation
<https://arxiv.org/abs/1801.04381>`_"

Howard A. et al. "`Searching for MobileNetV3
<https://arxiv.org/abs/1905.02244>`_"
"""
from copy import deepcopy
import tensorflow as tf

from . import TFModel
from .layers import conv_block
from .nn import h_swish, h_sigmoid

_V1_DEFAULT_BODY = {
    'strides': [1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2],
    'double_filters': [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    'width_factor': 1
}

_V2_DEFAULT_BODY = [
    dict(repeats=1, filters=16, expansion_factor=1, strides=1),
    dict(repeats=2, filters=24, expansion_factor=6, strides=2),
    dict(repeats=3, filters=32, expansion_factor=6, strides=2),
    dict(repeats=4, filters=64, expansion_factor=6, strides=2),
    dict(repeats=3, filters=96, expansion_factor=6, strides=1),
    dict(repeats=3, filters=160, expansion_factor=6, strides=2),
    dict(repeats=1, filters=320, expansion_factor=6, strides=1),
]

_V3_LARGE_DEFAULT_BODY = [
    dict(repeats=1, filters=16, expansion_factor=1, strides=1, kernel_size=3, se_block=False),
    dict(repeats=1, filters=24, expansion_factor=4, strides=2, kernel_size=3, se_block=False),
    dict(repeats=1, filters=24, expansion_factor=3, strides=1, kernel_size=3, se_block=False,
         residual=True),
    dict(repeats=3, filters=40, expansion_factor=3, strides=2, kernel_size=5, se_block=True),
    dict(repeats=1, filters=80, expansion_factor=6, strides=2, kernel_size=3, activation=h_swish, se_block=False),
    dict(repeats=1, filters=80, expansion_factor=2.5, strides=1, kernel_size=3, activation=h_swish, se_block=False,
         residual=True),
    dict(repeats=2, filters=80, expansion_factor=2.3, strides=1, kernel_size=3, activation=h_swish, se_block=False,
         residual=True),
    dict(repeats=2, filters=112, expansion_factor=6, strides=1, kernel_size=3, activation=h_swish, se_block=True),
    dict(repeats=3, filters=160, expansion_factor=6, strides=2, kernel_size=5, activation=h_swish, se_block=True),
]

_V3_SMALL_DEFAULT_BODY = [
    dict(repeats=1, filters=16, expansion_factor=1, strides=2, kernel_size=3, se_block=True),
    dict(repeats=1, filters=24, expansion_factor=4.5, strides=2, kernel_size=3, se_block=False),
    dict(repeats=1, filters=24, expansion_factor=3.7, strides=1, kernel_size=3, se_block=False,
         residual=True),
    dict(repeats=1, filters=40, expansion_factor=4, strides=2, kernel_size=5, activation=h_swish, se_block=True),
    dict(repeats=2, filters=40, expansion_factor=6, strides=1, kernel_size=5, activation=h_swish, se_block=True,
         residual=True),
    dict(repeats=2, filters=48, expansion_factor=3, strides=1, kernel_size=5, activation=h_swish, se_block=True),
    dict(repeats=3, filters=96, expansion_factor=6, strides=2, kernel_size=5, activation=h_swish, se_block=True),
]


class MobileNet_v1(TFModel):
    """ MobileNet

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`~.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block (default is 'cna', 32, 3, strides=2)

    body : dict
        strides : list of int
            strides in separable convolutions

        double_filters : list of bool
            if True, number of filters in 1x1 covolution will be doubled

        width_factor : float
            multiplier for the number of channels (default=1)
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['initial_block'] = dict(layout='cna', filters=32, kernel_size=3, strides=2)
        config['body'].update(_V1_DEFAULT_BODY)
        config['head'].update(dict(layout='Vf'))

        config['loss'] = 'ce'

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')
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
        sep_strides, double_filters, width_factor = \
            cls.pop(['strides', 'double_filters', 'width_factor'], kwargs)

        with tf.variable_scope(name):
            x = inputs
            for i, strides in enumerate(sep_strides):
                x = cls.block(x, strides=strides, double_filters=double_filters[i], width_factor=width_factor,
                              name='block-%d' % i, **kwargs)
        return x

    @classmethod
    def block(cls, inputs, strides=1, double_filters=False, width_factor=1, name=None, **kwargs):
        """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        strides : int
            strides in separable convolution
        double_filters : bool
            if True number of filters in 1x1 covolution will be doubled
        width_factor : float
            multiplier for the number of filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        num_filters = int(cls.num_channels(inputs, kwargs.get('data_format')) * width_factor)
        filters = [num_filters, num_filters*2] if double_filters else num_filters
        return conv_block(inputs, 'Cna cna', filters, [3, 1], name=name, strides=[strides, 1], **kwargs)



class MobileNet(TFModel):
    """ Base class for MobileNets after v2.
    Default configuration is MobileNet_v2.
    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`.TFModel._make_inputs`)

    initial_block : dict
        parameters for the initial block (default is 'cna', 32, 3, strides=2)

    body : dict
        layout : list of dict each consisting of parameters:
            repeats : int
                number of repeats of the block
            filters : int
                number of output filters in the block
            expansion_factor : int
                multiplier for the number of filters in internal convolutions
            strides : int
                stride for 3x3 convolution
            kernel_size : int
                kernel size for depthwise convolution
            activation : callable, optional
            se_block : bool or dict
                whether to include squeeze and excitation block
                If dict it must contain ratio and activation keys
                to customize se_block parameters
            residual : bool
                whether to make a residual connection
        width_factor : float
            multiplier for the number of channels (default=1)
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['initial_block'].update(dict(layout='cna', filters=32, kernel_size=3, strides=2))
        config['body'].update(dict(width_factor=1, layout=deepcopy(_V2_DEFAULT_BODY)))
        config['head'].update(dict(layout='cnacnV', filters=[1280, 2], kernel_size=1))
        config['common'].update(dict(activation=tf.nn.relu6))
        config['loss'] = 'ce'
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        if isinstance(config['head/filters'], list):
            config['head/filters'][-1] = self.num_classes('targets')
        else:
            config['head/filters'] = self.num_classes('targets')
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
        width_factor, layout = cls.pop(['width_factor', 'layout'], kwargs)
        with tf.variable_scope(name):
            x = inputs
            i = 0
            for block in layout:
                repeats = block.pop('repeats')
                block['width_factor'] = width_factor
                for k in range(repeats):
                    if k > 0:
                        block['strides'] = 1
                    residual = block.pop('residual', False) or k > 0
                    x = cls.block(x, **block, residual=residual, name='block-%d' % i, **kwargs)
                    i += 1
        return x

    @classmethod
    def block(cls, inputs, filters, residual=False, strides=1, expansion_factor=4, width_factor=1, kernel_size=3,
              se_block=False, name=None, **kwargs):
        """ An inverted residual bottleneck block consisting of a separable depthwise convolution and 1x1 pointise
        convolution and an optional Squeeze-and-Excitation block.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        repeats : int
            number of repeats of the block
        kernel_size : int
            kernel size for depthwise convolution
        filters : int
            number of output filters
        residual : bool
            whether to make a residual connection
        strides : int
            stride for 3x3 convolution
        expansion_factor : int
            multiplier for the number of filters in internal convolutions
        width_factor : float
            multiplier for the number of filters
        activation : callable, optional
            If not specified tf.nn.relu is used.
        se_block : bool
            whether to include squeeze and excitation block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """

        num_filters = int(cls.num_channels(inputs, kwargs.get('data_format')) * expansion_factor * width_factor)
        x = conv_block(inputs, 'cnawna', [num_filters, num_filters], [1, kernel_size], strides=[1, strides],
                       name='%s-exp' % name, **kwargs)

        if se_block:
            if not isinstance(se_block, dict):
                se_block = dict(activation=[kwargs.get('activation', tf.nn.relu), h_sigmoid], ratio=num_filters // 4)
            x = cls.se_block(x, **se_block, name='%s-se' % name, data_format=kwargs.get('data_format'))

        x = conv_block(x, 'cn', filters, 1, name='%s-down' % name, **kwargs)

        if residual:
            x = inputs + x
        return x


class MobileNet_v3(MobileNet):
    """ MobileNet version 3 architecture """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['initial_block'].update(dict(layout='cna', filters=16, kernel_size=3, strides=2, activation=h_swish))
        config['body'].update(dict(width_factor=1, layout=deepcopy(_V3_LARGE_DEFAULT_BODY)))
        config['head'].update(dict(layout='cnavcacV', filters=[960, 1280, 2], pool_size=7, kernel_size=1,
                                   activation=h_swish))
        config['loss'] = 'ce'
        return config


class MobileNet_v3_small(MobileNet):
    """ MobileNet version 3 small architecture """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['initial_block'].update(dict(layout='cna', filters=16, kernel_size=3, strides=2, activation=h_swish))
        config['body'].update(dict(width_factor=1, layout=deepcopy(_V3_SMALL_DEFAULT_BODY)))
        config['head'].update(dict(layout='cnavcacV', filters=[576, 1280, 2], pool_size=7,
                                   kernel_size=1, activation=h_swish,
                                   se_block=dict(activation=[h_swish, h_sigmoid], ratio=144)))
        config['loss'] = 'ce'
        return config

    @classmethod
    def head(cls, inputs, se_block=None, name='head', **kwargs):
        """ The last network layers which produce predictions

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
        kwargs = cls.fill_params('head', **kwargs)
        layout = kwargs.pop('layout')
        filters = kwargs.pop('filters')
        if kwargs.get('se_block'):
            x = conv_block(inputs, layout=layout[:3], filters=filters[0], name='%s-conv1' % name, **kwargs)
            x = cls.se_block(x, **se_block, data_format=kwargs.get('data_format'), name='%s-se' % name)
            x = conv_block(x, layout=layout[3:], filters=filters[1:], name='%s-conv2' % name, **kwargs)
            return x
        return conv_block(inputs, layout=layout, filters=filters, name=name, **kwargs)
