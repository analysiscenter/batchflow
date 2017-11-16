"""Contains class for FCN"""
import tensorflow as tf

from . import TFModel, VGG16
from .layers import conv_block


class FCN(TFModel):
    """ Base Fully convolutional network (FCN)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016)
    """

    def _build_config(self, names=None):
        names = names if names else ['images', 'masks']
        config = super()._build_config(names)

        config['default']['data_format'] = self.data_format('images')
        config['default']['dropout_rate'] = self.get_from_config('default/dropout_rate', .5)

        config['input_block']['inputs'] = self.inputs['images']
        config['input_block']['input_class'] = self.get_from_config('base_network', VGG16)
        config['head']['num_classes'] = self.num_classes('masks')
        config['head']['images'] = self.inputs['images']#.get_shape().as_list()[1:-1]
        config['body']['num_classes'] = self.num_classes('masks')
        config['body']['filters'] = self.get_from_config('body/filters', 100)

        return config


    @classmethod
    def input_block(cls, inputs, input_class, name='input_block', **kwargs):
        """ VGG

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of output filters
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        return input_class.body(inputs, name=name, **kwargs)

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor

        Returns
        -------
        tf.Tensor
        """
        raise NotImplementedError()

    @classmethod
    def head(cls, inputs, filters, factor, images, num_classes, name='head', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters
        image_size : tuple
            the output image size
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        x = conv_block(inputs, num_classes, filters, 't', name=name, strides=factor, **kwargs)
        x = cls.crop(x, images, kwargs.get('data_format'))
        return x


class FCN32(FCN):
    """  Fully convolutional network (FCN32)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    base_network : class
        base network (VGG16 by default)
    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
    head : dict
        factor : int
            upsampling factor (default=32)
        filters : int
            number of filters in the final upsampling block (default=32)
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['head']['filters'] = self.get_from_config('head/filters', 32)
        config['head']['factor'] = self.get_from_config('head/factor', 32)
        return config

    @classmethod
    def body(cls, inputs, filters, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        layout = kwargs.pop('layout', 'cnad cnad')
        return conv_block(inputs, filters, [7, 1], layout=layout, name=name, **kwargs)


class FCN16(FCN):
    """  Fully convolutional network (FCN16)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    base_network : class
        base network (VGG16 by default)
    input_block : dict
        skip_name : str
            tensor name for the skip connection.
            Default='block-3/output:0' for VGG16.
    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
    head : dict
        factor : int
            upsampling factor (default=32)
        filters : int
            number of filters in the final upsampling block (default=32)
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['head']['factor'] = self.get_from_config('head/factor', 16)
        config['head']['filters'] = self.get_from_config('head/filters', 16)
        config['input_block']['skip_name'] = self.graph.get_name_scope() + '/input_block/' + \
                                             self.get_from_config('/input_block/skip_name', 'block-3/output:0')
        return config

    @classmethod
    def input_block(cls, inputs, input_class, skip_name, name='input_block', **kwargs):
        x = FCN.input_block(inputs, input_class, name, **kwargs)
        skip = tf.get_default_graph().get_tensor_by_name(skip_name)
        return x, skip

    @classmethod
    def body(cls, inputs, filters, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            two input tensors
        filters : int
            number of filters
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x, skip = inputs
            x = FCN32.body(x, filters, name='fcn32', **kwargs)
            x = conv_block(x, num_classes, 1, 't', 'fcn32_2', strides=2, **kwargs)

            skip = conv_block(skip, num_classes, 1, 'c', 'pool', **kwargs)
            x = cls.crop(x, skip, kwargs.get('data_format'))
            output = tf.add(x, skip, name='output')
        return output


class FCN8(FCN):
    """  Fully convolutional network (FCN8)
    https://arxiv.org/abs/1605.06211 (E.Shelhamer et al, 2016)

    **Configuration**

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    base_network : class
        base network (VGG16 by default)
    input_block : dict
        skip1_name : str
            tensor name for the first skip connection.
            Default='block-3/output:0' for VGG16.
        skip2_name : str
            tensor name for the second skip connection.
            Default='block-2/output:0' for VGG16.
    body : dict
        filters : int
            number of filters in convolutions after base network (default=100)
    head : dict
        factor : int
            upsampling factor (default=32)
        filters : int
            number of filters in the final upsampling block (default=32)
    """
    def _build_config(self, names=None):
        config = super()._build_config(names)
        config['head']['factor'] = self.get_from_config('head/factor', 8)
        config['head']['filters'] = self.get_from_config('head/filters', 8)
        config['input_block']['skip1_name'] = self.graph.get_name_scope() + '/input_block/' + \
                                             self.get_from_config('/input_block/skip1_name', 'block-3/output:0')
        config['input_block']['skip2_name'] = self.graph.get_name_scope() + '/input_block/' + \
                                             self.get_from_config('/input_block/skip2_name', 'block-2/output:0')
        return config

    @classmethod
    def input_block(cls, inputs, input_class, skip1_name, skip2_name, name='input_block', **kwargs):
        x = FCN.input_block(inputs, input_class, name, **kwargs)
        skip1 = tf.get_default_graph().get_tensor_by_name(skip1_name)
        skip2 = tf.get_default_graph().get_tensor_by_name(skip2_name)
        return x, skip1, skip2

    @classmethod
    def body(cls, inputs, filters, num_classes, name='body', **kwargs):
        """ Base layers

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters
        num_classes : int
            number of classes

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x, skip1, skip2 = inputs

            x = FCN16.body((x, skip1), filters, num_classes, name='fcn16', **kwargs)
            x = conv_block(x, num_classes, 1, 't', name='fcn16_2', strides=2, **kwargs)

            skip2 = conv_block(skip2, num_classes, 1, 'c', name='pool2')

            x = cls.crop(x, skip2, kwargs.get('data_format'))
            output = tf.add(x, skip2, name='output')
        return output
