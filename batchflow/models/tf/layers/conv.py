""" Contains convolutional layers """
from functools import partial
import tensorflow.compat.v1 as tf
import tensorflow.keras.layers as K #pylint: disable=import-error

from .layer import Layer, add_as_function



@add_as_function
class Conv(Layer):
    """ Nd convolution layer. Just a wrapper around TensorFlow layers for corresponding dimensions.
    Used for `c` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    See also
    --------
    `tf.layers.conv1d <https://www.tensorflow.org/api_docs/python/keras/layers/Conv1D>`_,
    `tf.layers.conv2d <https://www.tensorflow.org/api_docs/python/keras/layers/Conv2D>`_,
    `tf.layers.conv3d <https://www.tensorflow.org/api_docs/python/keras/layers/Conv3D>`_
    """
    LAYERS = {
        1: K.Conv1D,
        2: K.Conv2D,
        3: K.Conv3D
    }

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 padding='same', data_format='channels_last', dilation_rate=(1, 1), **kwargs):
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate = dilation_rate
        self.kwargs = kwargs

        if self.filters is None or not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError("Filters must be a positive integer, instead got {}".format(self.filters))

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)



@add_as_function
class Conv1DTranspose:
    """ Transposed 1D convolution layer.

    Parameters
    ----------
    filters : int
        Number of filters in the ouput tensor.
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', data_format='channels_last', *args, **kwargs):
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        axis = 1 if self.data_format == 'channels_last' else 2
        x = tf.expand_dims(inputs, axis=axis)
        x = K.Conv2DTranspose(filters=self.filters, kernel_size=(1, self.kernel_size),
                              strides=(1, self.strides), padding=self.padding, **self.kwargs)(x)
        x = tf.squeeze(x, [axis])
        return x



@add_as_function
class Conv1DTransposeNn:
    """ Transposed 1D convolution layer. Analogue of the tf.nn.conv2d_transpose.

    Parameters
    ----------
    filters : tf.Tensor
        Convolutional filter.
    output_shape : tf.Tensor
        The output shape of the deconvolution op.
    strides : list
        The stride of the sliding window for each dimension of the input tensor.
    padding : str
        'VALID' or 'SAME'. Default - 'SAME'.
    data_format : str
        'NWC' or 'NCW'. Default - 'NWC'.
    name : str
        Scope name
    """
    def __init__(self, filters, output_shape, strides, padding='SAME', data_format='NWC', *args, **kwargs):
        self.filters, self.output_shape, self.strides = filters, output_shape, strides
        self.padding, self.data_format = padding, data_format
        self.args, self.kwargs = args, kwargs

    def __call__(self, inputs):
        axis = 1 if self.data_format == 'NWC' else 2
        inputs = tf.expand_dims(inputs, axis=axis)
        filters = tf.expand_dims(self.filters, axis=0)
        output_shape = tf.concat([self.output_shape[:axis], (1, ), self.output_shape[axis:]], axis=-1)
        strides = self.strides[:axis] + [1] + self.strides[axis:]
        x = tf.nn.conv2d_transpose(inputs, filters, output_shape, strides,
                                   self.padding, self.data_format, *self.args, **self.kwargs)
        x = tf.squeeze(x, [axis])
        return x



@add_as_function
class ConvTranspose(Layer):
    """ Transposed Nd convolution layer.
    Used for `t` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    filters : int
        Number of filters in the ouput tensor.
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.

    See also
    --------
    :func:`.conv1d_transpose`,
    `tf.layers.conv2d_transpose <https://www.tensorflow.org/api_docs/python/keras/layers/Conv2DTranspose>`_,
    `tf.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/keras/layers/Conv3DTranspose>`_
    """
    LAYERS = {
        1: Conv1DTranspose,
        2: K.Conv2DTranspose,
        3: K.Conv3DTranspose
    }

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 padding='same', data_format='channels_last', **kwargs):
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

        if self.filters is None or not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError("Filters must be a positive integer, instead got {}".format(self.filters))

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)



class DepthwiseConvND:
    """ TensorFlow implementation of depthwise convolution, applicable to any shape. """
    def __init__(self, transpose, kernel_size, strides=1, padding='same', data_format='channels_last',
                 dilation_rate=1, depth_multiplier=1, activation=None, name=None, **kwargs):
        self.transpose = transpose
        self.kernel_size, self.strides = kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.name = name
        self.kwargs = kwargs

    def __call__(self, inputs):
        context = None
        if self.name is not None:
            context = tf.variable_scope(self.name)
            context.__enter__()

        if self.transpose:
            conv_layer = ConvTranspose
        else:
            conv_layer = Conv
            self.kwargs['dilation_rate'] = self.dilation_rate

        kwargs = {**self.kwargs,
                  'kernel_size': self.kernel_size,
                  'filters': self.depth_multiplier,
                  'strides': self.strides,
                  'padding': self.padding,
                  'activation': self.activation,
                  'data_format': self.data_format}

        # Get all the shapes
        inputs_shape = inputs.get_shape().as_list()
        axis = -1 if self.data_format == 'channels_last' else 1
        size = [-1] * inputs.shape.ndims
        size[axis] = 1
        channels_in = inputs_shape[axis]

        # Loop through feature maps
        depthwise_layers = []
        for channel in range(channels_in):
            start = [0] * inputs.shape.ndims
            start[axis] = channel

            input_slice = tf.slice(inputs, start, size)

            _kwargs = {**kwargs,
                       'name': 'slice-%d' % channel}
            slice_conv = conv_layer(**_kwargs)(input_slice)
            depthwise_layers.append(slice_conv)

        # Concatenate the per-channel convolutions along the channel dimension.
        output = tf.concat(depthwise_layers, axis=axis)

        if context is not None:
            context.__exit__(None, None, None)

        return output



@add_as_function
class DepthwiseConv(Layer):
    """ Make Nd depthwise convolutions that act separately on channels.
    Used for `w` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Deafault - 1.
    activation : callable
        Default is None: linear activation.
    name : str
        The name of the layer. Default - None.
    """
    LAYERS = {
        1: partial(DepthwiseConvND, False),
        2: K.DepthwiseConv2D,
        3: partial(DepthwiseConvND, False)
    }

    def __init__(self, kernel_size, strides=(1, 1), padding='same',
                 data_format=None, dilation_rate=(1, 1), depth_multiplier=1, **kwargs):
        self.kernel_size, self.strides = kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.kwargs = kwargs

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)



@add_as_function
class DepthwiseConvTranspose(Layer):
    """ Make Nd depthwise transpose convolutions that act separately on channels.
    Used for `W` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Deafault - 1.
    activation : callable
        Default is None: linear activation.
    name : str
        The name of the layer. Default - None.
    """
    def __init__(self, kernel_size, strides=(1, 1), padding='same', data_format=None,
                 dilation_rate=(1, 1), depth_multiplier=1, activation=None, **kwargs):
        self.kernel_size, self.strides = kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.kwargs = kwargs

    def __call__(self, inputs):
        layer_fn = partial(DepthwiseConvND, True)
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)



class SeparableConvND:
    """ TensorFlow implementation of separable convolution, applicable to any shape. """
    def __init__(self, transpose, filters, kernel_size, strides=1, padding='same', data_format='channels_last',
                 dilation_rate=1, depth_multiplier=1, activation=None, name=None, **kwargs):
        self.transpose = transpose
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.name = name
        self.kwargs = kwargs

    def __call__(self, inputs):
        context = None
        if self.name is not None:
            context = tf.variable_scope(self.name)
            context.__enter__()

        # Make arguments for depthwise part and call it
        if self.transpose:
            depthwise_layer = DepthwiseConvTranspose
        else:
            depthwise_layer = DepthwiseConv

        _kwargs = {**self.kwargs,
                   'kernel_size': self.kernel_size,
                   'strides': self.strides,
                   'dilation_rate': self.dilation_rate,
                   'depth_multiplier': self.depth_multiplier,
                   'activation': self.activation,
                   'padding': self.padding,
                   'data_format': self.data_format,
                   'name': 'depthwise'}
        depthwise = depthwise_layer(**_kwargs)(inputs)

        # If needed, make arguments for pointwise part and call it
        shape_out = depthwise.get_shape().as_list()
        axis = -1 if self.data_format == 'channels_last' else 1
        filters_out = shape_out[axis]

        if filters_out != self.filters:
            _kwargs = {**self.kwargs,
                       'filters': self.filters,
                       'kernel_size': 1,
                       'strides': 1,
                       'dilation_rate': 1,
                       'data_format': self.data_format,
                       'name': 'pointwise'}
            output = Conv(**_kwargs)(depthwise)
        else:
            output = depthwise

        if context is not None:
            context.__exit__(None, None, None)

        return output



@add_as_function
class SeparableConv(Layer):
    """ Make Nd depthwise convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.
    Used for `C` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same',
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate : int
        Default is 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Default - 1.
    activation : callable
        Default is `tf.nn.relu`.
    name : str
        The name of the layer. Default - None.
    """
    LAYERS = {
        1: K.SeparableConv1D,
        2: K.SeparableConv2D,
        3: partial(SeparableConvND, False)
    }

    def __init__(self, filters, kernel_size, strides=(1, 1),
                 padding='same', data_format='channels_last',
                 dilation_rate=(1, 1), depth_multiplier=1, **kwargs):
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.kwargs = kwargs

        if self.filters is None or not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError("Filters must be a positive integer, instead got {}".format(self.filters))

    def __call__(self, inputs):
        layer_fn = self.LAYERS[inputs.shape.ndims - 2]
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)



@add_as_function
class SeparableConvTranspose(Layer):
    """ Make Nd depthwise transpose convolutions that acts separately on channels,
    followed by a pointwise convolution that mixes channels.
    Used for `T` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size.
    strides : int
        Convolution stride. Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to
        ``num_filters_in`` * ``depth_multiplier``. Deafault - 1.
    activation : callable
        Default is `tf.nn.relu`.
    name : str
        The name of the layer. Default - None.
    """
    def __init__(self, filters, kernel_size, strides=(1, 1),
                 padding='same', data_format='channels_last',
                 dilation_rate=(1, 1), depth_multiplier=1, **kwargs):
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.padding, self.data_format = padding, data_format
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.kwargs = kwargs

        if self.filters is None or not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError("Filters must be a positive integer, instead got {}".format(self.filters))

    def __call__(self, inputs):
        layer_fn = partial(SeparableConvND, True)
        return layer_fn(**self.params_dict, **self.kwargs)(inputs)
