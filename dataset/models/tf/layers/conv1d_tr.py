""" Contains 1d transposed convolution layer """
import tensorflow as tf


def conv1d_transpose(inputs, filters, kernel_size, strides=1, *args, **kwargs):
    """ Transposed 1D convolution layer

    Parameters
    ----------
    input_tensor : tf.Tensor
        input tensor
    filters : int
        number of filters in the ouput tensor
    kernel_size : int
        kernel size
    strides : int
        convolution stride. Default is 1.

    Returns
    -------
    tf.Tensor
    """
    up_tensor = tf.expand_dims(inputs, 1)
    conv_output = tf.layers.conv2d_transpose(up_tensor, filters=filters, kernel_size=(1, kernel_size),
                                             strides=(1, strides), *args, **kwargs)
    output = tf.squeeze(conv_output, [1])
    return output
