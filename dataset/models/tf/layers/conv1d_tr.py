""" Contains 1d transposed convolution layer """
import tensorflow as tf


def conv1d_transpose(tensor, filters, kernel_size, strides, padding, use_bias):
    """ Transposed 1D convolution layer

    Parameters
    ----------
    input_tensor : tf.Tensor
    filters : int - number of filters in the ouput tensor
    kernel_size : int - kernel size
    strides : int
    padding : str - padding mode, can be 'same' or 'valid'
    use_bias : bool

    """
    up_tensor = tf.expand_dims(tensor, 1)
    conv_output = tf.layers.conv2d_transpose(up_tensor, filters=filters, kernel_size=(1, kernel_size),
                                             strides=(1, strides), padding=padding, use_bias=use_bias)
    output = tf.squeeze(conv_output, [1])
    return output
