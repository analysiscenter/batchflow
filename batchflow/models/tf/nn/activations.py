""" Contains custom tf activation functions """
import tensorflow.compat.v1 as tf

def h_swish(inputs):
    """ Hard-swish nonlinearity function.
    See https://arxiv.org/abs/1905.02244.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor

    Returns
    -------
    tf.Tensor
    """
    return inputs * tf.nn.relu6(inputs + 3) / 6.0

def h_sigmoid(inputs):
    """ Hard-sigmoid nonlinearity function.
    A piece-wise linear analog of sigmoid function.

    Parameters
    ----------
    inputs : tf.Tensor
        input tensor

    Returns
    -------
    tf.Tensor
    """
    return tf.nn.relu6(inputs + 3) / 6.0
