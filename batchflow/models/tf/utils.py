""" Utility functions. """
import tensorflow.compat.v1 as tf



def get_shape(tensor, dynamic=False):
    """ Return shape of the input tensor without batch size.

    Parameters
    ----------
    tensor : tf.Tensor

    dynamic : bool
        If True, returns tensor which represents shape. If False, returns list of ints and/or Nones.

    Returns
    -------
    shape : tf.Tensor or list
    """
    if dynamic:
        shape = tf.shape(tensor)
    else:
        shape = tensor.get_shape().as_list()
    return shape[1:]

def get_num_dims(tensor):
    """ Return a number of semantic dimensions (i.e. excluding batch and channels axis)"""
    shape = get_shape(tensor)
    dim = len(shape)
    return max(1, dim - 2)


def get_channels_axis(data_format='channels_last'):
    """ Return the integer channels axis based on string data format. """
    return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1

def get_num_channels(tensor, data_format='channels_last'):
    """ Return number of channels in the input tensor.

    Parameters
    ----------
    tensor : tf.Tensor

    Returns
    -------
    shape : tuple of ints
    """
    shape = tensor.get_shape().as_list()
    axis = get_channels_axis(data_format)
    return shape[axis]


def get_batch_size(tensor, dynamic=False):
    """ Return batch size (the length of the first dimension) of the input tensor.

    Parameters
    ----------
    tensor : tf.Tensor

    Returns
    -------
    batch size : int or None
    """
    if dynamic:
        return tf.shape(tensor)[0]
    return tensor.get_shape().as_list()[0]


def get_spatial_dim(tensor):
    """ Return spatial dim of the input tensor (without channels and batch dimension).

    Parameters
    ----------
    tensor : tf.Tensor

    Returns
    -------
    dim : int
    """
    return len(tensor.get_shape().as_list()) - 2

def get_spatial_shape(tensor, data_format='channels_last', dynamic=False):
    """ Return the tensor spatial shape (without batch and channels dimensions).

    Parameters
    ----------
    tensor : tf.Tensor

    dynamic : bool
        If True, returns tensor which represents shape. If False, returns list of ints and/or Nones.

    Returns
    -------
    shape : tf.Tensor or list
    """
    if dynamic:
        shape = tf.shape(tensor)
    else:
        shape = tensor.get_shape().as_list()
    axis = slice(1, -1) if data_format == "channels_last" else slice(2, None)
    return shape[axis]
