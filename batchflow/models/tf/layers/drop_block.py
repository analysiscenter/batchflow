"""
Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le "`DropBlock: A regularization method for convolutional networks
<https://arxiv.org/pdf/1810.12890.pdf>`_"
"""

import numpy as np
import tensorflow as tf
from .pooling import max_pooling

# TODO:
# Add scheduling scheme to gradually decrease keep_prob from 1 to target value.

def dropblock(inputs, dropout_rate, block_size, is_training, data_format, seed=None):
    """ Drop Block module.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor
    dropout_rate : float
        Default is 0
    block_size : int or float or tuple of ints or floats
        Size of the block to drop. If tuple, should be of the same size as spatial
        dimensions of inputs.
        If float < 0, block_size is calculated as a fraction of corresponding spatial
        dimension.
    is_training : bool or tf.Tensor
        Default is True.
    seed : int
        seed to use in tf.distributions.Bernoulli.sample method.
    data_format : str
        `channels_last` or `channels_first`. Default - 'channels_last'.

    Returns
    -------
    tf.Tensor
    """
    keep_prob = 1. - dropout_rate
    return tf.cond(tf.logical_or(tf.logical_not(is_training), tf.equal(keep_prob, 1.0)),
                   true_fn=lambda: inputs,
                   false_fn=lambda: _dropblock(inputs, keep_prob, block_size, seed, data_format),
                   name='dropblock')

def _dropblock(inputs, keep_prob, block_size, seed, data_format):
    """
    """
    shape = inputs.shape.as_list()

    if data_format == 'channels_first':
        spatial_dims, channels = shape[2:], shape[1]
    else:
        spatial_dims, channels = shape[1:-1], shape[-1]

    if isinstance(block_size, (float, int)):
        block_size = [block_size] * len(spatial_dims)
    if isinstance(block_size, tuple):
        if len(block_size) != len(spatial_dims):
            raise ValueError('Length of `block_size` should be the same as spatial dimensions of input.')
        block_size = list(block_size)
        for i, _ in enumerate(block_size):
            if block_size[i] < 1:
                block_size[i] = int(block_size[i] * spatial_dims[i])
            if block_size[i] > spatial_dims[i]:
                block_size[i] = spatial_dims[i]
        block_size = [max(bs, 1) for bs in block_size]
    else:
        raise ValueError('block_size should be tuple, float or int')

    inner_area = [spatial_dims[i] - block_size[i] + 1 for i in range(len(spatial_dims))]

    gamma = ((1. - keep_prob) * np.product(spatial_dims) / np.product(block_size) /
             np.product(inner_area))

    # Mask is sampled for each featuremap independently and applied identically to all batch items
    noise_dist = tf.distributions.Bernoulli(probs=gamma, dtype=tf.float32)
    sampling_mask_shape = ([1] + [channels] + inner_area if data_format == 'channels_first'
                           else [1] + inner_area + [channels])
    sampling_mask_shape = tf.stack(sampling_mask_shape)
    mask = noise_dist.sample(sampling_mask_shape, seed=seed)

    spatial_pads = [[(bs - 1) // 2, bs - 1 - (bs - 1) // 2] for bs in block_size]
    pad_shape = ([[0, 0]] + [[0, 0]] + spatial_pads if data_format == 'channels_first'
                 else [[0, 0]] + spatial_pads + [[0, 0]])
    mask = tf.pad(mask, pad_shape)

    # Using max pool operation to extend sampled points to blocks of desired size
    pool_size = block_size
    mask = max_pooling(mask, pool_size=pool_size, strides=[1] * (len(spatial_dims)),
                       data_format=data_format, padding='same')
    mask = tf.cast(1 - mask, tf.float32)
    output = tf.multiply(inputs, mask)

    # Scaling the output as in inverted dropout
    output = output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask)
    return output
