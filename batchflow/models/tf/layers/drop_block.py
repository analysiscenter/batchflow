"""
Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le "`DropBlock: A regularization method for convolutional networks
<https://arxiv.org/pdf/1810.12890.pdf>`_"
"""

import numpy as np
import tensorflow as tf

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
    block_size : int or float
        size of the square block to drop.
        If float < 0, block_size is calculated as a fraction of smallest spatial
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
    # Transpose
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, perm=[0, *list(range(2, n_dims)), 1])

    shape = inputs.shape.as_list()
    n_dims = len(shape)

    spatial_dims, channels = shape[1:-1], shape[-1]

    if block_size < 1:
        block_size = int(block_size * min(spatial_dims))
    if block_size > min(spatial_dims):
        block_size = min(spatial_dims)
    block_size = max(block_size, 1)

    gamma = ((1. - keep_prob) * np.product(spatial_dims) / block_size ** len(spatial_dims) /
             np.product([dim - block_size + 1 for dim in spatial_dims]))

    # Mask is sampled for each featuremap independently
    noise_dist = tf.distributions.Bernoulli(probs=gamma, dtype=tf.float32)
    sampling_mask_shape = tf.stack([1] + [dim - block_size + 1 for dim in spatial_dims] + [channels])
    mask = noise_dist.sample(sampling_mask_shape, seed=seed)

    pad_one = (block_size - 1) // 2
    pad_two = (block_size - 1) - pad_one
    pad_shape = [[0, 0]] + [[pad_two, pad_one]] * len(spatial_dims) + [[0, 0]]
    mask = tf.pad(mask, pad_shape)

    pool_size = [1] + [block_size] * len(spatial_dims) + [1]
    mask = tf.nn.max_pool(mask, pool_size, [1] * (len(spatial_dims) + 2), 'SAME')
    mask = tf.cast(1 - mask, tf.float32)
    output = tf.multiply(inputs, mask)

    # Scaling the output as in inverted dropout
    output = output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask)

    if data_format == 'channels_first':
        output = tf.transpose(output, perm=[0, -1, *list(range(1, n_dims - 1))])
    return output
