"""
Golnaz Ghiasi, Tsung-Yi Lin, Quoc V. Le "`DropBlock: A regularization method for convolutional networks
<https://arxiv.org/abs/1810.12890>`_"
"""
import tensorflow.compat.v1 as tf

from .layer import Layer, add_as_function
from .pooling import MaxPooling

# TODO:
# When max_pooling allows for dynamic kernel size, implement block_size as fraction
# of spatial_dims.
# Write predefined callables to control dropout_rate


@add_as_function
class Dropblock(Layer):
    """ Drop Block module
    Used for `O` letter in layout convention of :class:`~.tf.layers.ConvBlock`.

    Parameters
    ----------
    dropout_rate : float, tf.Tensor or callable.
        Default is 0
    block_size : int or tuple of ints
        Size of the block to drop. If tuple, should be of the same size as spatial
        dimensions of inputs.
    is_training : bool or tf.Tensor
        Default is True.
    data_format : str
        `channels_last` or `channels_first`. Default - 'channels_last'.
    global_step: misc
        If `dropout_rate` is callable, and `global_step` is passed to it as the
        first positional argument.
    seed : int
        Seed to use in tf.distributions.Bernoulli.sample method.
    """
    def __init__(self, dropout_rate, block_size, data_format, global_step=None, seed=None, **kwargs):
        self.dropout_rate, self.block_size = dropout_rate, block_size
        self.data_format, self.global_step, self.seed = data_format, global_step, seed
        self.kwargs = kwargs

    def __call__(self, inputs, training):
        if callable(self.dropout_rate):
            self.dropout_rate = self.dropout_rate(self.global_step, **self.kwargs)

        if self.dropout_rate != 0.0:
            return tf.cond(training,
                           true_fn=lambda: self._dropblock(inputs, self.dropout_rate, self.block_size,
                                                           self.seed, self.data_format),
                           false_fn=lambda: inputs,
                           name='dropblock')
        return inputs


    def _dropblock(self, inputs, dropout_rate, block_size, seed, data_format):
        one = tf.convert_to_tensor([1], dtype=tf.int32)
        zeros_pad = tf.convert_to_tensor([[0, 0]], dtype=tf.int32)

        input_shape = tf.shape(inputs)

        if data_format == 'channels_first':
            spatial_dims, channels = input_shape[2:], input_shape[1:2]
        else:
            spatial_dims, channels = input_shape[1:-1], input_shape[-1:]
        spatial_ndim = spatial_dims.get_shape().as_list()[0]

        if isinstance(block_size, int):
            block_size = [block_size] * spatial_ndim
            block_size_tf = tf.convert_to_tensor(block_size, dtype=tf.int32)
        elif isinstance(block_size, tuple):
            if len(block_size) != spatial_ndim:
                raise ValueError('Length of `block_size` should be the same as spatial dimensions of input.')
            block_size_tf = tf.convert_to_tensor(block_size, dtype=tf.int32)
        else:
            raise ValueError('block_size should be int or tuple!')
        block_size_tf = tf.math.minimum(block_size_tf, spatial_dims)
        block_size_tf = tf.math.maximum(block_size_tf, one)

        spatial_dims_float = tf.cast(spatial_dims, dtype=tf.float32)
        block_size_tf_float = tf.cast(block_size_tf, dtype=tf.float32)

        inner_area = spatial_dims - block_size_tf + one
        inner_area_float = tf.cast(inner_area, dtype=tf.float32)

        gamma = (tf.convert_to_tensor(dropout_rate) * tf.math.reduce_prod(spatial_dims_float) /
                 tf.math.reduce_prod(block_size_tf_float) / tf.math.reduce_prod(inner_area_float))

        # Mask is sampled for each featuremap independently and applied identically to all batch items
        noise_dist = tf.distributions.Bernoulli(probs=gamma, dtype=tf.float32)

        if data_format == 'channels_first':
            sampling_mask_shape = tf.concat((one, channels, inner_area), axis=0)
        else:
            sampling_mask_shape = tf.concat((one, inner_area, channels), axis=0)
        mask = noise_dist.sample(sampling_mask_shape, seed=seed)

        left_spatial_pad = (block_size_tf - one) // 2
        right_spatial_pad = block_size_tf - one - left_spatial_pad
        spatial_pads = tf.stack((left_spatial_pad, right_spatial_pad), axis=1)
        if data_format == 'channels_first':
            pad_shape = tf.concat((zeros_pad, zeros_pad, spatial_pads), axis=0)
        else:
            pad_shape = tf.concat((zeros_pad, spatial_pads, zeros_pad), axis=0)
        mask = tf.pad(mask, pad_shape)

        # Using max pool operation to extend sampled points to blocks of desired size
        pool_size = block_size
        strides = [1] * spatial_ndim
        mask = MaxPooling(pool_size=pool_size, pool_strides=strides, data_format=data_format, padding='same')(mask)
        mask = tf.cast(1 - mask, tf.float32)
        output = tf.multiply(inputs, mask)

        # Scaling the output as in inverted dropout
        output = output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask)
        return output
