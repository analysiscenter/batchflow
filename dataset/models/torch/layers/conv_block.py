""" Contains convolution block """
import torch.nn as nn


C_LAYERS = {
    'a': 'activation',
    'R': 'residual_start',
    '+': 'residual_end',
    '.': 'residual_end',
    'f': 'dense',
    'c': 'conv',
    't': 'transposed_conv',
    'C': 'separable_conv',
    'T': 'separable_conv_transpose',
    'p': 'pooling',
    'v': 'pooling',
    'P': 'global_pooling',
    'V': 'global_pooling',
    'n': 'batch_norm',
    'd': 'dropout',
    'D': 'alpha_dropout',
    'm': 'mip',
}


def _conv_block(inputs, layout='', filters=0, kernel_size=3, name=None,
                strides=1, padding='same', data_format='channels_last', dilation_rate=1, depth_multiplier=1,
                activation=tf.nn.relu, pool_size=2, pool_strides=2, dropout_rate=0., is_training=True,
                **kwargs):




def _unpack_args(args, layer_no, layers_max):
    new_args = {}
    for arg in args:
        if isinstance(args[arg], list) and layers_max > 1:
            arg_value = args[arg][layer_no]
        else:
            arg_value = args[arg]
        new_args.update({arg: arg_value})
    return new_args