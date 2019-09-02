""" EfficientNet """

import tensorflow as tf

from . import TFModel
from .layers import conv_block


class EfficientNet(TFModel):
    """

    phi - scaling parameter
    alpha - depth scaling base, depth factor `d=alpha^phi`
    beta - width (number of channels) scaling base, width factor `w=beta^phi`
    resolution is set explicitly via inputs resolution.
    helper function `get_resolution_factor` is provided to calculate resolution factor `r`
    by given `alpha`, `beta`, `phi`, so that :math: `r^2 * w^2 * d \approx 2`

    """

    @classmethod
    def get_resolution_factor(cls, alpha, beta, phi):
        return (2 / alpha / beta / beta)**(phi/2)

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        kwargs = cls.fill_params('body', **kwargs)

        alpha = kwargs.pop('alpha')
        beta = kwargs.pop('beta')
        phi = kwargs.pop('phi')

        with tf.variable_scope(name):
            base_class = kwargs.get('base', None)
            if base_class is not None:
                return base_class.make_efficientnet_scaling(inputs, alpha, beta, phi, **kwargs)

            blocks = kwargs.pop('blocks')
            x = inputs
            for i, block_args in enumerate(blocks):
                with tf.variable_scope('block-%d' % i):
                    base_block = block_args.get('base', None)
                    scalable = block_args.get('scalable')

                    d_factor = alpha ** phi if scalable else 1
                    w_factor = beta ** phi if scalable else 1

                    repeats = block_args.get('repeats')
                    repeats = int(repeats * d_factor)

                    layer_args = {**kwargs, **block_args}
                    filters = layer_args.pop('filters')

                    if base_block is None:
                        base_block = conv_block

                        if not isinstance(filters, (tuple, list)):
                            filters = [filters]
                        filters = [int(f * w_factor) for f in filters]

                    for j in range(repeats):
                        name = 'block-%d-layer-%d' % (i, j)
                        x = base_block(x, w_factor=w_factor, filters=filters, name=name, **layer_args)
            return x
