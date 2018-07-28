""" Contains convolution block """
import logging
import torch.nn as nn

from .core import Dense, Activation, \
                  Conv, ConvTranspose, SeparableConv, SeparableConvTranspose, \
                  Dropout, BatchNorm, Pool, GlobalPool
from ..utils import get_output_shape, get_shape
from ...utils import unpack_args


logger = logging.getLogger(__name__)


FUNC_LAYERS = {
    'activation': Activation,
    'residual_start': None,
    'residual_end': None,
    'dense': Dense,
    'conv': Conv,
    'transposed_conv': ConvTranspose,
    'separable_conv': SeparableConv,
    'separable_conv_transpose': SeparableConvTranspose,
    'pooling': Pool,
    'global_pooling': GlobalPool,
    'batch_norm': BatchNorm,
    'dropout': Dropout,
    'alpha_dropout': nn.AlphaDropout,
}


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
}


LAYER_KEYS = ''.join(list(C_LAYERS.keys()))
GROUP_KEYS = (
    LAYER_KEYS
    .replace('t', 'c')
    .replace('C', 'c')
    .replace('T', 'c')
    .replace('v', 'p')
    .replace('V', 'P')
    .replace('D', 'd')
)

C_GROUPS = dict(zip(LAYER_KEYS, GROUP_KEYS))


class ConvBlock(nn.Module):
    """ Complex multi-dimensional block with a sequence of convolutions, batch normalization, activation, pooling,
    dropout and even dense layers.
    """
    def __init__(self, inputs=None, layout='', filters=0, kernel_size=3, strides=1, padding='same', dilation_rate=1,
                 depth_multiplier=1, activation='relu', pool_size=2, pool_strides=2, dropout_rate=0, units=None,
                 shape=None, **kwargs):
        super().__init__()

        layout = layout or ''
        self.layout = layout.replace(' ', '')

        if len(self.layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do')
            return

        shape = shape or get_shape(inputs)

        layout_dict = {}
        for layer in self.layout:
            if C_GROUPS[layer] not in layout_dict:
                layout_dict[C_GROUPS[layer]] = [-1, 0]
            layout_dict[C_GROUPS[layer]][1] += 1

        modules = []
        for _, layer in enumerate(self.layout):

            layout_dict[C_GROUPS[layer]][0] += 1
            layer_name = C_LAYERS[layer]
            layer_fn = FUNC_LAYERS[layer_name]

            layer_args = kwargs.get(layer_name, {})
            skip_layer = layer_args is False or isinstance(layer_args, dict) and layer_args.get('disable', False)

            if skip_layer:
                pass

            elif layer == 'a':
                args = dict(activation=activation, shape=shape)

            elif layer == 'f':
                if units is None:
                    raise ValueError('units cannot be None if layout includes dense layers')
                args = dict(units=units, shape=shape)

            elif layer in ['c', 't']:
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            dilation_rate=dilation_rate, shape=shape)

            elif layer in ['C', 'T']:
                args = dict(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            dilation_rate=dilation_rate, depth_multiplier=depth_multiplier, shape=shape)

            elif layer == 'n':
                args = dict(shape=shape)

            elif C_GROUPS[layer] == 'p':
                pool_op = 'mean' if layer == 'v' else kwargs.pop('pool_op', 'max')
                args = dict(op=pool_op, kernel_size=pool_size, stride=pool_strides, padding=padding, dilation=1,
                            shape=shape)

            elif C_GROUPS[layer] == 'P':
                pool_op = 'mean' if layer == 'V' else kwargs.pop('pool_op', 'max')
                args = dict(op=pool_op, shape=shape)

            elif layer in ['d', 'D']:
                if dropout_rate:
                    args = dict(p=dropout_rate, shape=shape)
                else:
                    logger.warning('ConvBlock: dropout_rate is zero or undefined, so dropout layer is skipped')
                    skip_layer = True

            else:
                raise ValueError('Unknown layer symbol', layer)

            if not skip_layer:
                layer_args = layer_args.copy()
                if 'disable' in layer_args:
                    del layer_args['disable']
                args = {**args, **layer_args}
                args = unpack_args(args, *layout_dict[C_GROUPS[layer]])

                new_layer = layer_fn(**args)
                modules.append(new_layer)
                shape = get_output_shape(new_layer, shape)

        self.block = nn.Sequential(*modules)
        self.output_shape = shape


    def forward(self, x):
        """ Make forward pass """
        if self.layout:
            x = self.block(x)
        return x
