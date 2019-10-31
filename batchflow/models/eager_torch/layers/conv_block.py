""" Convenient combining block """
import logging
import inspect
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .core import *
from ...utils import unpack_args


logger = logging.getLogger(__name__)



class ConvBlock(nn.Module):
    """ Complex multi-dimensional block to apply sequence of different operations.

    Parameters
    ----------
    layout : str
        A sequence of letters, each letter meaning individual operation:

        - c - convolution
        - t - transposed convolution
        - C - separable convolution
        - T - separable transposed convolution
        - w - depthwise convolution
        - W - depthwise transposed convolution
        - f - dense (fully connected)
        - n - batch normalization
        - a - activation
        - p - pooling (default is max-pooling)
        - v - average pooling
        - P - global pooling (default is max-pooling)
        - V - global average pooling
        - d - dropout
        - D - alpha dropout
        - O - dropblock
        - m - maximum intensity projection (:class:`~.layers.Mip`)
        - b - upsample with bilinear resize
        - B - upsample with bilinear additive resize
        - N - upsample with nearest neighbors resize
        - X - upsample with subpixel convolution (:class:`~.layers.SubpixelConv`)
        - R - start residual connection
        - A - start residual connection with bilinear additive upsampling
        - `+` - end residual connection with summation
        - `*` - end residual connection with multiplication
        - `.` - end residual connection with concatenation

        Default is ''.

    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size.
    name : str
        Name of the layer that will be used as a scope.
    units : int
        Number of units in the dense layer.
    strides : int
        Default is 1.
    padding : str
        Padding mode, can be 'same' or 'valid'. Default - 'same'.
    data_format : str
        'channels_last' or 'channels_first'. Default - 'channels_last'.
    dilation_rate: int
        Default is 1.
    activation : callable
        Default is `tf.nn.relu`.
    pool_size : int
        Default is 2.
    pool_strides : int
        Default is 2.
    pool_op : str
        Pooling operation ('max', 'mean', 'frac')
    dropout_rate : float
        Default is 0.
    factor : int or tuple of int
        Upsampling factor
    upsampling_layout : str
        Layout for upsampling layers
    is_training : bool or tf.Tensor
        Default is True.
    reuse : bool
        Whether to user layer variables if exist

    dense : dict
        Parameters for dense layers, like initializers, regularalizers, etc.
    conv : dict
        Parameters for convolution layers, like initializers, regularalizers, etc.
    transposed_conv : dict
        Parameters for transposed conv layers, like initializers, regularalizers, etc.
    batch_norm : dict or None
        Parameters for batch normalization layers, like momentum, intiializers, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    pooling : dict
        Parameters for pooling layers, like initializers, regularalizers, etc.
    dropout : dict or None
        Parameters for dropout layers, like noise_shape, etc
        If None or inculdes parameters 'off' or 'disable' set to True or 1,
        the layer will be excluded whatsoever.
    dropblock : dict or None
        Parameters for dropblock layers, like dropout_rate, block_size, etc.
    subpixel_conv : dict or None
        Parameters for subpixel convolution like layout, activation, etc.
    resize_bilinear : dict or None
        Parameters for bilinear resize.
    resize_bilinear_additive : dict or None
        Parameters for bilinear additive resize like layout, activation, etc.

    Notes
    -----
    When ``layout`` includes several layers of the same type, each one can have its own parameters,
    if corresponding args are passed as lists (not tuples).

    Spaces may be used to improve readability.


    Examples
    --------
    A simple block: 3x3 conv, batch norm, relu, 2x2 max-pooling with stride 2::

        x = ConvBlock('cnap', filters=32, kernel_size=3)(x)

    A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

        x = ConvBlock('nac nac nac', [64, 64, 256], [1, 3, 1])(x)

    A complex Nd block:

    - 5x5 conv with 32 filters
    - relu
    - 3x3 conv with 32 filters
    - relu
    - 3x3 conv with 64 filters and a spatial stride 2
    - relu
    - batch norm
    - dropout with rate 0.15

    ::

        x = ConvBlock('ca ca ca nd', [32, 32, 64], [5, 3, 3], strides=[1, 1, 2], dropout_rate=.15)(x)

    A residual block::

        x = ConvBlock('R nac nac +', [16, 16, 64], [1, 3, 1])(x)

    """
    LETTERS_LAYERS = {
        'a': 'activation',
        'R': 'residual_start',
        '+': 'residual_end',
        '.': 'residual_end',
        '*': 'residual_end',
        'f': 'dense',
        'c': 'conv',
        't': 'transposed_conv',
        'C': 'separable_conv',
        'T': 'separable_conv_transpose',
        'w': 'depthwise_conv',
        'W': 'depthwise_conv_transpose',
        'p': 'pooling',
        'v': 'pooling',
        'P': 'global_pooling',
        'V': 'global_pooling',
        'n': 'batch_norm',
        'd': 'dropout',
        'D': 'alpha_dropout',
        'O': 'dropblock',
        'm': 'mip',
        'A': 'residual_bilinear_additive',
        'b': 'resize_bilinear',
        'N': 'resize_nn',
        'X': 'subpixel_conv'
    }

    LAYERS_CLASSES = {
        'activation': Activation,
        'residual_start': None,
        'residual_end': None,
        'dense': Dense,
        'conv': Conv,
        'transposed_conv': ConvTranspose,
        'separable_conv': SeparableConv,
        'separable_conv_transpose': SeparableConvTranspose,
        'depthwise_conv': DepthwiseConv,
        'depthwise_conv_transpose': DepthwiseConvTranspose,
        'pooling': Pool,
        'global_pooling': GlobalPool,
        'batch_norm': BatchNorm,
        'dropout': Dropout,
        'alpha_dropout': AlphaDropout,
        'dropblock': None, # TODO
        'mip': None, # TODO?
        'residual_bilinear_additive': Interpolate, # TODO
        'resize_bilinear': Interpolate,
        'resize_nn': Interpolate,
        'subpixel_conv': SubPixelConv,
    }

    DEFAULT_LETTERS = LETTERS_LAYERS.keys()
    LETTERS_GROUPS = dict(zip(DEFAULT_LETTERS, DEFAULT_LETTERS))
    LETTERS_GROUPS.update({
        'C': 'c',
        't': 'c',
        'T': 'c',
        'w': 'c',
        'W': 'c',
        'v': 'p',
        'V': 'P',
        'D': 'd',
        'O': 'd',
        'n': 'd',
        'A': 'b',
        'B': 'b',
        'N': 'b',
        'X': 'b',
        })


    def add_letter(self, letter, cls, name=None):
        """ Add custom letter to layout parsing procedure.

        Parameters
        ----------
        letter : str
            Letter to add.
        cls : class
            Tensor-processing layer. Must have layer-like signature (both init and call overloaded).
        name : str
            Name of parameter dictionary. Defaults to `letter`.

        Examples
        --------
        Add custom `Q` letter::

            block = ConvBlock('cnap Q', filters=32, custom_params={'key': 'value'})
            block.add_letter('Q', my_layer_class, 'custom_params')
            x = block(x)
        """
        name = name or letter
        self.LETTERS_LAYERS.update({letter: name})
        self.LAYERS_CLASSES.update({name: cls})
        self.LETTERS_GROUPS.update({letter: letter})


    def __init__(self, inputs=None, layout='',
                 filters=0, kernel_size=3, strides=1, dilation_rate=1, depth_multiplier=1,
                 activation='relu',
                 pool_size=2, pool_strides=2,
                 dropout_rate=0.,
                 padding='same', data_format='channels_first', name=None,
                 **kwargs):
        super().__init__()

        self.inputs = inputs
        self.layout = layout
        self.filters, self.kernel_size, self.strides = filters, kernel_size, strides
        self.dilation_rate, self.depth_multiplier = dilation_rate, depth_multiplier
        self.activation = activation
        self.pool_size, self.pool_strides = pool_size, pool_strides
        self.dropout_rate = dropout_rate
        self.padding, self.data_format = padding, data_format
        self.kwargs = kwargs

        block_modules, skip_modules, combine_modules = self.parse_params(self.inputs)
        self.block_modules = block_modules
        self.skip_modules = skip_modules if skip_modules else None
        self.combine_modules = combine_modules if combine_modules else None

    def forward(self, x):
        b_counter, s_counter, c_counter = 0, 0, 0
        residuals = []

        for letter in self.module_layout:
            if letter == '_':
                x = self.block_modules[b_counter](x)
                b_counter += 1
            elif letter in ['R', 'A']:
                residuals += [self.skip_modules[s_counter](x)]
                s_counter += 1
            elif letter in ['+', '*', '.']:
                x = self.combine_modules[c_counter]([x, residuals[-1]])
                residuals = residuals[:-1]
                c_counter += 1
        return x


    def fill_layer_params(self, layer_class):
        layer_params = inspect.getfullargspec(layer_class.__init__)[0]
        layer_params.remove('self')

        args = {param: getattr(self, param) if hasattr(self, param) else self.kwargs.get(param, None)
                for param in layer_params
                if (hasattr(self, param) or (param in self.kwargs))}
        return args

    def parse_params(self, inputs=None):
        layout = self.layout or ''
        layout = layout.replace(' ', '')
        if len(layout) == 0:
            logger.warning('ConvBlock: layout is empty, so there is nothing to do, just returning inputs.')
            return nn.Sequential([])
        self.module_layout = ''


        layout_dict = {}
        for letter in layout:
            letter_group = self.LETTERS_GROUPS[letter]
            letter_counts = layout_dict.setdefault(letter_group, [-1, 0])
            letter_counts[1] += 1

        modules, skip_modules, combine_modules = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layers, residuals = [], []

        for i, letter in enumerate(layout):
            # Arguments for layer creating; arguments for layer call
            args = {}

            letter_group = self.LETTERS_GROUPS[letter]
            layer_name = self.LETTERS_LAYERS[letter]
            layer_class = self.LAYERS_CLASSES[layer_name]
            layout_dict[letter_group][0] += 1

            if letter in ['R', 'A', '+', '*', '.']:
                if len(layers) >= 1:
                    self.module_layout += '_'
                    modules.append(nn.Sequential(OrderedDict(layers)))
                    layers = []
                self.module_layout += letter

                if letter == 'R':
                    residuals += [self.inputs]

                    layer_desc = 'Layer {}, letter "{}"'.format(i, letter)
                    layer = nn.Sequential(OrderedDict([(layer_desc, nn.Sequential())]))
                    skip_modules.append(layer)
                elif letter == 'A':
                    args = self.fill_layer_params(layer_class)
                    args['mode'] = args.get('mode', 'b')
                    layer = layer_class(**args)
                    skip = layer(self.inputs)
                    residuals += [skip]

                    layer_desc = 'Layer {}, letter "{}"; {} -> {}'.format(i, letter, get_shape(self.inputs), get_shape(skip))
                    layer = nn.Sequential(OrderedDict([(layer_desc, layer)]))
                    skip_modules.append(layer)
                elif letter in ['+', '*', '.']:
                    layer = Combine(op=letter)
                    shape_before = get_shape(self.inputs)
                    self.inputs = layer([self.inputs, residuals[-1]])
                    shape_after = get_shape(self.inputs)
                    residuals = residuals[:-1]

                    shape_before, shape_after = (None, *shape_before[1:]), (None, *shape_after[1:])
                    layer_desc = 'Layer {}: {} -> {}'.format(i, shape_before, shape_after)
                    layer = nn.Sequential(OrderedDict([(layer_desc, layer)]))
                    combine_modules.append(layer)
            else:
                layer_args = self.kwargs.get(layer_name, {})
                skip_layer = layer_args is False \
                             or isinstance(layer_args, dict) and layer_args.get('disable', False)

                # Create params for the layer call
                if skip_layer:
                    pass
                elif letter in self.DEFAULT_LETTERS:
                    args = self.fill_layer_params(layer_class)
                else:
                    if letter not in self.LETTERS_LAYERS.keys():
                        raise ValueError('Unknown letter symbol - %s' % letter)

                # Additional params for some layers
                if letter_group.lower() == 'p':
                    # Choosing pooling operation
                    pool_op = 'mean' if letter.lower() == 'v' else self.kwargs.pop('pool_op', 'max')
                    args['op'] = pool_op

                elif letter_group == 'b':
                    # Additional layouts for all the upsampling layers
                    args['mode'] = args.get('mode', letter.lower())
                    if self.kwargs.get('upsampling_layout'):
                        args['layout'] = self.kwargs.get('upsampling_layout')

                if not skip_layer:
                    args = {**args, **layer_args}
                    args = unpack_args(args, *layout_dict[letter_group])
                    layer = layer_class(**args)

                    shape_before = get_shape(self.inputs)
                    self.inputs = layer(self.inputs)
                    shape_after = get_shape(self.inputs)

                    shape_before, shape_after = (None, *shape_before[1:]), (None, *shape_after[1:])
                    layer_desc = 'Layer {}, letter "{}"; {} -> {}'.format(i, letter, shape_before, shape_after)
                    layers.append((layer_desc, layer))

        if len(layers) > 0:
            self.module_layout += '_'
            modules.append(nn.Sequential(OrderedDict(layers)))

        return modules, skip_modules, combine_modules


def update_layers(letter, func, name=None):
    """ Add custom letter to layout parsing procedure.

    Parameters
    ----------
    letter : str
        Letter to add.
    func : class
        Tensor-processing layer. Must have layer-like signature (both init and call overloaded).
    name : str
        Name of parameter dictionary. Defaults to `letter`.

    Examples
    --------
    Add custom `Q` letter::

        block = ConvBlock('cnap Q', filters=32, custom_params={'key': 'value'})
        block.add_letter('Q', my_func, 'custom_params')
        x = block(x)
    """
    name = name or letter
    ConvBlock.LETTERS_LAYERS.update({letter: name})
    ConvBlock.LAYERS_CLASSES.update({name: func})
    ConvBlock.LETTERS_GROUPS.update({letter: letter})



class Upsample(nn.Module):
    """ Upsample inputs with a given factor

    Parameters
    ----------
    factor : int
        an upsamping scale
    shape : tuple of int
        a shape to upsample to (used by bilinear and NN resize)
    layout : str
        resizing technique, a sequence of:

        - b - bilinear resize
        - N - nearest neighbor resize
        - t - transposed convolution
        - T - separable transposed convolution
        - X - subpixel convolution

        all other :class:`~.torch.ConvBlock` layers are also allowed.

    inputs
        an input tensor

    Examples
    --------
    A simple bilinear upsampling::

        x = Upsample(layout='b', shape=(256, 256), inputs=inputs)

    Upsampling with non-linear normalized transposed convolution::

        x = Upsample(layout='nat', factor=2, kernel_size=3, inputs=inputs)

    Subpixel convolution::

        x = Upsample(layout='X', factor=2, inputs=inputs)
    """
    def __init__(self, factor=2, shape=None, layout='b', *args, inputs=None, **kwargs):
        super().__init__()

        _ = args

        if 't' in layout or 'T' in layout:
            if 'kernel_size' not in kwargs:
                kwargs['kernel_size'] = factor
            if 'strides' not in kwargs:
                kwargs['strides'] = factor

        self.layer = ConvBlock(inputs=inputs, layout=layout, factor=factor, shape=shape, **kwargs)

    def forward(self, x):
        return self.layer(x)



class Combine(nn.Module):
    """ Combine list of tensor into one.

    Parameters
    ----------
    inputs : sequence of torch.Tensors
        Tensors to combine.

    op : str {'concat', 'sum', 'conv'}
        If 'concat', inputs are concated along channels axis.
        If 'sum', inputs are summed.
        If 'softsum', every tensor is passed through 1x1 convolution in order to have
        the same number of channels as the first tensor, and then summed.
    """
    def __init__(self, inputs=None, op='concat'):
        super().__init__()

        self.op = op

        if op == ['softsum', '&']:
            args = dict(layout='c', filters=get_shape(inputs[0])[1],
                        kernel_size=1)
            conv = [ConvBlock(get_shape(tensor), **args) for tensor in inputs]
            self.conv = nn.ModuleList(conv)


    def forward(self, inputs):
        if self.op in ['concat', '.']:
            return torch.cat(inputs, dim=1)
        if self.op in ['sum', '+']:
            return torch.stack(inputs, dim=0).sum(dim=0)
        if self.op in ['multi', '*']:
            result = 1
            for item in inputs:
                result *= item
            return result
        if self.op in ['softsum', '&']:
            inputs = [conv(tensor)
                      for conv, tensor in zip(self.conv, inputs)]
            return torch.stack(inputs, dim=0).sum(dim=0)
        raise ValueError('Combine `op` must be one of `concat`, `sum`, `softsum`, got {}.'.format(self.op))

    def extra_repr(self):
        return 'op=' + self.op
