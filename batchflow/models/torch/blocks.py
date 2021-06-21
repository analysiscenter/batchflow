""" Contains blocks for various deep network architectures."""
import numpy as np

from .layers import ConvBlock
from .utils import get_num_channels, safe_eval



CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']



class DefaultBlock(ConvBlock):
    """ Block with default parameters.
    Allows creation of modules with predefined arguments for :class:`~.ConvBlock`.
    """
    LAYOUT = 'cna'
    FILTERS = 'same'

    def __init__(self, inputs=None, **kwargs):
        attrs = {name.lower(): value for name, value in vars(type(self)).items()
                 if name.isupper()}
        kwargs = {**attrs, **kwargs}

        super().__init__(inputs=inputs, **kwargs)



class XceptionBlock(DefaultBlock):
    """ Xception building block.
    François Chollet. "`Xception: Deep Learning with Depthwise Separable Convolutions
    <https://arxiv.org/abs/1610.02357>`_"
    """
    LAYOUT = 'R' + 'wnacna'*3 + '&'
    FILTERS = 'same'
    STRIDES = [1]*6
    BRANCH_END = {'strides': 1}



class VGGBlock(ConvBlock):
    """ Convenient VGG block.

    Parameters
    ----------
    depth3 : int
        Number of 3x3 convolutions.
    depth1 : int
        Number of 1x1 convolutions.
    """
    def __init__(self, inputs=None, layout='cna', filters=None, depth3=1, depth1=0, **kwargs):
        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        layout = layout * (depth3 + depth1)
        kernels = [3]*depth3 + [1]*depth1
        super().__init__(inputs=inputs, layout=layout, filters=filters, kernel_size=kernels, **kwargs)



class ResBlock(ConvBlock):
    """ ResNet Module: pass tensor through one or multiple (`n_reps`) blocks, each of which is a
    configurable residual layer, potentially including downsampling, bottleneck, squeeze-and-excitation and groups.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation.
        See more in :class:`~.layers.conv_block.BaseConvBlock` documentation. Default is 'cna cna'.
    filters : int, str, list of int, list of str
        If `str`, then number of filters is calculated by its evaluation. ``'S'`` and ``'same'`` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor. Default value is 'same'.
    kernel_size : int, list of int
        Convolution kernel size. Default is 3.
    strides : int, list of int
        Convolution stride. Default is 1.
    downsample : int, bool
        If int, the first repetition of block will use downsampling with that factor.
        If True, the first repetition of block will use downsampling with a factor of 2.
        If False, then no downsampling. Default is False.
    bottleneck : bool, int
        If True, then add a canonical bottleneck (1x1 conv-batchnorm-activation) with that factor of filters increase.
        If False, then bottleneck is not used. Default is False.
    attention : None, bool or str
        If None or False, then nothing is added. Default is False.
        If True, then add a squeeze-and-excitation block.
        If str, then any of allowed self-attentions. For more info about possible operations,
        check :class:`~.layers.SelfAttention`.
    groups : int
        Use `groups` convolution side by side, each  seeing 1 / `groups` the input channels,
        and producing 1 / `groups` the output channels, and both subsequently concatenated.
        Number of `inputs` channels must be divisible by `groups`. Default is 1.
    op : str or callable
        Operation for combination shortcut and residual.
        See more :class:`~.layers.Combine` documentation. Default is '+a'.
    n_reps : int
        Number of times to repeat the whole block. Default is 1.
    kwargs : dict
        Other named arguments for the :class:`~.layers.ConvBlock`
    """
    def __init__(self, inputs=None, layout='cnacn', filters='same', kernel_size=3, strides=1,
                 downsample=False, bottleneck=False, attention=None, groups=1, op='+a', n_reps=1, **kwargs):
        num_convs = sum(letter in CONV_LETTERS for letter in layout)

        filters = [filters] * num_convs if isinstance(filters, (int, str)) else filters
        filters = [safe_eval(item, get_num_channels(inputs)) if isinstance(item, str) else item
                   for item in filters]

        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        strides = [strides] * num_convs if isinstance(strides, int) else strides
        groups = [groups] * num_convs
        branch_stride = np.prod(strides)

        # Used in the first repetition of the block.
        # Different from strides and branch_stride in other blocks if `downsample` is not ``False``.
        strides_downsample = list(strides)
        branch_stride_downsample = int(branch_stride)

        # Parse all the parameters
        if downsample:
            # The first repetition of the block optionally downsamples inputs
            downsample = 2 if downsample is True else downsample
            strides_downsample[0] *= downsample
            branch_stride_downsample *= downsample
        if bottleneck:
            # Bottleneck: apply 1x1 conv before and after main flow computations to change number of filters
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout + 'acn'
            kernel_size = [1] + kernel_size + [1]
            strides = [1] + strides + [1]
            strides_downsample = [1] + strides_downsample + [1]
            groups = [1] + groups + [1]
            filters = [filters[0]] + filters + [filters[0] * bottleneck]
        if attention:
            # Attention: add self-attention to the main flow
            layout += 'S'
        if get_num_channels(inputs) != filters[-1]:
            # If main flow changes the number of filters, so must do the side branch.
            # No activation, because it will be applied after summation with the main flow
            branch_params = {'layout': 'cn', 'filters': filters[-1],
                             'kernel_size': 1, 'strides': branch_stride_downsample}
        else:
            branch_params = {}
        layout = 'R' + layout + op

        # Pass optional downsample parameters both to the main flow and to the side branch:
        # Only the first repetition is to be changed
        layer_params = [{'strides': strides_downsample,
                         'branch': branch_params,
                         'branch/strides': branch_stride_downsample}]
        layer_params += [{}]*(n_reps-1)

        super().__init__(*layer_params, inputs=inputs, layout=layout, filters=filters,
                         kernel_size=kernel_size, strides=strides, groups=groups, attention=attention,
                         **kwargs)


class MBConvBlock(ConvBlock):
    """ Inverted Recidual, or MBConv Module: pass tensor through one or multiple (`n_reps`) blocks,
    each of which has a sequence of narrow-wide-narrow layers, possibly with SE block in between.
    Skip connections are added if input shape is equal to output shape.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    expand_ratio: int
        number of filters in the inner wide layer is `expand_ratio` times greater than number of input filters
    filters : int, str
        If `str`, then number of filters is calculated by its evaluation. ``'S'`` and ``'same'`` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor. Default value is 'same'.
    kernel_size : int
        Convolution kernel size. Default is 3.
    strides : int
        Convolution stride. Default is 1.
    attention : None, bool or str
        If None or False, then nothing is added. Default is False.
        If True, then add a squeeze-and-excitation block.
        If str, then any of allowed self-attentions.
        For more info about possible operations, check :class:`~.layers.SelfAttention`.
    n_reps : int
        Number of times to repeat the whole block. Default is 1.
    kwargs : dict
        Other named arguments for the :class:`~.layers.ConvBlock`
    """
    def __init__(self, inputs=None, n_reps=1, expand_ratio=6, strides=1, filters='same', kernel_size=3,
                 attention=False, **kwargs):

        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        inp_filters = get_num_channels(inputs)

        layer_params = []
        for k in range(n_reps):
            if k > 0:
                strides = 1

            inner_filters = inp_filters * expand_ratio

            use_res_connect = (strides == 1) and (inp_filters == filters)

            block_params = dict(layout='', filters=[], kernel_size=[], strides=[])

            if use_res_connect:
                block_params['layout'] += 'R'

            if expand_ratio != 1:
                block_params['layout'] += 'cna'
                block_params['filters'].append(inner_filters)
                block_params['kernel_size'].append(1)
                block_params['strides'].append(1)

            block_params['layout'] += 'wna'
            block_params['filters'].append('dummy')
            block_params['kernel_size'].append(kernel_size)
            block_params['strides'].append(strides)

            if attention:
                block_params['layout'] += 'S'
                block_params['attention'] = attention

            block_params['layout'] += 'cn'
            block_params['filters'].append(filters)
            block_params['kernel_size'].append(1)
            block_params['strides'].append(1)


            if use_res_connect:
                block_params['layout'] += '+'

            layer_params.append(block_params)

            inp_filters = filters

        super().__init__(*layer_params, inputs=inputs, **kwargs)


InvResBlock = MBConvBlock


class DenseBlock(ConvBlock):
    """ DenseBlock module.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation.
        See more in :class:`~.layers.conv_block.BaseConvBlock` documentation. Default is 'nacd'.
    num_layers : int
        Number of consecutive layers to make. Each layer is made upon all of the previous tensors concatted together.
    growth_rate : int
        Amount of filters added after each layer.
    skip : bool
        Whether to concatenate inputs to the output result.
    bottleneck : bool, int
        If True, then add a canonical bottleneck (1x1 conv-batchnorm-activation) with that factor of filters increase.
        If False, then bottleneck is not used. Default is False.
    filters : int or None
        If int and is bigger than number of channels in the input tensor, then `growth_rate` is adjusted so
        that the number of output features is that number.
        If int and is smaller or equal to the number of channels in the input tensor, then `growth_rate` is adjusted so
        that the number of added output features is that number.
        If None, then is not used.
    """
    def __init__(self, inputs=None, layout='nacd', filters=None, kernel_size=3, strides=1, dropout_rate=0.2,
                 num_layers=4, growth_rate=12, skip=True, bottleneck=False, **kwargs):
        self.skip = skip
        self.input_num_channels = get_num_channels(inputs)

        if filters is not None:
            if isinstance(filters, str):
                filters = safe_eval(filters, get_num_channels(inputs))

            if filters > self.input_num_channels:
                growth_rate = (filters - self.input_num_channels) // num_layers
            else:
                growth_rate = filters // num_layers
        filters = growth_rate

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout
            kernel_size = [1, kernel_size]
            strides = [1, strides]
            filters = [growth_rate * bottleneck, filters]

        layout = 'R' + layout + '|'
        super().__init__(layout=layout, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate,
                         filters=filters, n_repeats=num_layers, inputs=inputs, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return x if self.skip else x[:, self.input_num_channels:]


class ResNeStBlock(ConvBlock):
    """ ResNeSt Module: apply following operations to a supplied tensor:
        * First of all, it goes through 1x1 convolution with normalization and activation.
        * Then Split Attention Convolution is applied:
            * Here, we split feature maps into `cardinality`*`radix` groups and apply convolution with
            kernel_size=`kernel_size` with normalization and activation (operations are controlled by the `layout`)
            * Then we split the result into `radix` groups.
            * Then attention takes place:
                * Here, we summarize feature maps by groups and apply Global Average Pooling.
                * Then we apply two 1x1 convolutions with groups=`cardinality`. The number of filters in the first
                convolution is `filters`*`radix` // `reduction_factor`.
                * Then we use RadixSoftmax, which applies a softmax for feature maps grouped into `radix` groups.
                * Then, the resulting groups are summed up with feature maps before the attention part.
        * The last layer of the block is a 1x1 convolution that increases the feature map from `filters` to
        `filters`*`scaling_factor`.
        * In the end, skip connection applies to the result of the ResNeStBlock.

    The number of filters inside ResNeSt Attention calculates as following:
    >>> filters = int(filters // reduction_factor) * cardinality

    The implementation is inspired by the authors' code (`<https://github.com/zhanghang1989/ResNeSt>`_), thus
    the first 1x1 convolution is not split into groups, the second fully connected block does not contain
    normalization.

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        A sequence of letters, each letter meaning individual operation.
        See more in :class:`~.layers.conv_block.BaseConvBlock` documentation.
        `layout` describes a first convolution in the :class:`~.attention.SplitAttentionConv`.
        Default is 'cna'.
    filters : int, str
        If `str`, then number of filters is calculated by its evaluation. ``'S'`` and ``'same'`` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor. Default value is 'same'.
    kernel_size : int, list of int
        Convolution kernel size. Default is 3.
    radix : int
        The number of splits within a cardinal group. Default is 2.
    cardinality : int
        The number of feature-map groups. Given feature-map is splitted to groups with same size. Default is 1.
    strides : int, list of int
        Convolution stride. Default is 1.
    reduction_factor : int
        Factor of the filter reduction during :class:`~.attention.SplitAttentionConv`. Default is 1.
    scaling_factor : int
        Factor increasing the number of filters after ResNeSt block. Thus, the number of output filters is
        `filters`*`scaling_factor`. Default 1.
    op : str or callable
        Operation for combination shortcut and residual.
        See more :class:`~.layers.Combine` documentation. Default is '+a'.
    n_reps : int
        Number of times to repeat the whole block. Default is 1.
    kwargs : dict
        Other named arguments for the first :class:`~.layers.ConvBlock` in an :class:`~.attention.SplitAttentionConv`.
    """
    def __init__(self, inputs=None, layout='cna', filters='same', kernel_size=3, radix=2, cardinality=1,
                 strides=1, reduction_factor=1, scaling_factor=1, op='+a', n_reps=1, **kwargs):
        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        # Multiplying by `cardinality` is needed to maintain the possibility of division into groups within
        # `SplitAttentionConv`.
        block_filters = int(filters // reduction_factor) * cardinality

        if get_num_channels(inputs) != (filters*scaling_factor):
            # If main flow changes the number of filters, so must do the side branch.
            # No activation, because it will be applied after summation with the main flow
            branch_params = {'layout': 'cn', 'filters': filters*scaling_factor,
                             'kernel_size': 1, 'strides': strides}
        else:
            branch_params = {}
        layout = 'R' + 'cnaScn' + op

        layer_params = [{'branch': branch_params,
                         'branch/strides': strides}]

        layer_params += [{}]*(n_reps-1)

        # All given parameters are going directly to attention module due to the lack of other operations in the block.
        self_attention = {
            "radix": radix, "cardinality": cardinality,
            "kernel_size": kernel_size, "filters": block_filters, "strides": strides,
            "reduction_factor": reduction_factor, "scaling_factor": scaling_factor,
            **kwargs
        }

        super().__init__(*layer_params, inputs=inputs, layout=layout, attention='sac',
                         self_attention=self_attention, kernel_size=1,
                         filters=[block_filters, filters*scaling_factor])
