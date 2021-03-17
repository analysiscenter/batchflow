""" Contains attention layers.
Note that we can't import :class:`~.layers.ConvBlock` directly due to recursive imports.
"""
import numpy as np
import torch
import torch.nn as nn

from .conv import Conv
from .resize import Combine
from .pooling import GlobalPool, ChannelPool
from .activation import RadixSoftmax, Activation
from ..utils import get_shape, get_num_dims, get_num_channels, safe_eval



class SelfAttention(nn.Module):
    """ Attention based on tensor itself.

    Parameters
    ----------
    attention_mode : str or callable
        If callable, then directly applied to the input tensor.
        If str, then one of predefined attention layers:
            If `se`, then squeeze and excitation.
            Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

            If `scse`, then concurrent spatial and channel squeeze and excitation.
            Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
            in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

            If `ssa`, then simple self attention.
            Wang Z. et al. "'Less Memory, Faster Speed: Refining Self-Attention Module for Image
            Reconstruction <https://arxiv.org/abs/1905.08008>'_"

            If `bam`, then bottleneck attention module.
            Jongchan Park. et al. "'BAM: Bottleneck Attention Module
            <https://arxiv.org/abs/1807.06514>'_"

            If `cbam`, then convolutional block attention module.
            Sanghyun Woo. et al. "'CBAM: Convolutional Block Attention Module
            <https://arxiv.org/abs/1807.06521>'_"

            If `fpa`, then feature pyramid attention.
            Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang.
            Pyramid Attention Network for Semantic Segmentation <https://arxiv.org/abs/1805.10180>'_"

            If `sac`, then split attention.
            Hang Zhang et al. "`ResNeSt: Split-Attention Networks
            <https://arxiv.org/abs/2004.08955>`_"
    """
    @staticmethod
    def identity(inputs, **kwargs):
        """ Return tensor unchanged. """
        _ = inputs, kwargs
        return nn.Identity()

    @staticmethod
    def squeeze_and_excitation(inputs, ratio=4, **kwargs):
        """ Squeeze and excitation. """
        return SEBlock(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def scse(inputs, ratio=2, **kwargs):
        """ Concurrent spatial and channel squeeze and excitation. """
        return SCSEBlock(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def ssa(inputs, ratio=8, **kwargs):
        """ Simple Self Attention. """
        return SimpleSelfAttention(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def bam(inputs, ratio=16, **kwargs):
        """ Bottleneck Attention Module. """
        return BAM(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def cbam(inputs, ratio=16, **kwargs):
        """ Convolutional Block Attention Module. """
        return CBAM(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def fpa(inputs, pyramid_kernel_size=(7, 5, 3), bottleneck=False, **kwargs):
        """ Feature Pyramid Attention. """
        return FPA(inputs=inputs, pyramid_kernel_size=pyramid_kernel_size, bottleneck=bottleneck, **kwargs)

    @staticmethod
    def sac(inputs, radix=2, cardinality=1, **kwargs):
        """ Split-Attention Block. """
        return SplitAttentionConv(inputs=inputs, radix=radix, cardinality=cardinality, **kwargs)

    ATTENTIONS = {
        squeeze_and_excitation: ['se', 'squeeze_and_excitation', 'SE', True],
        scse: ['scse', 'SCSE'],
        ssa: ['ssa', 'SSA'],
        bam: ['bam', 'BAM'],
        cbam: ['cbam', 'CBAM'],
        fpa: ['fpa', 'FPA'],
        identity: ['identity', None, False],
        sac: ['sac', 'SAC']
    }
    ATTENTIONS = {alias: getattr(method, '__func__') for method, aliases in ATTENTIONS.items() for alias in aliases}

    def __init__(self, inputs=None, attention='se', **kwargs):
        super().__init__()
        self.attention = attention

        if attention in self.ATTENTIONS:
            op = self.ATTENTIONS[attention]
            self.op = op(inputs, **kwargs)
        elif callable(attention):
            self.op = attention(inputs, **kwargs)
        else:
            raise ValueError('Attention mode must be a callable or one from {}, instead got {}.'
                             .format(list(self.ATTENTIONS.keys()), attention))

    def forward(self, inputs):
        return self.op(inputs)

    def extra_repr(self):
        """ Report used attention in a repr. """
        if isinstance(self.attention, (str, bool)):
            return 'op={}'.format(self.attention)
        return 'op=callable {}'.format(self.attention.__name__)



class SEBlock(nn.Module):
    """ Squeeze and excitation block.
    Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of filters.
    """
    def __init__(self, inputs=None, ratio=4, bias=False, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        in_units = get_shape(inputs)[1]
        units = [in_units // ratio, in_units]
        activations = ['relu', 'sigmoid']
        kwargs = {'layout': 'Vfafa >',
                  'units': units, 'activation': activations,
                  'dim': get_num_dims(inputs),
                  'bias': bias,
                  **kwargs}
        self.layer = ConvBlock(inputs=inputs, **kwargs)

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'units': units,
            'ratio': ratio,
            'bias': bias,
        }

    def forward(self, x):
        return Combine.mul((x, self.layer(x)))

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(units={units}, ratio={ratio}, bias={bias})'
                      .format(**self.desc_kwargs))
        return layer_desc


class SCSEBlock(nn.Module):
    """ Concurrent spatial and channel squeeze and excitation.
    Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

    Parameters
    ----------
    ratio : int, optional
        Squeeze ratio for the number of filters.
    """
    def __init__(self, inputs=None, ratio=2, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()

        self.cse = SEBlock(inputs=inputs, ratio=ratio, **kwargs)
        kwargs = {'layout': 'ca',
                  'filters': 1, 'kernel_size': 1, 'activation': 'sigmoid',
                  **kwargs}
        self.sse = ConvBlock(inputs=inputs, **kwargs)

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'ratio': ratio,
        }

    def forward(self, x):
        cse = self.cse(x)
        sse = self.sse(x)
        return Combine.sum((cse, Combine.mul((x, sse))))

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(ratio={ratio})'
                      .format(**self.desc_kwargs))
        return layer_desc



class SimpleSelfAttention(nn.Module):
    """ Improved self Attention module.

    Wang Z. et al. "'Less Memory, Faster Speed: Refining Self-Attention Module for Image
    Reconstruction <https://arxiv.org/abs/1905.08008>'_"

    Parameters
    ----------
    reduction_ratio : int
        The reduction ratio of filters in the inner convolutions.
    kernel_size : int
        Kernel size.
    layout : str
        Layout for convolution layers.
    """
    def __init__(self, inputs=None, layout='cna', kernel_size=1, ratio=8, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, device=inputs.device))

        args = {**kwargs, **dict(inputs=inputs, layout=layout, kernel_size=kernel_size)}
        self.top_branch = ConvBlock(**args, filters='same//{}'.format(ratio))
        self.mid_branch = ConvBlock(**args, filters='same//{}'.format(ratio))
        self.bot_branch = ConvBlock(**args, filters='same')

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'layout': layout,
            'kernel_size': kernel_size,
            'ratio': ratio,
        }

    def forward(self, x):
        batch_size, spatial = x.shape[0], x.shape[2:]
        num_features = np.prod(spatial)

        phi = self.mid_branch(x).view(batch_size, -1, num_features) # (B, C/8, N)
        theta = self.bot_branch(x).view(batch_size, num_features, -1) # (B, N, C)
        attention = torch.bmm(phi, theta) / num_features # (B, C/8, C)

        out = self.top_branch(x).view(batch_size, num_features, -1) # (B, N, C/8)
        out = torch.bmm(out, attention).view(batch_size, -1, *spatial)
        return self.gamma*out + x

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(layout={layout}, kernel_size={kernel_size}, ratio={ratio})'
                      .format(**self.desc_kwargs))
        return layer_desc



class BAM(nn.Module):
    """ Bottleneck Attention Module.

    Jongchan Park. et al. "'BAM: Bottleneck Attention Module
    <https://arxiv.org/abs/1807.06514>'_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of filters.
        Default is 16.
    dilation_rate : int
        The dilation rate in the convolutions in the spatial attention submodule.
        Default is 4.
    """
    def __init__(self, inputs=None, ratio=16, dilation_rate=4, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        in_channels = get_num_channels(inputs)

        self.bam_attention = ConvBlock(
            inputs=inputs, layout='R' + 'cna'*3  + 'c' + '+ a',
            filters=['same//{}'.format(ratio), 'same', 'same', 1], kernel_size=[1, 3, 3, 1],
            dilation_rate=[1, dilation_rate, dilation_rate, 1], activation=['relu']*3+['sigmoid'], bias=True,
            branch={'layout': 'Vfnaf >', 'units': [in_channels//ratio, in_channels], 'dim': get_num_dims(inputs),
                    'activation': 'relu', 'bias': True, **kwargs})

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_filters': in_channels,
            'out_filters': in_channels,
            'dilation_rate': dilation_rate,
            'ratio': ratio
        }

    def forward(self, x):
        return x * (1 + self.bam_attention(x))

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'dilation_rate={dilation_rate}, ratio={ratio})').format(**self.desc_kwargs)
        return layer_desc



class CBAM(nn.Module):
    """ Convolutional Block Attention Module.

    Sanghyun Woo. et al. "'CBAM: Convolutional Block Attention Module
    <https://arxiv.org/abs/1807.06521>'_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of filters.
        Default is 16.
    pool_ops : list of str
        Pooling operations for channel_attention module.
        Default is `('avg', 'max')`.
    """
    def __init__(self, inputs=None, ratio=16, pool_ops=('avg', 'max'), **kwargs):
        super().__init__()
        self.channel_attention(inputs, ratio, pool_ops, **kwargs)
        self.spatial_attention(inputs, **kwargs)
        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_filters': get_num_channels(inputs),
            'out_filters': get_num_channels(inputs),
            'pool_ops': pool_ops,
            'ratio': ratio
        }

    def channel_attention(self, inputs, ratio, pool_ops, **kwargs):
        """ Channel attention module."""
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        self.pool_layers = []
        num_dims = get_num_dims(inputs)
        num_channels = get_num_channels(inputs)

        for pool_op in pool_ops:
            pool = GlobalPool(inputs=inputs, op=pool_op)
            self.pool_layers.append(pool)

        tensor = self.pool_layers[0](inputs)
        self.shared_layer = ConvBlock(inputs=tensor, layout='faf>',
                                      units=[num_channels // ratio, num_channels],
                                      activation='relu', dim=num_dims, **kwargs)

        self.combine_cam = Combine(op='sum')

    def spatial_attention(self, inputs, **kwargs):
        """ Spatial attention module."""
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        self.combine_sam = Combine(op='concat')
        cat_features = self.combine_sam([ChannelPool(op='mean')(inputs),
                                         ChannelPool(op='max')(inputs)])
        self.sam = ConvBlock(inputs=cat_features, layout='cna', filters=1, kernel_size=7,
                             activation='sigmoid', **kwargs)

    def forward(self, x):
        tensor_list = []
        for pool in self.pool_layers:
            pool_feature = pool(x)
            tensor = self.shared_layer(pool_feature)
            tensor_list.append(tensor)
        tensor = self.combine_cam(tensor_list)
        attention = Activation('sigmoid')(tensor)
        x = x * attention
        cat_features = self.combine_sam([ChannelPool(op='mean')(x),
                                         ChannelPool(op='max')(x)])
        attention = self.sam(cat_features)
        return x * attention

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'pool_ops={pool_ops}, ratio={ratio})').format(**self.desc_kwargs)
        return layer_desc



class FPA(nn.Module):
    """ Feature Pyramid Attention.
    Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang.
    Pyramid Attention Network for Semantic Segmentation <https://arxiv.org/abs/1805.10180>'_"

    Parameters
    ----------
    pyramid_kernel_size: list of ints
        Kernel sizes in pyramid block convolutions
    layout: str
        Layout for convolution layers.
    downsample_layout: str
        Layout for downsampling layers. Default is 'p'
    upsample_layout: str
        Layout for upsampling layers. Default is 't
    factor: int
        Scaling factor for upsampling layers. Default is 2
    bottleneck : bool, int
        If True, then add a 1x1 convolutions before and after pyramid block with that factor of filters reduction.
        If False, then bottleneck is not used. Default is False.
    use_dilation: bool
        If True, then the convolutions with bigger kernels in the pyramid block are replaced by top one
        with corresponding dilation_rate, i.e. 5x5 -> 3x3 with dilation=2, 7x7 -> 3x3 with dilation=3.
        If False, the dilated convolutions are not used. Default is False.
    """
    def __init__(self, inputs=None, pyramid_kernel_size=(7, 5, 3), bottleneck=False, layout='cna',
                 downsample_layout='p', upsample_layout='t', factor=2, use_dilation=False, **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        depth = len(pyramid_kernel_size)
        spatial_shape = get_shape(inputs)[2:]
        num_dims = get_num_dims(inputs)

        self.attention = ConvBlock(inputs=inputs, layout='V >' + layout + 'b', kernel_size=1,
                                   filters='same', shape=spatial_shape, dim=num_dims, **kwargs)

        enc_layout = ('B' + downsample_layout + layout) * depth # B pcna B pcna B pcna
        emb_layout = layout + upsample_layout # cnat
        combine_layout = ('+' + upsample_layout) * (depth - 1) + '*' # +t+t*
        main_layout = enc_layout + emb_layout + combine_layout # B pcna B pcna B pcna cnat +t+t*
        main_strides = [1] * (depth + 1) + [factor] * depth # [1, 1, 1, 1, 2, 2, 2]

        # list with args for BaseBlocks of each branch
        branches = [dict(layout=layout, filters='same', kernel_size=1)] # the mid branch from inputs tensor directly

        if use_dilation:
            base_kernel_size = 3
            main_kernel_size = [base_kernel_size] * (depth + 1) + [factor] * (depth) # 3 3 3 3 2 2 2
            # infering coresponding dilation for every kernel_size in pyramid block
            pyramid_dilation = [round((rf - base_kernel_size) / (base_kernel_size - 1)) + 1
                                for rf in pyramid_kernel_size] # 3 2 1
            main_dilation = pyramid_dilation + [pyramid_dilation[-1]] + [1] * depth # 3 2 1 1 1 1 1
            for d in pyramid_dilation[:-1]:
                args = dict(layout=layout, kernel_size=base_kernel_size, dilation_rate=d, filters='same')
                branches.append(args)
        else:
            main_kernel_size = list(pyramid_kernel_size) + [pyramid_kernel_size[-1]] + [factor] * (depth) # [7533222]
            main_dilation = 1
            for kernel_size in pyramid_kernel_size[:-1]:
                args = dict(layout=layout, kernel_size=kernel_size, filters='same')
                branches.append(args)

        pyramid_args = {'layout': main_layout, 'filters': 'same', 'kernel_size': main_kernel_size,
                        'strides': main_strides, 'dilation_rate': main_dilation, 'branch': branches}

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            out_filters = get_num_channels(inputs)
            inner_filters = out_filters // bottleneck
            self.pyramid = ConvBlock(dict(layout=layout, kernel_size=1, filters=inner_filters),
                                     pyramid_args,
                                     dict(layout=layout, kernel_size=1, filters=out_filters), inputs=inputs, **kwargs)
        else:
            self.pyramid = ConvBlock(pyramid_args, inputs=inputs, **kwargs)

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'pyramid_kernel_size': pyramid_kernel_size,
            'bottleneck': bottleneck,
            'use_dilation': use_dilation,
            'factor': factor
        }

    def forward(self, x):
        attention = self.attention(x)
        main = self.pyramid(x)
        return Combine.sum([attention, main])

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(pyramid_kernel_size={pyramid_kernel_size}, bottleneck={bottleneck}, '
                      'use_dilation={use_dilation}, factor={factor})').format(**self.desc_kwargs)
        return layer_desc



class SelectiveKernelConv(nn.Module):
    """ Selective Kernel Convolution.

    Xiang Li. et al. "'Selective Kernel Networks
    <https://arxiv.org/abs/1903.06586>'_"

    Parameters
    ----------
    kernels : tuple of int
        Tuple of kernel_sizes for branches in split part.
        Default is `(3, 5)`.
    use_dilation : bool
        If ``True``, then convolution in split part uses instead of the `kernel_size`
        from the `kernels` the `kernel_size=3` and the appropriate dilation rate.
        If ``False``, then dilated convolutions are not used. Default is ``False``.
    min_units: int
        Minimum length of fused vector. Default is 32.
    """

    def __init__(self, filters, kernels=(3, 5), strides=1, padding='same',
                 use_dilation=False, groups=1, bias=False, ratio=4, min_units=32, inputs=None):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()

        if isinstance(filters, str):
            filters = safe_eval(filters, get_num_channels(inputs))

        num_kernels = len(kernels)
        num_dims = get_num_dims(inputs)

        if use_dilation:
            dilations = tuple((kernel - 3) // 2 + 1 for kernel in kernels)
            kernels = (3,) * num_kernels
        else:
            dilations = (1,) * num_kernels

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_filters': get_num_channels(inputs),
            'out_filters': filters,
            'kernel_sizes': kernels,
            'dilations': dilations
        }

        self.split_layers = nn.ModuleList()
        tensors = []
        for kernel_size, dilation_rate in zip(kernels, dilations):
            branch = ConvBlock(inputs=inputs, layout='cna', filters=filters,
                               kernel_size=kernel_size, strides=strides, padding=padding,
                               dilation_rate=dilation_rate, groups=groups, bias=bias)
            self.split_layers.append(branch)
            tensors.append(branch(inputs))

        self.combine = Combine(op='sum')
        tensor = self.combine(tensors)
        self.fuse = ConvBlock(inputs=tensor, layout='Vfna>', units='max(same // {}, {})'.format(ratio, min_units),
                              dim=num_dims, bias=bias)

        fused_tensor = self.fuse(tensor)
        self.attention_branches = nn.ModuleList([
            Conv(inputs=fused_tensor, filters=filters, kernel_size=1, bias=bias) for i in range(num_kernels)])

    def forward(self, x):
        tensors = [layer(x) for layer in self.split_layers]
        tensor = self.combine(tensors)
        fused_tensor = self.fuse(tensor)
        attention_vectors = torch.stack([attention(fused_tensor) for attention in self.attention_branches], dim=-1)
        attention_vectors = nn.Softmax(dim=-1)(attention_vectors)
        attention_vectors = [attention_vectors[..., idx] for idx in range(attention_vectors.shape[-1])]

        result = [tensor * attention for tensor, attention in zip(tensors, attention_vectors)]
        result = torch.stack(result, dim=-1).sum(-1)
        return result

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'kernel_sizes={kernel_sizes}, dilations={dilations})').format(**self.desc_kwargs)
        return layer_desc


class SplitAttentionConv(nn.Module):
    """ Split Attention.

    Hang Zhang et al. "`ResNeSt: Split-Attention Networks
    <https://arxiv.org/abs/2004.08955>`_"

    This block contains the following operations:
    * First of all, we split feature maps into `cardinality`*`radix` groups and apply convolution with
      kernel_size=`kernel_size` with normalization and activation (these operations are controlled by the `layout`)
    * Then we split the result into `cardinality` groups.
    * Then attention takes place:
        * Here, we summarize feature maps by groups and apply Global Average Pooling.
        * Then we apply two 1x1 convolutions with groups=`cardinality`. The number of filters in the first
            convolution is `filters`*`radix` // `reduction_factor`.
        * Then we use RadixSoftmax, which applies a softmax for feature maps grouped into `cardinality` groups.
        * Then, the resulting groups are summed up with feature maps before the attention part.

    Parameters
    ----------
    radix : int
        The number of splits within a cardinal group. Default is 2.
    cardinality : int
        The number of feature-map groups. Given feature-map is splitted to groups with same size. Default is 1.
    reduction_factor : int
        Factor of the filter reduction during :class:`~.attention.SplitAttentionConv`. Default is 1.
    scaling_factor : int
        Factor increasing the number of filters after ResNeSt block. Thus, the number of output filters is
        `filters`*`scaling_factor`. Default 1.
    kwargs : dict
        Other named arguments only for the first :class:`~.layers.ConvBlock`.
    """
    def __init__(self, inputs, filters, layout='cna', radix=1, cardinality=1, reduction_factor=1, scaling_factor=1,
                 strides=1, padding='same', **kwargs):
        from .conv_block import ConvBlock # can't be imported in the file beginning due to recursive imports
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = filters * self.radix

        self.inner_filters = self.channels // reduction_factor
        channel_dim = inputs.dim() - 2
        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_filters': get_num_channels(inputs),
            'out_filters': filters*scaling_factor,
            'radix': self.radix,
            'cardinality': self.cardinality,
            'reduction_factor': reduction_factor,
            'scaling_factor': scaling_factor
        }

        self.inner_radix_conv = ConvBlock(inputs=inputs, layout=layout, filters=self.channels,
                                          groups=self.cardinality*self.radix, strides=strides, padding=padding,
                                          **kwargs)
        inputs = self.inner_radix_conv(inputs)
        x = inputs

        rchannel = inputs.shape[1]
        if self.radix > 1:
            splitted = torch.split(inputs, rchannel//self.radix, dim=1)
            inputs = sum(splitted)

        inner_conv1d_layout = 'V>' + 'cnac'
        self.avgpool_conv1d = ConvBlock(inputs=inputs, layout=inner_conv1d_layout,
                                        filters=[self.inner_filters, self.channels],
                                        kernel_size=1, groups=self.cardinality, dim=channel_dim,
                                        bias=True)
        inputs = self.avgpool_conv1d(inputs)

        self.rsoftmax = RadixSoftmax(self.radix, self.cardinality)
        inputs = self.rsoftmax(inputs)

        if self.radix > 1:
            inputs = torch.split(inputs, rchannel//self.radix, dim=1)
            inputs = sum([inp*split for (inp, split) in zip(inputs, splitted)])
        else:
            inputs = inputs * x

    def forward(self, x):
        x = self.inner_radix_conv(x)
        rchannel = x.shape[1]
        if self.radix > 1:
            splitted = torch.split(x, rchannel//self.radix, dim=1)
            concatted = sum(splitted)
        else:
            concatted = x
        att = self.avgpool_conv1d(concatted)
        att = self.rsoftmax(att)
        if self.radix > 1:
            attens = torch.split(att, rchannel//self.radix, dim=1)
            result = sum([att*split for (att, split) in zip(attens, splitted)])
        else:
            result = att * x
        return result

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'radix={radix}, cardinality={cardinality}, '
                      'reduction_factor={reduction_factor}, '
                      'scaling_factor={scaling_factor})').format(**self.desc_kwargs)
        return layer_desc
