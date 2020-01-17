""" Functional modules for various deep network architectures."""
import numpy as np
import torch
import torch.nn as nn

from .layers import ConvBlock, Upsample, Combine, Pool, ChannelPool, Activation, Conv
from .utils import get_shape, get_num_dims, get_num_channels, safe_eval


class PyramidPooling(nn.Module):
    """ Pyramid Pooling module
    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        Sequence of operations in convolution layer.
    filters : int
        Number of filters in pyramid branches.
    kernel_size : int
        Kernel size.
    pool_op : str
        Pooling operation ('mean' or 'max').
    pyramid : tuple of int
        Number of feature regions in each dimension.
        `0` is used to include `inputs` into the output tensor.
    """
    def __init__(self, inputs, layout='cna', filters=None, kernel_size=1, pool_op='mean',
                 pyramid=(0, 1, 2, 3, 6), **kwargs):
        super().__init__()

        spatial_shape = np.array(get_shape(inputs)[2:])
        filters = filters if filters else 'same // {}'.format(len(pyramid))

        modules = nn.ModuleList()
        for level in pyramid:
            if level == 0:
                module = nn.Identity()
            else:
                x = inputs
                pool_size = tuple(np.ceil(spatial_shape / level).astype(np.int32).tolist())
                pool_strides = tuple(np.floor((spatial_shape - 1) / level + 1).astype(np.int32).tolist())

                layer = ConvBlock(inputs=x, layout='p' + layout, filters=filters, kernel_size=kernel_size,
                                  pool_op=pool_op, pool_size=pool_size, pool_strides=pool_strides, **kwargs)
                x = layer(x)

                upsample_layer = Upsample(inputs=x, factor=None, layout='b',
                                          shape=tuple(spatial_shape.tolist()), **kwargs)
                module = nn.Sequential(layer, upsample_layer)
            modules.append(module)

        self.blocks = modules
        self.combine = Combine(op='concat')

    def forward(self, x):
        levels = [layer(x) for layer in self.blocks]
        return self.combine(levels)


class ASPP(nn.Module):
    """ Atrous Spatial Pyramid Pooling module.

    Chen L. et al. "`Rethinking Atrous Convolution for Semantic Image Segmentation
    <https://arxiv.org/abs/1706.05587>`_"

    Parameters
    ----------
    layout : str
        Layout for convolution layers.
    filters : int
        Number of filters in the output tensor.
    kernel_size : int
        Kernel size for dilated branches.
    rates : tuple of int
        Dilation rates for branches, default=(6, 12, 18).
    pyramid : int or tuple of int
        Number of image level features in each dimension.
        Default is 2, i.e. 2x2=4 pooling features will be calculated for 2d images,
        and 2x2x2=8 features per 3d item.
        Tuple allows to define several image level features, e.g (2, 3, 4).

    See also
    --------
    PyramidPooling
    """
    def __init__(self, inputs=None, layout='cna', filters='same', kernel_size=3,
                 rates=(6, 12, 18), pyramid=2, **kwargs):
        super().__init__()
        pyramid = pyramid if isinstance(pyramid, (tuple, list)) else [pyramid]

        modules = nn.ModuleList()
        bottleneck = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=1, **kwargs)
        modules.append(bottleneck)

        for level in rates:
            layer = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=kernel_size,
                              dilation_rate=level, **kwargs)
            modules.append(layer)

        pyramid_layer = PyramidPooling(inputs=inputs, filters=filters, pyramid=pyramid, **kwargs)
        modules.append(pyramid_layer)

        self.blocks = modules
        self.combine = Combine(op='concat')

    def forward(self, x):
        levels = [layer(x) for layer in self.blocks]
        return self.combine(levels)


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

        self.combine = Combine(op='+')

    def forward(self, x):
        attention = self.attention(x)
        main = self.pyramid(x)
        return self.combine([attention, main])


class SelfAttention(nn.Module):
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
    def __init__(self, inputs=None, layout='cna', kernel_size=1, reduction_ratio=8, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, device=inputs.device))
        args = {**kwargs, **dict(inputs=inputs, layout=layout, kernel_size=kernel_size)}
        self.top_branch = ConvBlock(**args, filters='same//{}'.format(reduction_ratio))
        self.mid_branch = ConvBlock(**args, filters='same//{}'.format(reduction_ratio))
        self.bot_branch = ConvBlock(**args, filters='same')

    def forward(self, x):
        batch_size, spatial = x.shape[0], x.shape[2:]
        num_features = np.prod(spatial)

        phi = self.mid_branch(x).view(batch_size, -1, num_features) # (B, C/8, N)
        theta = self.bot_branch(x).view(batch_size, num_features, -1) # (B, N, C)
        attention = torch.bmm(phi, theta) / num_features # (B, C/8, C)

        out = self.top_branch(x).view(batch_size, num_features, -1) # (B, N, C/8)
        out = torch.bmm(out, attention).view(batch_size, -1, *spatial)
        return self.gamma*out + x


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
        super().__init__()
        self.bam_attention = ConvBlock(
            inputs=inputs, layout='S' + 'cna'*4 + '+ a',
            filters=['same//{}'.format(ratio), 'same', 'same', 1], kernel_size=[1, 3, 3, 1],
            dilation_rate=[1, dilation_rate, dilation_rate, 1], activation=['relu']*4+['sigmoid'],
            squeeze_layout='Vfnaf', ratio=ratio, squeeze_activations='relu', **kwargs)
        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_filters': get_num_channels(inputs),
            'out_filters': get_num_channels(inputs),
            'dilation_rate': dilation_rate,
            'ratio': ratio
        }

    def forward(self, x):
        return x * (1 + self.bam_attention(x))

    def __repr__(self):
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
            'pool_ops': ('avg', 'max'),
            'ratio': ratio
        }

    def channel_attention(self, inputs, ratio, pool_ops, **kwargs):
        " Channel attention module."
        self.pool_layers = []
        num_dims = get_num_dims(inputs)
        num_channels = get_num_channels(inputs)

        for pool_op in pool_ops:
            pool = Pool(inputs=inputs, op=pool_op)
            self.pool_layers.append(pool)

        tensor = self.pool_layers[0](inputs)
        self.shared_layer = ConvBlock(inputs=tensor, layout='faf>',
                                     units=[num_channels // ratio, num_channels],
                                     activation='relu', dim=num_dims, **kwargs)

        self.combine_cam = Combine(op='sum')

    def spatial_attention(self, inputs, **kwargs):
        " Spatial attention module."
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
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'pool_ops={pool_ops}, ratio={ratio})').format(**self.desc_kwargs)
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
    """

    def __init__(self, filters, kernels=(3, 5), strides=1, padding='same',
                 use_dilation=False, groups=1, bias=True, ratio=4, inputs=None):
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
        self.fuse = ConvBlock(inputs=tensor, layout='Vfna>', dim=num_dims, units='same // {}'.format(ratio))

        fused_tensor = self.fuse(tensor)
        self.attention_branches = nn.ModuleList([
            Conv(inputs=fused_tensor, filters=filters, kernel_size=1) for i in range(num_kernels)])

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
        layer_desc = ('{class}({in_filters}, {out_filters}, '
                      'kernel_sizes={kernel_sizes}, dilations={dilations})').format(**self.desc_kwargs)
        return layer_desc
