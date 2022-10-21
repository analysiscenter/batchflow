""" Contains attention layers. """
import numpy as np
import torch
from torch import nn

from einops import rearrange

from .core import Block
from ..layers import Activation, Conv, ChannelPool, RadixSoftmax, Combine
from ..utils import get_shape, get_num_dims, get_num_channels, safe_eval




class SEBlock(nn.Module):
    """ Squeeze and excitation block.
    Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of channels.
    """
    def __init__(self, inputs=None, ratio=4, bias=False, **kwargs):
        super().__init__()
        in_channels = get_num_channels(inputs)
        channels = [in_channels // ratio, in_channels]
        activations = ['relu', 'sigmoid']
        kwargs = {'layout': 'V>caca',
                  'kernel_size': 1,
                  'channels': channels,
                  'activation': activations,
                  'dim': get_num_dims(inputs),
                  'bias': bias,
                  **kwargs}

        self.layer = Block(inputs=inputs, **kwargs)

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'channels': channels,
            'ratio': ratio,
            'bias': bias,
        }

    def forward(self, x):
        return x * self.layer(x)

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(channels={channels}, ratio={ratio}, bias={bias})'
                      .format(**self.desc_kwargs))
        return layer_desc


class SCSEBlock(nn.Module):
    """ Concurrent spatial and channel squeeze and excitation.
    Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
    in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

    Parameters
    ----------
    ratio : int, optional
        Squeeze ratio for the number of channels.
    """
    def __init__(self, inputs=None, ratio=2, **kwargs):
        super().__init__()

        self.cse = SEBlock(inputs=inputs, ratio=ratio, **kwargs)
        kwargs = {'layout': 'ca',
                  'channels': 1, 'kernel_size': 1, 'activation': 'sigmoid',
                  **kwargs}
        self.sse = Block(inputs=inputs, **kwargs)

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'ratio': ratio,
        }

    def forward(self, x):
        cse = self.cse(x)
        sse = self.sse(x)
        return x * sse + cse

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
        The reduction ratio of channels in the inner convolutions.
    kernel_size : int
        Kernel size.
    layout : str
        Layout for convolution layers.
    """
    def __init__(self, inputs=None, layout='cna', kernel_size=1, ratio=8, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, device=inputs.device))

        args = {**kwargs, **dict(inputs=inputs, layout=layout, kernel_size=kernel_size)}
        self.top_branch = Block(**args, channels=f'same//{ratio}')
        self.mid_branch = Block(**args, channels=f'same//{ratio}')
        self.bot_branch = Block(**args, channels='same')

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'layout': layout,
            'kernel_size': kernel_size,
            'ratio': ratio,
        }

    def forward(self, x):
        batch_size, spatial = x.shape[0], x.shape[2:]
        num_features = np.prod(spatial)

        phi = self.mid_branch(x).reshape(batch_size, -1, num_features) # (B, C/8, N)
        theta = self.bot_branch(x).reshape(batch_size, num_features, -1) # (B, N, C)
        attention = torch.bmm(phi, theta) / num_features # (B, C/8, C)

        out = self.top_branch(x).reshape(batch_size, num_features, -1) # (B, N, C/8)
        out = torch.bmm(out, attention).reshape(batch_size, -1, *spatial)
        return self.gamma*out + x

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}(layout={layout}, kernel_size={kernel_size}, ratio={ratio})'
                      .format(**self.desc_kwargs))
        return layer_desc


class EfficientMultiHeadAttention(nn.Module):
    """ Attention layer, popularized by transformer architectures.
    Efficient in a sense of reducing the number of sequence elements `ratio^2` times.
    Reduction is implemented as a convolution, and attention is implemented by a native `PyTorch` layer.

    Parameters
    ----------
    ratio : int
        Spatial reduction ratio. As this is applied across both spatial dimensions,
        the actual reduction in number of sequence elements is `ratio` squared.
    num_heads : int
        Number of parallel attention heads. Must be a divisor of `input` number of channels.
    """
    def __init__(self, inputs=None, ratio=4, num_heads=8):
        super().__init__()
        channels = get_num_channels(inputs)
        self.num_heads = num_heads

        self.reducer = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ratio, stride=ratio)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # Store input shape for later, apply spatial reduction
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)

        # Attention accepts tensor of shape (batch, sequence_length, channels)
        x = rearrange(x, 'b c h w -> b (h w) c')
        reduced_x = rearrange(reduced_x, 'b c h w -> b (h w) c')

        # Apply attention, reshape to the input shape
        out = self.attention(x, reduced_x, reduced_x)[0]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out

    def extra_repr(self):
        return f'num_heads={self.num_heads}'


class BAM(nn.Module):
    """ Bottleneck Attention Module.

    Jongchan Park. et al. "'BAM: Bottleneck Attention Module
    <https://arxiv.org/abs/1807.06514>'_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of channels.
        Default is 16.
    dilation : int
        The dilation rate in the convolutions in the spatial attention submodule.
        Default is 4.
    """
    def __init__(self, inputs=None, ratio=16, dilation=4, **kwargs):
        super().__init__()
        in_channels = get_num_channels(inputs)

        self.bam_attention = Block(
            inputs=inputs, layout='R' + 'cna'*3  + 'c' + '+ a',
            channels=[f'same//{ratio}', 'same', 'same', 1], kernel_size=[1, 3, 3, 1],
            dilation=[1, dilation, dilation, 1], activation=['relu']*3+['sigmoid'], bias=True,
            branch={'layout': 'Vfnaf >',
                    'features': [in_channels//ratio, in_channels],
                    'dim': get_num_dims(inputs),
                    'activation': 'relu', 'bias': True, **kwargs}
        )

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_channels': in_channels,
            'out_channels': in_channels,
            'dilation': dilation,
            'ratio': ratio
        }

    def forward(self, x):
        return x * (1 + self.bam_attention(x))

    def __repr__(self):
        if getattr(self, 'debug', False):
            return super().__repr__()
        layer_desc = ('{class}({in_channels}, {out_channels}, '
                      'dilation={dilation}, ratio={ratio})').format(**self.desc_kwargs)
        return layer_desc



class CBAM(nn.Module):
    """ Convolutional Block Attention Module.

    Sanghyun Woo. et al. "'CBAM: Convolutional Block Attention Module
    <https://arxiv.org/abs/1807.06521>'_"

    Parameters
    ----------
    ratio : int
        Squeeze ratio for the number of channels.
        Default is 16.
    pool_ops : list of str
        Pooling operations for channel_attention module.
        Default is `('avg', 'max')`.
    """
    def __init__(self, inputs=None, ratio=16, pool_ops=('V', 'P'), **kwargs):
        super().__init__()
        self.channel_attention(inputs, ratio, pool_ops, **kwargs)
        self.spatial_attention(inputs, **kwargs)
        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_channels': get_num_channels(inputs),
            'out_channels': get_num_channels(inputs),
            'pool_ops': pool_ops,
            'ratio': ratio
        }

    def channel_attention(self, inputs, ratio, pool_ops, **kwargs):
        """ Channel attention module."""
        self.pool_layers = []
        num_dims = get_num_dims(inputs)
        num_channels = get_num_channels(inputs)

        for pool_op in pool_ops:
            pool = Block(inputs=inputs, layout=pool_op)
            self.pool_layers.append(pool)

        tensor = self.pool_layers[0](inputs)
        self.shared_layer = Block(inputs=tensor, layout='faf>',
                                  features=[num_channels // ratio, num_channels],
                                  activation='relu', dim=num_dims, **kwargs)

        self.combine_cam = Combine(op='sum')

    def spatial_attention(self, inputs, **kwargs):
        """ Spatial attention module."""
        self.combine_sam = Combine(op='concat')
        cat_features = self.combine_sam([ChannelPool(op='mean')(inputs),
                                         ChannelPool(op='max')(inputs)])
        self.sam = Block(inputs=cat_features, layout='cna', channels=1, kernel_size=7,
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
        layer_desc = ('{class}({in_channels}, {out_channels}, '
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
        If True, then add a 1x1 convolutions before and after pyramid block with that factor of channels reduction.
        If False, then bottleneck is not used. Default is False.
    use_dilation: bool
        If True, then the convolutions with bigger kernels in the pyramid block are replaced by top one
        with corresponding dilation, i.e. 5x5 -> 3x3 with dilation=2, 7x7 -> 3x3 with dilation=3.
        If False, the dilated convolutions are not used. Default is False.
    """
    def __init__(self, inputs=None, pyramid_kernel_size=(7, 5, 3), bottleneck=False, layout='cna',
                 downsample_layout='p', upsample_layout='t', factor=2, use_dilation=False, **kwargs):
        super().__init__()
        depth = len(pyramid_kernel_size)
        spatial_shape = get_shape(inputs)[2:]
        num_dims = get_num_dims(inputs)

        self.attention = Block(inputs=inputs, layout='V >' + layout + 'b', kernel_size=1,
                               channels='same', shape=spatial_shape, dim=num_dims, **kwargs)

        enc_layout = ('B' + downsample_layout + layout) * depth # B pcna B pcna B pcna
        emb_layout = layout + upsample_layout # cnat
        combine_layout = ('+' + upsample_layout) * (depth - 1) + '*' # +t+t*
        main_layout = enc_layout + emb_layout + combine_layout # B pcna B pcna B pcna cnat +t+t*
        main_strides = [1] * (depth + 1) + [factor] * depth # [1, 1, 1, 1, 2, 2, 2]

        # list with args for BaseBlocks of each branch
        branches = [dict(layout=layout, channels='same', kernel_size=1)] # the mid branch from inputs tensor directly

        if use_dilation:
            base_kernel_size = 3
            main_kernel_size = [base_kernel_size] * (depth + 1) + [factor] * (depth) # 3 3 3 3 2 2 2
            # infering coresponding dilation for every kernel_size in pyramid block
            pyramid_dilation = [round((rf - base_kernel_size) / (base_kernel_size - 1)) + 1
                                for rf in pyramid_kernel_size] # 3 2 1
            main_dilation = pyramid_dilation + [pyramid_dilation[-1]] + [1] * depth # 3 2 1 1 1 1 1
            for d in pyramid_dilation[:-1]:
                args = dict(layout=layout, kernel_size=base_kernel_size, dilation=d, channels='same')
                branches.append(args)
        else:
            main_kernel_size = list(pyramid_kernel_size) + [pyramid_kernel_size[-1]] + [factor] * (depth) # [7533222]
            main_dilation = 1
            for kernel_size in pyramid_kernel_size[:-1]:
                args = dict(layout=layout, kernel_size=kernel_size, channels='same')
                branches.append(args)

        pyramid_args = {'layout': main_layout, 'channels': 'same', 'kernel_size': main_kernel_size,
                        'stride': main_strides, 'dilation': main_dilation, 'branch': branches}

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            out_channels = get_num_channels(inputs)
            inner_channels = out_channels // bottleneck
            self.pyramid = Block(dict(layout=layout, kernel_size=1, channels=inner_channels),
                                 pyramid_args,
                                 dict(layout=layout, kernel_size=1, channels=out_channels), inputs=inputs, **kwargs)
        else:
            self.pyramid = Block(pyramid_args, inputs=inputs, **kwargs)

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
        return attention + main

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
    min_features: int
        Minimum length of fused vector. Default is 32.
    """

    def __init__(self, channels, kernels=(3, 5), stride=1, padding='same',
                 use_dilation=False, groups=1, bias=False, ratio=4, min_features=32, inputs=None):
        super().__init__()

        if isinstance(channels, str):
            channels = safe_eval(channels, get_num_channels(inputs))

        num_kernels = len(kernels)
        num_dims = get_num_dims(inputs)

        if use_dilation:
            dilations = tuple((kernel - 3) // 2 + 1 for kernel in kernels)
            kernels = (3,) * num_kernels
        else:
            dilations = (1,) * num_kernels

        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_channels': get_num_channels(inputs),
            'out_channels': channels,
            'kernel_sizes': kernels,
            'dilations': dilations
        }

        self.split_layers = nn.ModuleList()
        tensors = []
        for kernel_size, dilation in zip(kernels, dilations):
            branch = Block(inputs=inputs, layout='cna', channels=channels,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=bias)
            self.split_layers.append(branch)
            tensors.append(branch(inputs))

        self.combine = Combine(op='sum')
        tensor = self.combine(tensors)
        self.fuse = Block(inputs=tensor, layout='Vfna>',
                          features=f'max(same // {ratio}, {min_features})',
                          dim=num_dims, bias=bias)

        fused_tensor = self.fuse(tensor)
        self.attention_branches = nn.ModuleList([
            Conv(inputs=fused_tensor, channels=channels, kernel_size=1, bias=bias) for i in range(num_kernels)])

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
        layer_desc = ('{class}({in_channels}, {out_channels}, '
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
        * Then we apply two 1x1 convolutions with groups=`cardinality`. The number of channels in the first
            convolution is `channels`*`radix` // `reduction_factor`.
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
        Factor increasing the number of channels after ResNeSt block. Thus, the number of output channels is
        `channels`*`scaling_factor`. Default 1.
    kwargs : dict
        Other named arguments only for the first :class:`~.layers.Block`.
    """
    def __init__(self, inputs, channels, layout='cna', radix=1, cardinality=1, reduction_factor=1, scaling_factor=1,
                 stride=1, padding='same', **kwargs):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels * self.radix

        self.inner_channels = self.channels // reduction_factor
        channel_dim = inputs.dim() - 2
        self.desc_kwargs = {
            'class': self.__class__.__name__,
            'in_channels': get_num_channels(inputs),
            'out_channels': channels*scaling_factor,
            'radix': self.radix,
            'cardinality': self.cardinality,
            'reduction_factor': reduction_factor,
            'scaling_factor': scaling_factor
        }

        self.inner_radix_conv = Block(inputs=inputs, layout=layout, channels=self.channels,
                                      groups=self.cardinality*self.radix, stride=stride, padding=padding,
                                      **kwargs)
        inputs = self.inner_radix_conv(inputs)
        x = inputs

        rchannel = inputs.shape[1]
        if self.radix > 1:
            splitted = torch.split(inputs, rchannel//self.radix, dim=1)
            inputs = sum(splitted)

        inner_conv1d_layout = 'V>' + 'cnac'
        self.avgpool_conv1d = Block(inputs=inputs, layout=inner_conv1d_layout,
                                    channels=[self.inner_channels, self.channels],
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
        layer_desc = ('{class}({in_channels}, {out_channels}, '
                      'radix={radix}, cardinality={cardinality}, '
                      'reduction_factor={reduction_factor}, '
                      'scaling_factor={scaling_factor})').format(**self.desc_kwargs)
        return layer_desc
