""" Functional modules for various deep network architectures."""

from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from .layers import ConvBlock, Upsample, Combine, BaseConvBlock
from .utils import get_shape


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


class SelfAttention(nn.Module):
    """ Improved self Attention module.

    Wang Z. et al. "'Less Memory, Faster Speed: Refining Self-Attention Module for Image
    Reconstruction <https://arxiv.org/abs/1905.08008>'_"

    Parameters
    ----------
    reduction_ratio : float
        The reduction ratio of filters in the inner convolutions.
    """
    def __init__(self, inputs=None, layout='c', kernel_size=1, reduction_ratio=8, strides=1, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, device=inputs.device))

        args = {**kwargs, **dict(inputs=inputs, layout=layout, kernel_size=kernel_size, strides=strides)}
        self.conv1 = ConvBlock(**args, filters='same//{}'.format(reduction_ratio),)
        self.conv2 = ConvBlock(**args, filters='same')
        self.conv3 = ConvBlock(**args, filters='same//{}'.format(reduction_ratio))

    def forward(self, x):
        bs, spatial = x.shape[0], x.shape[2:]
        N = np.prod(spatial)

        phi = self.conv1(x).view(bs, -1, N) # (B, C/8, N)
        theta = self.conv2(x).view(bs, N, -1) # (B, N, C)
        attention = torch.bmm(phi, theta) / N # (B, C/8, C)

        out = self.conv3(x).view(bs, N, -1) # (B, N, C/8)
        out = torch.bmm(out, attention).view(bs, -1, *spatial)
        return self.gamma*out + x


class FeaturePyramidAttention(nn.Module):
    """ https://arxiv.org/abs/1805.10180 """
    def __init__(self, inputs=None, pyramid_kernel_size=[7, 5, 3], filters='same', layout='cna', 
                 downsample_layout='p', upsample_layout='t', **kwargs):
        super().__init__()
        depth = len(pyramid_kernel_size)
        inputs_spatial_shape = get_shape(inputs)[2:]
        self.attention = ConvBlock(inputs=inputs, layout='V >>' + layout + upsample_layout, 
                                       filters=filters, kernel_size=[1, inputs_spatial_shape], **kwargs)

        enc_layout = ('B' + downsample_layout + layout) * depth
        emb_layout = layout + upsample_layout
        combine_layout = '+' * (depth - 1) + '*'
        main_layout = enc_layout + emb_layout + combine_layout
        main_kernel_size = pyramid_kernel_size + [pyramid_kernel_size[-1]] + [2]
        main_strides = [1] * (depth + 1) + [2]

        branches = [dict(layout=layout, filters=filters, kernel_size=1)]
        for kernel_size in pyramid_kernel_size[:-1]:
            args = dict(layout = layout + upsample_layout, kernel_size=[kernel_size, 2], strides=[1, 2], filters=filters)
            branches.append(args)

        self.pyramid = ConvBlock(layout=main_layout, inputs=inputs, filters=filters, kernel_size=main_kernel_size,
                                 strides=main_strides, branch=branches, **kwargs)
        self.combine = Combine(op='+')

    def forward(self, x):
        attention = self.attention(x)
        main = self.pyramid(x)
        return self.combine([attention, main])

class FPA(nn.Module):
    def __init__(self, inputs=None, pyramid_kernel_size=[7, 5, 3], filters='same', layout='cna', 
                 downsample_layout='p', upsample_layout='t', **kwargs):
        super().__init__()
        inputs_spatial_shape = get_shape(inputs)[2:]
        self.attention_branch = ConvBlock(inputs=inputs, layout='V >>' + layout + upsample_layout, filters=filters, 
                                              kernel_size=[1, inputs_spatial_shape], **kwargs)

        self.mid_branch = ConvBlock(inputs=inputs, layout=layout, kernel_size=1, filters=filters, **kwargs)

        x = inputs
        skips, upsamples, depth  = torch.nn.ModuleList(), torch.nn.ModuleList(), torch.nn.ModuleList()
        for kernel_size in pyramid_kernel_size:
            args = dict(layout=downsample_layout + layout, kernel_size=kernel_size, filters=filters, **kwargs)
            depth_layer = ConvBlock(args, inputs=x)
            depth.append(depth_layer)

            x = depth_layer(x)

            skip_layer = ConvBlock({**args, 'layout': layout}, inputs=x)
            skips.append(skip_layer)

            skip_out = skip_layer(x)

            upsample_layer = Upsample(layout=upsample_layout, filters=filters, inputs=skip_out)
            upsamples.append(upsample_layer)
        
        self.depth_layers = depth
        self.skip_layers = skips
        self.upsample_layers = upsamples
        self.combine_pyramid = [Combine(op='*')] + [Combine(op='+')] * (len(pyramid_kernel_size) - 1)
        self.combine_attention = Combine(op='+')

    def forward(self, x):
        mid = self.mid_branch(x)
        skips = [mid]
        for depth_layer, skip_layer in zip(self.depth_layers, self.skip_layers):
            x = depth_layer(x)
            skip = skip_layer(x)
            skips.append(skip)

        pyramid_out = skips[-1]
        for i, (upsample, combine) in enumerate(self.upsample_layers[::-1], self.combine_pyramid[::-1]):
            i += 1
            pyramid_out = upsample(pyramid_out)
            pyramid_out = combine([skip[-i -1], pyramid_out])

        attention = self.attention_branch(x)
        return self.combine_attention([pyramid_out, attention])
