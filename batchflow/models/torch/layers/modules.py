""" Functional modules for various deep network architectures."""
import numpy as np
import torch.nn as nn

from .resize import Upsample, Combine
from .conv_block import ConvBlock
from .conv import Conv
from ..utils import get_shape, get_num_dims


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
                 rates=(6, 12, 18), pyramid=None, **kwargs):
        super().__init__()

        modules = nn.ModuleList()
        global_pooling = ConvBlock(inputs=inputs, layout='V>cna', filters=filters, kernel_size=1,
                                   dim=get_num_dims(inputs))
        modules.append(global_pooling)

        bottleneck = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=1, **kwargs)
        modules.append(bottleneck)

        for level in rates:
            layer = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=kernel_size,
                              padding=level, dilation_rate=level, **kwargs)
            modules.append(layer)

        if pyramid is not None:
            pyramid = pyramid if isinstance(pyramid, (tuple, list)) else [pyramid]
            pyramid_layer = PyramidPooling(inputs=inputs, filters=filters, pyramid=pyramid, **kwargs)
            modules.append(pyramid_layer)


        self.blocks = modules
        self.combine = Combine(op='concat')

    def forward(self, x):
        levels = [layer(x) for layer in self.blocks]
        return self.combine(levels)



class KSAC(nn.Module):
    """ Kernel sharing atrous convolution.

    Huang Y. et al. "`See More Than Once -- Kernel-Sharing Atrous Convolution for Semantic Segmentation
    <https://arxiv.org/abs/1908.09443>`_"

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
    """
    LAYERS = {
        1: nn.functional.conv1d,
        2: nn.functional.conv2d,
        3: nn.functional.conv3d,
    }

    def __init__(self, inputs=None, layout='cna', filters=None, kernel_size=3,
                 rates=(6, 12, 18), pyramid=None, **kwargs):
        super().__init__()
        filters = filters if filters else f'max(1, same // {len(rates)})'
        self.bottleneck = ConvBlock(inputs=inputs, layout=layout, filters=filters, kernel_size=1, **kwargs)

        self.n = get_num_dims(inputs)
        self.global_pooling = ConvBlock(inputs=inputs, layout='V>cna', filters=filters, kernel_size=1,
                                        dim=self.n)

        self.layer = Conv(inputs=inputs, filters=filters, kernel_size=kernel_size).to(inputs.device)
        _ = self.layer(inputs)
        self.conv = self.LAYERS[self.n]
        self.rates = rates

        if pyramid is not None:
            pyramid = pyramid if isinstance(pyramid, (tuple, list)) else [pyramid]
            layer = PyramidPooling(inputs=inputs, filters=filters, pyramid=pyramid, **kwargs)
            self.pyramid = layer
        else:
            self.pyramid = None

        self.combine = Combine(op='concat')

    def forward(self, x):
        levels = [self.bottleneck(x), self.layer(x)]

        for level in self.rates:
            tensor = self.conv(x, self.layer.layer.weight, padding=level, dilation=level)
            levels.append(tensor)

        global_info = nn.functional.interpolate(self.global_pooling(x), size=tensor.size()[2:],
                                                mode='bilinear', align_corners=True)
        levels.append(global_info)

        if self.pyramid:
            levels.append(self.pyramid(x))
        return self.combine(levels)
