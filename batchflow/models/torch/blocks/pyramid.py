""" Functional modules for various deep network architectures."""
import numpy as np
from torch import nn

from .core import Block
from ..layers import Conv, Normalization, Combine
from ..utils import get_shape, get_num_dims, get_num_channels


class PyramidPooling(nn.Module):
    """ Pyramid Pooling module
    Zhao H. et al. "`Pyramid Scene Parsing Network <https://arxiv.org/abs/1612.01105>`_"

    Parameters
    ----------
    inputs : torch.Tensor
        Example of input tensor to this layer.
    layout : str
        Sequence of operations in convolution layer.
    channels : int
        Number of channels in pyramid branches.
    kernel_size : int
        Kernel size.
    pool_op : str
        Pooling operation ('mean' or 'max').
    pyramid : tuple of int
        Number of feature regions in each dimension.
        `0` is used to include `inputs` into the output tensor.
    """
    def __init__(self, inputs, layout='cna', channels=None, kernel_size=1, pool_op='mean',
                 pyramid=(0, 1, 2, 3, 6), **kwargs):
        super().__init__()

        spatial_shape = np.array(get_shape(inputs)[2:])
        channels = channels if channels else f'same // {len(pyramid)}'

        modules = nn.ModuleList()
        for level in pyramid:
            if level == 0:
                module = nn.Identity()
            else:
                x = inputs
                pool_size = tuple(np.ceil(spatial_shape / level).astype(np.int32))
                pool_stride = tuple(np.floor((spatial_shape - 1) / level + 1).astype(np.int32))

                module = Block(inputs=x, layout='p' + layout + 'b', channels=channels, kernel_size=kernel_size,
                               pool_op=pool_op, pool_size=pool_size, pool_stride=pool_stride,
                               factor=None, shape=tuple(spatial_shape), **kwargs)
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
    channels : int
        Number of channels in the output tensor.
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
    def __init__(self, inputs=None, layout='cna', channels='same', kernel_size=3,
                 rates=(6, 12, 18), pyramid=None, **kwargs):
        super().__init__()

        modules = nn.ModuleList()
        global_pooling = Block(inputs=inputs, layout='V>cnab', channels=channels,
                               kernel_size=1, dim=get_num_dims(inputs),
                               factor=None, shape=get_shape(inputs)[2:], **kwargs)
        modules.append(global_pooling)

        bottleneck = Block(inputs=inputs, layout=layout, channels=channels, kernel_size=1, **kwargs)
        modules.append(bottleneck)

        for level in rates:
            layer = Block(inputs=inputs, layout=layout, channels=channels, kernel_size=kernel_size,
                          padding=level, dilation=level, **kwargs)
            modules.append(layer)

        if pyramid is not None:
            pyramid = pyramid if isinstance(pyramid, (tuple, list)) else [pyramid]
            pyramid_layer = PyramidPooling(inputs=inputs, channels=channels, pyramid=pyramid, **kwargs)
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
        Layout for final postprocessing layer.
    channels : int
        Number of channels in the output tensor.
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

    def __init__(self, inputs=None, layout='cnad', channels=None, kernel_size=3,
                 rates=(6, 12, 18), pyramid=None, **kwargs):
        super().__init__()
        self.n = get_num_dims(inputs)
        self.conv = self.LAYERS[self.n]
        self.rates = rates

        out_channels = channels or get_num_channels(inputs)
        feature_channels = max(1, out_channels // len(rates))
        tensors = []

        # Bottleneck: 1x1 convolution
        self.bottleneck = Block(inputs=inputs, layout='cna', channels=feature_channels, kernel_size=1, **kwargs)
        tensors.append(self.bottleneck(inputs))

        # Convolutions with different dilations and shared weights
        self.layer = Conv(inputs=inputs, channels=feature_channels, kernel_size=kernel_size).to(inputs.device)
        tensor = self.layer(inputs)
        tensors.append(tensor)

        self.batch_norm = Normalization(inputs=tensor).to(inputs.device)

        for level in self.rates:
            tensor = self.conv(inputs, self.layer.layer.weight, padding=level, dilation=level)
            tensor = self.batch_norm(tensor)
            tensors.append(tensor)

        # Optional pyramid branch
        if pyramid is not None:
            pyramid = pyramid if isinstance(pyramid, (tuple, list)) else [pyramid]
            self.pyramid = PyramidPooling(inputs=inputs, channels=feature_channels, pyramid=pyramid, **kwargs)
            tensors.append(self.pyramid(inputs))
        else:
            self.pyramid = None

        # Global pooling
        self.global_pooling = Block(inputs=inputs, layout='V>cnab', channels=feature_channels, kernel_size=1,
                                    dim=self.n, shape=inputs.size()[2:], align_corners=True)
        tensors.append(self.global_pooling(inputs))

        # Concatenation of features
        self.combine = Combine(op='concat')
        combined = self.combine(tensors)

        # Final postprocessing
        self.post = Block(inputs=combined, layout=layout, channels=out_channels, kernel_size=kernel_size, **kwargs)

    def forward(self, x):
        # Bottleneck and base convolution layer
        tensors = [self.bottleneck(x), self.layer(x)]

        # Convolutions with different dilations and shared weights
        for level in self.rates:
            tensor = self.conv(x, self.layer.layer.weight, padding=level, dilation=level)
            tensor = self.batch_norm(tensor)
            tensor = nn.functional.relu(tensor)
            tensors.append(tensor)

        # Optional pyramid branch
        if self.pyramid:
            tensors.append(self.pyramid(x))

        # Global pooling
        global_info = self.global_pooling(x)
        tensors.append(global_info)

        # Concatenate features and apply final postprocessing
        combined = self.combine(tensors)
        return self.post(combined)
