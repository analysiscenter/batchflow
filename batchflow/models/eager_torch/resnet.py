"""
Kaiming He et al. "`Deep Residual Learning for Image Recognition
<https://arxiv.org/abs/1512.03385>`_"

Kaiming He et al. "`Identity Mappings in Deep Residual Networks
<https://arxiv.org/abs/1603.05027>`_"

Sergey Zagoruyko, Nikos Komodakis. "`Wide Residual Networks
<https://arxiv.org/abs/1605.07146>`_"

Xie S. et al. "`Aggregated Residual Transformations for Deep Neural Networks
<https://arxiv.org/abs/1611.05431>`_"

Jie Hu. et al. "`Squeeze-and-Excitation Networks
<https://arxiv.org/abs/1709.01507`_"
"""
#pylint: disable=too-many-ancestors
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
from .encoder_decoder import Encoder
from .utils import get_num_channels, safe_eval


CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']

class ResBlock(nn.Module):
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
        If str, then number of filters is calculated by its evaluation. `S` and `same` stand for the
        number of filters in the previous tensor. Note the `eval` usage under the hood.
        If int, then number of filters in the output tensor. Default value is 'same'.
        Lists are used for layouts that require multiple values, and elements are interpreted as described above.
    kernel_size : int, list of int
        Convolution kernel size. Default is 3.
    strides : int, list of int
        Convolution stride. Default is 1.
    downsample : int, bool
        If int, in first repetition of block downsampling with a factor `downsample`.
        If True, in first repetition of block downsampling with a factor 2.
        If False, without downsampling. Default is False.
    bottleneck : bool, int
        If True, then construct a canonical bottleneck block from the given layout.
        If False, then bottleneck is not used. Default is False.
    se : bool
        If True, then construct a SE-ResNet block from the given layout.
        If False, then squeeze and excitation is not used. Default is False.
    groups : int
        Use `groups` convolution side by side, each  seeing 1 / `groups` the input channels,
        and producing 1 / `groups` the output channels, and both subsequently concatenated.
        Number of `inputs` channels must be divisible by `groups`. Default is 1.
    op : str or callable
        Operation for combination shortcut and residual.
        See more :class:`~.layers.Combine` documentation. Default is '+'.
    n_reps : int
        Number of times to repeat the whole block. Default is 1.
    kwargs : dict
        Other named arguments for the :class:`~.layers.ConvBlock`
    """
    def __init__(self, inputs=None, layout='cnacna', filters='same', kernel_size=3, strides=1,
                 downsample=False, bottleneck=False, se=False, groups=1, op='+', n_reps=1, **kwargs):
        super().__init__()

        num_convs = sum(letter in CONV_LETTERS for letter in layout)

        filters = [filters] * num_convs if isinstance(filters, (int, str)) else filters
        filters = [safe_eval(str(item), get_num_channels(inputs)) for item in filters]

        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        strides = [strides] * num_convs if isinstance(strides, int) else strides
        strides_d = list(strides)
        groups = [groups] * num_convs
        side_branch_stride = np.prod(strides)
        side_branch_stride_d = int(side_branch_stride)

        if downsample:
            downsample = 2 if downsample is True else downsample
            strides_d[0] *= downsample
            side_branch_stride_d *= downsample
        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout + 'cna'
            kernel_size = [1] + kernel_size + [1]
            strides = [1] + strides + [1]
            strides_d = [1] + strides_d + [1]
            groups = [1] + groups + [1]
            filters = [filters[0]] + filters + [filters[0] * bottleneck]
        if se:
            layout += 'S*'
        layout = 'B' + layout + op

        layer_params = [{'strides': strides_d, 'side_branch/strides': side_branch_stride_d}] + [{}]*(n_reps-1)
        self.block = ConvBlock(*layer_params, inputs=inputs, layout=layout, filters=filters,
                               kernel_size=kernel_size, strides=strides, groups=groups,
                               side_branch={'layout': 'c', 'filters': filters[-1], 'strides': side_branch_stride},
                               **kwargs)

    def forward(self, x):
        return self.block(x)


class ResNet(Encoder):
    """ Base ResNet model.

    Parameters
    ----------
    body : dict, optional
        encoder : dict, optional
            num_stages : int
                Number of different layers. Default is 4.

            order : str, sequence of str
                Determines order of applying layers.
                See more in :class:`~.encoder_decoder.Encoder` documentation.
                In default ResNet, only 'block' is needed.

            blocks : dict, optional
                Parameters for pre-processing blocks. Each of the parameters can be represented
                either by a single value or by a list with `num_stages` length.
                If it is a `list`, then the i-th block is formed using the i-th value of the `list`.
                If this is a single value, then all the blocks is formed using it.

                base : callable, list of callable
                    Tensor processing function. Default is :class:`ResBlock`.
                layout : str, list of str
                    A sequence of letters, each letter meaning individual operation.
                    See more in :class:`~.layers.conv_block.BaseConvBlock` documentation.
                filters : int, str, list of int, list of str
                    If str, then number of filters is calculated by its evaluation. `S` and `same` stand for the
                    number of filters in the previous tensor. Note the `eval` usage under the hood.
                    If int, then number of filters in the block.
                n_reps : int, list of int
                    Number of times to repeat the whole block.
                downsample : bool, int, list of bool, list of int
                    If int, in first repetition of block downsampling with a factor `downsample`.
                    If True, in first repetition of block downsampling with a factor 2.
                    If False, without downsampling.
                bottleneck : bool, int, list of bool, list of int
                    If True, then construct a canonical bottleneck block from the given layout.
                    If False, then bottleneck is not used.
                se : bool, list of bool
                    If True, then construct a SE-ResNet block from the given layout.
                    If False, then squeeze and excitation is not used.
                other args : dict
                    Parameters for the base block.

    Notes
    -----
    This class is intended to define custom ResNets.
    For more convenience use predefined :class:`ResNet18`, :class:`ResNet34` and others described down below.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/order'] = ['block']
        config['body/encoder/blocks'] += dict(base=ResBlock, layout='cnacna',
                                              filters=[64, 128, 256, 512],
                                              n_reps=[1, 1, 1, 1],
                                              downsample=[False, True, True, True],
                                              bottleneck=False,
                                              se=False)

        config['head'] += dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'
        return config



class ResNet18(ResNet):
    """ The original ResNet-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [2, 2, 2, 2]
        return config

class ResNet34(ResNet):
    """ The original ResNet-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [3, 4, 6, 3]
        return config

class ResNet50(ResNet34):
    """ The original ResNet-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/layout'] = 'cna'
        config['body/encoder/blocks/bottleneck'] = True
        return config

class ResNet101(ResNet50):
    """ The original ResNet-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [3, 4, 23, 3]
        return config

class ResNet152(ResNet50):
    """ The original ResNet-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [3, 8, 36, 3]
        return config



class ResNeXt18(ResNet18):
    """ The ResNeXt-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/groups'] = 32
        return config

class ResNeXt34(ResNet34):
    """ The ResNeXt-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/groups'] = 32
        return config

class ResNeXt50(ResNet50):
    """ The ResNeXt-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/groups'] = 32
        return config

class ResNeXt101(ResNet101):
    """ The ResNeXt-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/groups'] = 32
        return config

class ResNeXt152(ResNet152):
    """ The ResNeXt-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/groups'] = 32
        return config



class SEResNet18(ResNet18):
    """ The original SE-ResNet-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNet34(ResNet34):
    """ The original SE-ResNet-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNet50(ResNet50):
    """ The original SE-ResNet-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNet101(ResNet101):
    """ The original SE-ResNet-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNet152(ResNet152):
    """ The original SE-ResNet-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config



class SEResNeXt18(ResNeXt18):
    """ The SE-ResNeXt-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNeXt34(ResNeXt34):
    """ The SE-ResNeXt-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNeXt50(ResNeXt50):
    """ The SE-ResNeXt-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNeXt101(ResNeXt101):
    """ The SE-ResNeXt-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config

class SEResNeXt152(ResNeXt152):
    """ The SE-ResNeXt-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/se'] = True
        config['body/encoder/blocks/ratio'] = 16
        return config
