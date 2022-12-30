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
<https://arxiv.org/abs/1709.01507>`_"
"""
#pylint: disable=too-many-ancestors
from .base import TorchModel
from .blocks import ResBlock, BottleneckBlock



class ResNet(TorchModel):
    """ Base ResNet model.

    Parameters
    ----------
    body : dict, optional
        num_stages : int
            Number of different layers. Default is 4.

        order : str, sequence of str
            Determines order of applying layers.
            See more in :class:`~.encoder_decoder.Encoder` documentation.
            In default ResNet, only ``'block'`` is needed.

        blocks : dict, optional
            Parameters for pre-processing blocks. Each of the parameters can be represented
            either by a single value or by a list with `num_stages` length.
            If it is a `list`, then the i-th block is formed using the i-th value of the `list`.
            If this is a single value, then all the blocks is formed using it.

            base : callable, list of callable
                Tensor processing function. Default is :class:`~ResBlock`.
            layout : str, list of str
                A sequence of letters, each letter meaning individual operation.
                See more in :class:`~.layers.conv_block.BaseConvBlock` documentation.
            channels : int, str, list of int, list of str
                If `str`, then number of channels is calculated by its evaluation. ``'S'`` and ``'same'``
                stand for the number of channels in the previous tensor. Note the `eval` usage under the hood.
                If `int`, then number of channels in the block.
            n_reps : int, list of int
                Number of times to repeat the whole block.
            downsample : bool, int, list of bool, list of int
                If `int`, in first repetition of block downsampling with a factor ``downsample``.
                If ``True``, in first repetition of block downsampling with a factor 2.
                If ``False``, without downsampling.
            se : bool, list of bool
                If ``True``, then construct a SE-ResNet block from the given layout.
                If ``False``, then squeeze and excitation is not used.
            other args : dict
                Parameters for the base block.

    Notes
    -----
    This class is intended to define custom ResNets.
    For more convenience use predefined :class:`ResNet18`, :class:`ResNet34` and others described down below.
    """
    @classmethod
    def default_config(cls):
        """ Define model's defaults: general architecture. """
        config = super().default_config()

        config.update({
            'initial_block': {
                'layout': 'cnap',
                'channels': 64,
                'kernel_size': 7,
                'stride': 2,
                'pool_size': 3,
                'pool_stride': 2
            },
            'body': {
                'type': 'encoder',
                'output_type': 'tensor',
                'num_stages': 4,
                'order': ['block'],
                'blocks': {
                    'base_block': ResBlock,
                    'layout': 'cnacn',
                    'channels': [64, 128, 256, 512],
                    'n_reps': [1, 1, 1, 1],
                    'downsample': [False, True, True, True],
                    'attention': False,
                }
            },
            'head': {
                'layout': 'Vdf',
                'dropout_rate' : 0.4,
            },

            'loss': 'ce',
        })
        return config



class ResNet18(ResNet):
    """ The original ResNet-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/n_reps'] = [2, 2, 2, 2]
        return config

class ResNet34(ResNet):
    """ The original ResNet-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/n_reps'] = [3, 4, 6, 3]
        return config

class ResNet50(ResNet34):
    """ The original ResNet-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/base_block'] = BottleneckBlock
        config['body/blocks/layout'] = 'cna cna cn'
        return config

class ResNet101(ResNet50):
    """ The original ResNet-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/n_reps'] = [3, 4, 23, 3]
        return config

class ResNet152(ResNet50):
    """ The original ResNet-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/n_reps'] = [3, 8, 36, 3]
        return config



class ResNeXt18(ResNet18):
    """ The ResNeXt-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/groups'] = 32
        return config

class ResNeXt34(ResNet34):
    """ The ResNeXt-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/groups'] = 32
        return config

class ResNeXt50(ResNet50):
    """ The ResNeXt-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/groups'] = 32
        return config

class ResNeXt101(ResNet101):
    """ The ResNeXt-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/groups'] = 32
        return config

class ResNeXt152(ResNet152):
    """ The ResNeXt-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/groups'] = 32
        return config



class SEResNet18(ResNet18):
    """ The original SE-ResNet-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNet34(ResNet34):
    """ The original SE-ResNet-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNet50(ResNet50):
    """ The original SE-ResNet-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNet101(ResNet101):
    """ The original SE-ResNet-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNet152(ResNet152):
    """ The original SE-ResNet-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config



class SEResNeXt18(ResNeXt18):
    """ The SE-ResNeXt-18 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNeXt34(ResNeXt34):
    """ The SE-ResNeXt-34 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNeXt50(ResNeXt50):
    """ The SE-ResNeXt-50 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNeXt101(ResNeXt101):
    """ The SE-ResNeXt-101 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config

class SEResNeXt152(ResNeXt152):
    """ The SE-ResNeXt-152 architecture. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/blocks/attention'] = 'se'
        config['body/blocks/self_attention/ratio'] = 16
        return config
