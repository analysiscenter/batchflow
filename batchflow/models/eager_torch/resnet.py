"""."""
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
# from .base import EagerTorch
from .encoder_decoder import Encoder

CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, inputs=None, layout='cnacna', filters=64, kernel_size=3,
                 strides=1, downsample=False, bottleneck=False, groups=1, op='+', n_reps=1, **kwargs):
        super().__init__()

        num_convs = sum([letter in CONV_LETTERS for letter in layout])

        filters = [filters] * num_convs if isinstance(filters, int) else filters
        strides = [strides] * num_convs if isinstance(strides, int) else strides
        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        groups = [groups] * num_convs
        strides_d = list(strides)
        side_branch_stride_d = 1

        if downsample:
            downsample = 2 if downsample is True else downsample
            strides_d[0] = downsample
            side_branch_stride_d = downsample
        if bottleneck:
            layout = 'cna' + layout + 'cna'
            kernel_size = [1] + kernel_size + [1]
            strides = [1] + strides + [1]
            strides_d = [1] + strides_d + [1]
            groups = [1] + groups + [1]
            bottleneck = 4 if bottleneck is True else bottleneck
            filters = [filters[0]] + filters + [filters[0] * bottleneck]
        layout = 'B' + layout + op
        subblock_configs = [{'strides': strides_d, 'side_branch': {'strides': side_branch_stride_d}}] + [{}]*(n_reps-1)
        self.block = ConvBlock(*subblock_configs, inputs=inputs, layout=layout, filters=filters,
                               kernel_size=kernel_size, strides=strides, groups=groups,
                               side_branch={'layout': 'c', 'filters': filters[-1], 'strides': 1}, **kwargs)

    def forward(self, inputs):
        """ Make a forward pass """
        return self.block(inputs)


class ResNet(Encoder):
    """."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        # config['common/conv/bias'] = False
        # config['common/data_format'] = 'channels_first'
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/encoder/num_stages'] = 4
        config['body/encoder/downsample'] += dict(layout=None, order=['block'])
        config['body/encoder/blocks'] = dict(base=ResBlock, layout='cnacna', filters=None,
                                             downsample=[False, True, True, True],
                                             bottleneck=False)
        # config['body/width_factor'] = 1

        config['head'] += dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'

        return config

    def build_config(self):
        config = super().build_config()

        if config.get('body/filters') is None:
            # width = config.pop('body/width_factor')
            num_blocks = config['body/encoder/blocks/n_reps']
            filters = config['initial_block/filters']
            config['body/encoder/blocks/filters'] = (2 ** np.arange(len(num_blocks)) * filters).tolist()

        # if config.get('body/downsample') is None:
        #     num_blocks = config['body/num_blocks']
        #     config['body/downsample'] = [False] + [True] * (len(num_blocks) - 1)


        if config.get('head/units') is None:
            config['head/units'] = config['inputs/labels/classes']
        if config.get('head/filters') is None:
            config['head/filters'] = config['inputs/labels/classes']
        return config

    # @classmethod
    # def body(cls, inputs=None, **kwargs):
    #     kwargs = cls.get_defaults('body', kwargs)
    #     num_blocks = kwargs.get('num_blocks')
    #     filters = kwargs.get('filters')
    #     downsample_block = kwargs.get('downsample')

    #     block_list = []
    #     for num_resblocks, filter, downsample in zip(num_blocks, filters, downsample_block):

    #         resblock_args = {**kwargs['block'], 'inputs': inputs, 'filters': filter,
    #                          'downsample': downsample, 'n_reps': num_resblocks}
    #         block = ResBlock(**resblock_args)
    #         block_list.append(block)
    #         inputs = block(inputs)
    #     return nn.Sequential(*block_list)

class ResNet18(ResNet):
    """ The original ResNet-18 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/blocks/n_reps'] = [2, 2, 2, 2]
        return config


class ResNet34(ResNet):
    """ The original ResNet-34 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/num_blocks'] = [3, 4, 6, 3]
        return config


class ResNet50(ResNet):
    """ The original ResNet-50 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/num_blocks'] = [3, 4, 6, 3]
        config['body/block/layout'] = 'cna'
        config['body/block/bottleneck'] = True
        return config


class ResNet101(ResNet):
    """ The original ResNet-101 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/num_blocks'] = [3, 4, 23, 3]
        config['body/block/layout'] = 'cna'
        config['body/block/bottleneck'] = True
        return config


class ResNet152(ResNet):
    """ The original ResNet-152 architecture."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/num_blocks'] = [3, 8, 36, 3]
        config['body/block/layout'] = 'cna'
        config['body/block/bottleneck'] = True
        return config
