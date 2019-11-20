"""."""
import numpy as np
import torch.nn as nn

from .layers import ConvBlock, Combine
from .base import EagerTorch

CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, inputs=None, layout='cnacna', filters=64, kernel_size=3,
                 strides=1, downsample=False, bottleneck=False, groups=1, op='+'):
        super().__init__()

        num_convs = sum([letter in CONV_LETTERS for letter in layout])

        self.layout = layout
        self.filters = [filters] * num_convs if isinstance(filters, int) else filters
        self.strides = [strides] * num_convs if isinstance(strides, int) else strides
        self.kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        self.groups = [groups] * num_convs
        self.op = op
        self.side_branch_stride = 1

        if downsample:
            downsample = 2 if downsample is True else downsample
            self.strides[0] = downsample
            self.side_branch_stride = downsample
        if bottleneck:
            self.layout = 'cna' + self.layout + 'cna'
            self.kernel_size = [1] + self.kernel_size + [1]
            self.strides = [1] + self.strides + [1]
            self.groups = [1] + self.groups + [1]
            bottleneck = 4 if bottleneck is True else bottleneck
            self.filters = [self.filters[0]] + self.filters + [self.filters[0] * bottleneck]

        self.main_branch = ConvBlock(inputs=inputs, layout=self.layout, filters=self.filters,
                                     kernel_size=self.kernel_size, strides=self.strides, groups=self.groups)
        self.side_branch = ConvBlock(inputs=inputs, layout='c', filters=self.filters[-1],
                                     kernel_size=1, strides=self.side_branch_stride)
        layer = self.main_branch(inputs)
        shortcut = self.main_branch(inputs)
        self.combine = Combine(inputs=[layer, shortcut], op=self.op)

    def forward(self, inputs):
        """ Make a forward pass """
        layer = self.main_branch(inputs)
        shortcut = self.side_branch(inputs)
        return self.combine([layer, shortcut])


class ResNet(EagerTorch):
    """."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False
        config['common/data_format'] = 'channels_first'
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/block'] = dict(layout='cna cna', filters=None,
                                    downsample=None, bottleneck=False, groups=1)
        config['body/width_factor'] = 1

        config['head'] += dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'

        return config

    def build_config(self):
        config = super().build_config()

        if config.get('body/filters') is None:
            width = config.get('body/width_factor')
            num_blocks = config['body/num_blocks']
            filters = config['initial_block/filters']
            config['body/filters'] = (2 ** np.arange(len(num_blocks)) * filters * width).tolist()

        if config.get('body/downsample') is None:
            num_blocks = config['body/num_blocks']
            config['body/downsample'] = [False] + [True] * (len(num_blocks) - 1)


        if config.get('head/units') is None:
            config['head/units'] = config['inputs/labels/classes']
        if config.get('head/filters') is None:
            config['head/filters'] = config['inputs/labels/classes']
        return config

    @classmethod
    def body(cls, inputs=None, **kwargs):
        kwargs = cls.get_defaults('body', kwargs)
        num_blocks = kwargs.get('num_blocks')
        filters = kwargs.get('filters')
        downsample_layer = kwargs.get('downsample')

        x = inputs
        block_list = []
        for num_resblocks, filter, downsample in zip(num_blocks, filters, downsample_layer):
            for idx_resblock in range(num_resblocks):
                downsample = downsample if idx_resblock == 0 else False
                resblock_args = {**kwargs['block'], 'inputs': x, 'filters': filter, 'downsample': downsample}

                layer = ResBlock(**resblock_args)
                block_list.append(layer)
                x = layer(x)
        return nn.Sequential(*block_list)
