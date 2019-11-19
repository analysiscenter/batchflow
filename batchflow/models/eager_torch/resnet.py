"""."""
from copy import deepcopy
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
from .base import EagerTorch

CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T']

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, layout='cnacna', filters=64, downsample=False, bottleneck=False, groups=1):
        super().__init__()

        num_convs = sum([letter in CONV_LETTERS for letter in layout])

        self.layout = layout
        self.filters = [filter] * num_convs if isinstance(filters, int) else filters
        self.strides = [1] * num_convs
        self.kernel_size = [3] * num_convs
        self.groups = groups * num_convs

        if downsample:
            self.strides = [2] + self.strides[1:]
        if bottleneck:
            self.layout = 'cna' + self.layout + 'cna'
            self.kernel_size = [1] + self.kernel_size + [1]
            self.strides = [1] + self.strides + [1]
            if boottleneck is True:
                boottleneck = 4
            self.filters = [self.filters[0]] + self.filters + [self.filters[0] * bottleneck]
            self.groups = [1, self.groups, 1]


        self.main_branch = ConvBlock(layout=self.layout, filters=self.filters,
                                     kernel_size=self.kernel_size, strides=self.strides, groups=self.groups)
        self.side_branch = ConvBlock(layout='c', filters=self.filters[-1],
                                     kernel_size=1, strides=2)

    def forward(self, x):
        """ Make a forward pass """
        return self.main_branch(x) + self.side_branch(x)


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

        # config['head'] += dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'

        return config


    # def build_config(self):
    #     config = super().build_config()
    #     return config
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


        # if config.get('head/units') is None:
        #     config['head/units'] = self.num_classes('targets')
        # if config.get('head/filters') is None:
        #     config['head/filters'] = self.num_classes('targets')
        return config

    @classmethod
    def body(cls, inputs=None, **kwargs):

        kwargs = cls.get_defaults('body', kwargs)
        num_blocks = kwargs.get('num_blocks')
        filters = kwargs.get('filters')
        downsample_layer = kwargs.get('downsample')
        # list_layers = []

        resblock_args = deepcopy(kwargs['block'])
        # x = inputs
        for num_resblocks, filter, downsample in zip(num_blocks, filters, downsample_layer):
            for idx_resblock in range(num_resblocks):
                downsample = downsample if idx_resblock == 0 else False
                resblock_args.update({'filters': filter, 'downsample': downsample})
                print(resblock_args)

                # x = ResBlock(**resblock_args)

                downsample = False
        # kwargs = cls.get_defaults('body', kwargs)
        # filters, block_args = cls.pop(['filters', 'block'], kwargs)
        # block_args = {**kwargs, **block_args}
