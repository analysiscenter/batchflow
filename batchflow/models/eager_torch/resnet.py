"""."""
import numpy as np
import torch.nn as nn

from .layers import ConvBlock
from .base import EagerTorch

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, inputs, layout, filter, downsample, bottleneck, resnext):
        super().__init__()

        conv_letters = ['c', 'C', 'w', 'W', 't', 'T']
        num_convs = sum([letter in conv_letters for letter in layout])

        self.layout = layout
        self.kernel_size = [3] * num_convs
        self.filters = [filter] * num_convs
        self.strides = [1] * num_convs

        if downsample:
            self.strides = [2] + self.strides[1:]
        if bottleneck:
            self.kernel_size = [1] + self.kernel_size[1:-1] + [1]
            self.filters = self.filters[:-1] + [filter * 4]
        if resnext:
            self.groups = resnext
        else:
            self.groups = 1


    def forward(self, x):
        """ Make a forward pass """
        x = self.layer = ConvBlock(inputs=x, layout=self.layout, filters=self.filters,
                               kernel_size=self.kernel_size, strides=self.strides, groups=self.groups)
        return x


class ResNet(EagerTorch):
    """."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False
        config['common/data_format'] = 'channels_first'
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/block'] = dict(layout='R cna cna +', bottleneck=False, resnext=False)
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
        downsample_body = kwargs.get('downsample')
        list_layers = []

        block_params = kwargs['block']
        x = inputs
        print(kwargs)
        for num_block, filter, downsample_block in zip(num_blocks, filters, downsample_body):
            downsample_block = [downsample_block] + [False] * (num_block - 1)
            for block, downsample in zip(range(num_block), downsample_block):
                # block_params['downsample'] = downsample
                # block_params['filter'] = filter
                # print(downsample, filter, dict(block_params))
                x = ResBlock(inputs=x, downsample=downsample, filter=filter, **block_params)
                print(x)
                # layout, filter, downsample, bottleneck, resnext):
                downsample = False
        # kwargs = cls.get_defaults('body', kwargs)
        # filters, block_args = cls.pop(['filters', 'block'], kwargs)
        # block_args = {**kwargs, **block_args}
