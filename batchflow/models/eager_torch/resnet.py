"""."""
import numpy as np
import torch.nn as nn

from .base import EagerTorch

class ResBlock(nn.Module):
    """ Add modules """
    def __init__(self, conv, shortcut, padding=None):
        super().__init__()
        self.conv = conv
        self.shortcut = shortcut
        self.padding = padding
        # self.output_shape = get_shape(conv)

    def forward(self, x):
        """ Make a forward pass """
        shortcut = self.shortcut(x) if self.shortcut else x
        # if self.padding:
        #     shortcut = F.pad(shortcut, self.padding)
        return self.conv(x) + shortcut


class ResNet(EagerTorch):
    """."""
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['common/conv/bias'] = False
        config['common/data_format'] = 'channels_first'
        config['initial_block'] += dict(layout='cnap', filters=64, kernel_size=7, strides=2,
                                        pool_size=3, pool_strides=2)

        config['body/block'] = dict(layout=None, post_activation=None, downsample=False,
                                    bottleneck=None, width_factor=1, zero_pad=False,
                                    resnext=False)

        config['head'] += dict(layout='Vdf', dropout_rate=.4)

        config['loss'] = 'ce'

        return config

    @classmethod
    def default_layout(cls, bottleneck, **kwargs):
        """ Define conv block layout """
        _ = kwargs
        reps = 3 if bottleneck else 2
        return 'cna' * reps

    # def build_config(self):
    #     config = super().build_config()
    #     return config
    def build_config(self):
        config = super().build_config()

        if config.get('body/filters') is None:
            width = config['body/block/width_factor']
            num_blocks = config['body/num_blocks']
            filters = config['initial_block/filters']
            config['body/filters'] = (2 ** np.arange(len(num_blocks)) * filters * width).tolist()

        if config.get('body/downsample') is None:
            num_blocks = config['body/num_blocks']
            config['body/downsample'] = [[]] + [[0]] * (len(num_blocks) - 1)

        # if config.get('head/units') is None:
        #     config['head/units'] = self.num_classes('targets')
        # if config.get('head/filters') is None:
        #     config['head/filters'] = self.num_classes('targets')

        return config
