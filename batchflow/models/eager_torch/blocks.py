import numpy as np
import torch.nn as nn

from .layers import ConvBlock, Combine
from .utils import get_num_channels
from .encoder_decoder import EncoderDecoder

class DenseBlock(nn.Module):
    """ DenseBlock module. """
    def __init__(self, inputs=None, layout='nacd', filters=None, kernel_size=3, strides=1, dropout_rate=0.2,
                 num_layers=5, growth_rate=12, skip=True, bottleneck=False, **kwargs):
        super().__init__()
        layout = 'R' + layout + '.'
        self.skip = skip
        self.input_num_channels = get_num_channels(inputs)

        if filters is not None:
            growth_rate = (filters - self.input_num_channels) // num_layers 
        filters = growth_rate

        if bottleneck:
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout
            kernel_size = [1, kernel_size]
            strides = [1, strides]
            filters = [growth_rate * bottleneck, filters] 
        
        self.block = ConvBlock(dict(layout=layout, kernel_size=kernel_size, strides=strides, dropout_rate=dropout_rate, 
                                    filters=filters), n_repeats=num_layers, inputs=inputs)
        self.combine = Combine(op='concat')

    def forward(self, x):
        output = self.block(x)
        return output if self.skip else output[:, self.input_num_channels:]
