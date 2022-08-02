import numpy as np

from .core import Block
from ..utils import get_num_channels, safe_eval


CONV_LETTERS = ['c', 'C', 'w', 'W', 't', 'T', 'k', 'K', 'q', 'Q']

class ResBlock(Block):
    def __init__(self, inputs=None, layout='cnacn', channels='same', kernel_size=3, stride=1,
                 downsample=False, bottleneck=False, attention=None, groups=1, op='+a', branch=None,
                 n_reps=1, **kwargs):
        width_mid_channels = 2 if groups > 1 else 1
        num_convs = sum(letter in CONV_LETTERS for letter in layout)

        channels = [channels] * num_convs if isinstance(channels, (int, str)) else channels
        channels = [safe_eval(item, get_num_channels(inputs)) if isinstance(item, str) else item
                   for item in channels]

        kernel_size = [kernel_size] * num_convs if isinstance(kernel_size, int) else kernel_size
        stride = [stride] * num_convs if isinstance(stride, int) else stride
        groups = [groups] * num_convs
        if branch is None:
            branch = {}
        branch_stride = branch.get('stride', np.prod(stride))
        branch_layout = branch.get('layout', 'cn')

        # Used in the first repetition of the block.
        # Different from stride and branch_stride in other blocks if `downsample` is not ``False``.
        stride_downsample = list(stride)
        branch_stride_downsample = int(branch_stride)

        # Parse all the parameters
        if downsample:
            # The first repetition of the block optionally downsamples inputs
            downsample = 2 if downsample is True else downsample
            stride_downsample[0] *= downsample
            branch_stride_downsample *= downsample

        if bottleneck:
            # Bottleneck: apply 1x1 conv before and after main flow computations to change number of channels
            bottleneck = 4 if bottleneck is True else bottleneck
            layout = 'cna' + layout + 'acn'
            kernel_size = [1] + kernel_size + [1]
            stride = [1] + stride + [1]
            stride_downsample = [1] + stride_downsample + [1]
            groups = [1] + groups + [1]
            channels = [channels[0] * width_mid_channels] + [channels[i] * width_mid_channels
                                                             for i in range(len(channels))] + [channels[0] * bottleneck]

        if attention:
            # Attention: add self-attention to the main flow
            layout += 'S'

        if get_num_channels(inputs) != channels[-1] or np.prod(stride_downsample) != 1:
            # If main flow changes the number of channels, so must do the side branch.
            # No activation, because it will be applied after summation with the main flow
            branch_params = {'layout': branch_layout, 'channels': channels[-1],
                             'kernel_size': 1, 'stride': branch_stride_downsample}
        else:
            branch_params = {}
        layout = 'R' + layout + op

        # Pass optional downsample parameters both to the main flow and to the side branch:
        # Only the first repetition is to be changed
        layer_params = [{'stride': stride_downsample,
                         'branch': branch_params,
                         'branch/stride': branch_stride_downsample}]
        layer_params += [{}]*(n_reps-1)

        super().__init__(*layer_params, inputs=inputs, layout=layout, channels=channels,
                         kernel_size=kernel_size, stride=stride, groups=groups, attention=attention,
                         **kwargs)
