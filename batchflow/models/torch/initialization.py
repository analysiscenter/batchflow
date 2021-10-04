""" Functions for model weights initialization."""
import torch
from torch import nn

@torch.no_grad()
def common_used_weights_init(module):
    """ Common used non-default model weights initialization."""
    # Conv and linear layers
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)

    # Normalization layers
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)

    # Biases
    if getattr(module, 'bias', None) is not None:
        nn.init.constant_(module.bias, 0)
