""" Methods for model weights initialization."""
import torch
from torch import nn

@torch.no_grad()
def best_practice_resnet_init(module):
    """ Common used non-default model weights initialization for ResNet.

    It is similar to weights initialization in `PyTorch ResNet
    <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
