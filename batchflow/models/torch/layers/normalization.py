""" Normalization layers. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_num_channels, get_num_dims



class LayerNorm(nn.Module):
    """ Layer normalization layer. Works with both `channels_first` and `channels_last` format. """
    def __init__(self, num_features=None, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        self.num_features = num_features

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.num_features,), self.weight, self.bias, self.eps)

        # Use multiplication instead of `pow` as `torch2trt` throws warnings about it
        mean = x.mean(1, keepdim=True)
        normalized = x - mean
        std = (normalized * normalized).mean(1, keepdim=True)
        std = torch.sqrt(std + self.eps)
        x = normalized / std
        x = self.weight[None, :, None, None] * x + self.bias[None, :, None, None]
        return x

    def extra_repr(self):
        return f'data_format={self.data_format}'


class Normalization(nn.Module):
    """ Multi-dimensional normalization layer.
    Allows to select `normalization_type` to choose between batch, instance and layer normalizations.
    """
    LAYERS = {
        'batch': {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        },
        'instance': {
            1: nn.InstanceNorm1d,
            2: nn.InstanceNorm2d,
            3: nn.InstanceNorm3d,
        },
        'layer': {
            1: LayerNorm,
            2: LayerNorm,
            3: LayerNorm,
        }
    }

    def __init__(self, inputs=None, normalization_type='batch', **kwargs):
        super().__init__()
        ndims = get_num_dims(inputs)
        num_features = get_num_channels(inputs)

        if normalization_type in [None, False, 'none']:
            self.layer = None
        elif normalization_type in self.LAYERS:
            self.layer = self.LAYERS[normalization_type][ndims](num_features, **kwargs)
        else:
            raise KeyError(f'Unknown type of normalization `{normalization_type}`. '
                            f'Use one of {list(self.LAYERS.keys())}!')

    def forward(self, x):
        if self.layer is not None:
            x = self.layer(x)
        return x
