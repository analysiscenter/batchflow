""" Basic Torch layers. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_shape, get_num_channels, get_num_dims, safe_eval



class Flatten(nn.Module):
    """ Flatten input, optionally keeping provided dimensions.
    Batch dimension is always preserved """
    def __init__(self, keep_dims=0):
        super().__init__()
        if isinstance(keep_dims, int):
            keep_dims = (keep_dims, )
        if 0 not in keep_dims:
            keep_dims = (0, ) + keep_dims
        self.keep_dims = sorted(set(keep_dims))

    def forward(self, x):
        if self.keep_dims[-1] >= x.ndim:
            msg = f'Not enough dimensions in input! keep_dims={self.keep_dims}, but input shape is {x.shape}'
            raise ValueError(msg)

        if len(self.keep_dims) == x.ndim:
            return x

        for dim1, dim2 in enumerate(self.keep_dims):
            x = x.transpose(dim1, dim2)
        new_shape = [x.size(i) for i in range(len(self.keep_dims))]
        return x.reshape(*new_shape, -1)


class Dense(nn.Module):
    """ Dense layer.

    Parameters
    ----------
    features : int or srt
        Out_features in linear layer. see :meth:`~..utils.safe_eval` for details on str values.

    bias : bool, optional
        Whether to learn  an additive bias by default True.

    flatten : bool, optional
        Whether to flatten input prior to feeding it to linear layer, by default True.

    keep_dims : int, optional
        Dimensions to keep while flattening input, see :class:`~.Flatten`, by default 0.
    """
    def __init__(self, features, bias=True, inputs=None, flatten=True, keep_dims=0):
        super().__init__()

        self.flatten = Flatten(keep_dims) if flatten else nn.Identity()

        inputs = self.flatten(inputs)
        in_features = inputs.size(-1)

        if isinstance(features, str):
            features = safe_eval(features, in_features)

        self.linear = nn.Linear(in_features, features, bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class DenseAlongAxis(nn.Module):
    """ Dense layer along specified axis. Completely equivalent to a 1x1 convolution along the same axis. """
    def __init__(self, inputs=None, features=None, axis=1, bias=True):
        super().__init__()
        self.axis = axis

        # Move `axis` to the `1` position
        permuted_axis_order = list(range(inputs.ndim))
        permuted_axis_order[1], permuted_axis_order[axis] = axis, 1
        self.permuted_axis_order = permuted_axis_order
        inputs = inputs.permute(*permuted_axis_order)                                       # (B, C, H, W)

        # Flatten rest of the axes; swap order of the last two axes
        inputs = inputs.flatten(2).transpose(1, 2)                                          # (B, H*W, C)
        in_features = inputs.size(-1)

        # Apply linear: only along the last axis
        if isinstance(features, str):
            features = safe_eval(features, in_features)
        self.layer = nn.Linear(in_features=in_features, out_features=features, bias=bias)   # (B, H*W, C2)

    def forward(self, x):
        # Compute shape of the outputs
        final_shape = list(get_shape(x))
        final_shape[0], final_shape[self.axis] = -1, self.layer.out_features

        x = x.permute(*self.permuted_axis_order)         # (B, C, H, W), move `axis` to the 1 position
        x = x.flatten(2).transpose(1, 2)                 # (B, H*W, C)
        x = self.layer(x)                                # (B, H*W, C2)

        x = x.permute(0, 2, 1)                           # (B, C2, H*W)
        x = x.reshape(*final_shape)                      # (B, C2, H, W)
        return x



class BatchNorm(nn.Module):
    """ Multi-dimensional batch normalization layer """
    LAYERS = {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    }

    def __init__(self, inputs=None, **kwargs):
        super().__init__()
        num_features = get_num_channels(inputs)
        self.layer = self.LAYERS[get_num_dims(inputs)](num_features=num_features, **kwargs)

    def forward(self, x):
        return self.layer(x)


class LayerNorm(nn.Module):
    """ Layer normalization layer. Works with both `channels_first` and `channels_last` format. """
    def __init__(self, inputs=None, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.data_format = data_format
        if data_format == 'channels_last':
            self.channels = get_shape(inputs)[-1]
        else:
            self.channels = get_num_channels(inputs)

        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.eps = eps


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)

        # Use multiplication instead of `pow` as `torch2trt` throws warnings about it
        mean = x.mean(1, keepdim=True)
        normalized = x - mean
        std = (normalized * normalized).mean(1, keepdim=True)
        std = torch.sqrt(std + self.eps)
        x = normalized / std
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f'data_format={self.data_format}'


class Dropout(nn.Module):
    """ Multi-dimensional dropout layer.

    Parameters
    ----------
    dropout_rate : float
        The fraction of the input features to drop.

    multisample: bool, number, sequence
        If evaluates to True, then either multiple dropout applied to the whole batch and then averaged, or
        batch is split into multiple parts, each passed through dropout and then concatenated back.

        If True, then two different dropouts are applied to whole batch.
        If integer, then that number of different dropouts are applied to whole batch.
        If float, then batch is split into parts of `multisample` and `1 - multisample` sizes.
        If sequence of ints, then batch is split into parts of given sizes. Must sum up to the batch size.
        If sequence of floats, then each float means proportion of sizes in batch and must sum up to 1.
    """
    LAYERS = {
        1: nn.Dropout,
        2: nn.Dropout2d,
        3: nn.Dropout3d,
    }

    def __init__(self, inputs=None, dropout_rate=0.0, multisample=False):
        super().__init__()
        multisample = 2 if multisample is True else multisample
        multisample = [multisample, 1 - multisample] if isinstance(multisample, float) else multisample
        self.multisample = multisample

        self.layer = self.LAYERS[get_num_dims(inputs)](p=dropout_rate)

    def forward(self, x):
        if self.multisample is not False:
            if isinstance(self.multisample, int): # dropout to the whole batch, then average
                dropped = [self.layer(x) for _ in range(self.multisample)]
                output = torch.mean(torch.stack(dropped), dim=0)
            elif isinstance(self.multisample, (tuple, list)): # split batch into separate-dropout branches
                if all(isinstance(item, int) for item in self.multisample):
                    sizes = self.multisample
                elif all(isinstance(item, float) for item in self.multisample):
                    if sum(self.multisample) != 1.:
                        raise ValueError(f'Sequence of floats must sum up to one, got {self.multisample} instead!')

                    batch_size = x.shape[0]
                    sizes = [round(batch_size*item) for item in self.multisample[:-1]]
                    residual = batch_size - sum(sizes)
                    sizes += [residual]
                else:
                    raise ValueError(f'Elements of multisample must be either all ints or floats, '
                                     f'got "{self.multisample}" instead!')

                splitted = torch.split(x, sizes)
                dropped = [self.layer(branch) for branch in splitted]
                output = torch.cat(dropped, dim=0)
            else:
                raise ValueError(f'Unknown type of multisample: {self.multisample}')
        else:
            output = self.layer(x)
        return output


class AlphaDropout(Dropout):
    """ Multi-dimensional alpha-dropout layer. """
    LAYERS = {
        1: nn.AlphaDropout,
        2: nn.AlphaDropout,
        3: nn.AlphaDropout,
    }
