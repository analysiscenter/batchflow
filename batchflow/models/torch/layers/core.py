""" Basic Torch layers. """
import torch
import torch.nn as nn

from ..utils import get_num_channels, get_num_dims, safe_eval



class Flatten(nn.Module):
    """ Flatten input, optionaly keeping provided dimensions.
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

        new_shape = [x.size(i) for i in self.keep_dims]
        return x.view(*new_shape, -1)


class Dense(nn.Module):
    """ Dense layer.

    Parameters
        ----------
        units : int or srt
            out_features in linear layer. see :meth:`~..utils.safe_eval` for details on str values
        bias : bool, optional
            whether to learn  an additive bias by default True
        flatten : bool, optional
            whether to flatten input prior to feeding it to linear layer, by default True
        keep_dims : int, optional
            dimensions to keep while flattening input, see :class:`~.Flatten`, by default 0
    """
    def __init__(self, units, bias=True, inputs=None, flatten=True, keep_dims=0):
        super().__init__()

        self.flatten = Flatten(keep_dims) if flatten else nn.Identity()

        inputs = self.flatten(inputs)
        in_units = inputs.size(-1)

        if isinstance(units, str):
            units = safe_eval(units, in_units)

        self.linear = nn.Linear(in_units, units, bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)



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



class Dropout(nn.Module):
    """ Multi-dimensional dropout layer.

    Parameters
    ----------
    dropout_rate : float
        The fraction of the input units to drop.

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
                        raise ValueError('Sequence of floats must sum up to one for multisample dropout,\
                                          got instead {}'.format(self.multisample))

                    batch_size = x.shape[0]
                    sizes = [round(batch_size*item) for item in self.multisample[:-1]]
                    residual = batch_size - sum(sizes)
                    sizes += [residual]
                else:
                    raise ValueError('Elements of multisample must be either all ints or floats,\
                                      got instead {}'.format(self.multisample))

                splitted = torch.split(x, sizes)
                dropped = [self.layer(branch) for branch in splitted]
                output = torch.cat(dropped, dim=0)
            else:
                raise ValueError('Unknown type of multisample: {}'.format(self.multisample))
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
