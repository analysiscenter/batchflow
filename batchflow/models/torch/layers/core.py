""" Basic Torch layers. """
import numpy as np
import torch
import torch.nn as nn

from ..utils import get_shape, get_num_channels, get_num_dims, safe_eval



class Flatten(nn.Module):
    """ A module which reshapes inputs into 2-dimension (batch_items, features). """
    def forward(self, x):
        return x.view(x.size(0), -1)



class Dense(nn.Module):
    """ Dense layer. """
    def __init__(self, units=None, bias=True, inputs=None):
        super().__init__()

        in_units = np.prod(get_shape(inputs)[1:])
        if isinstance(units, str):
            units = safe_eval(units, in_units)

        self.linear = nn.Linear(in_units, units, bias)

    def forward(self, x):
        if x.dim() > 2:
            x = Flatten()(x)
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
