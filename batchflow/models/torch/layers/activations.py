""" Contains activations. """
import torch
from torch import nn


class RadixSoftmax(nn.Module):
    """ Radix Softmax activation.

    Hang Zhang et al. "`ResNeSt: Split-Attention Networks
    <https://arxiv.org/abs/2004.08955>`_"

    Softmax for items grouped by `cardinality` in each group. Obtaining feature-map is reshaped to the
    shape: (batch, `cardinality`, `radix`, -1) and softmax operation is taken by the first dimention.

    Parameters
    ----------
    radix : int
        The number of splits within a cardinal group. Default is 2.
    cardinality : int
        The number of feature-map groups. Given feature-map is splitted to groups with same size. Default is 1.
    Returns
    -------
    x : torch Tensor
        The output size will be (batch size, `radix`).

    Note
    ----
    If `radix` is 1, common sigmoid is used.
    """
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = torch.nn.functional.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
