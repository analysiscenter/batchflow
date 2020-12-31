""" Contains activations. """
import torch
from torch import nn


class RadixSoftmax(nn.Module):
    """ Radix Softmax activation.

    Hang Zhang et al. "`ResNeSt: Split-Attention Networks
    <https://arxiv.org/abs/2004.08955>`_"

    Applying the softmax for feature map grouped into `radix` gropus.

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
    def __init__(self, radix, cardinality, add_dims=0):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.add_dims = add_dims

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = torch.nn.functional.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        if self.add_dims > 0:
            ones = [1] * self.add_dims
            return x.view(*x.shape, *ones)
        return x
