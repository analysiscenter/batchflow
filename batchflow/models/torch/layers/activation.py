""" Contains activations. """
import inspect

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
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        input_dim = x.dim()
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = torch.nn.functional.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        if input_dim > x.dim():
            ones = [1] * (input_dim - x.dim())
            return x.view(*x.shape, *ones)
        return x



class Activation(nn.Module):
    """ Proxy activation module.

    Parameters
    ----------
    activation : str, nn.Module, callable or None
        If None, then identity function `f(x) = x`.
        If str, then name from `torch.nn` or `rsoftmax`.
            `rsoftmax` is a RadixSortmax from the paper: Hang Zhang et al. "`ResNeSt: Split-Attention Networks
            <https://arxiv.org/abs/2004.08955>`_".
        Also can be an instance of activation module (e.g. `torch.nn.ReLU()` or `torch.nn.ELU(alpha=2.0)`),
        or a class of activation module (e.g. `torch.nn.ReLU` or `torch.nn.ELU`),
        or a callable (e.g. `F.relu` or your custom function).

    args
        Positional arguments passed to either class initializer or callable.
    kwargs
        Additional named arguments passed to either class initializer or callable.
    """
    FUNCTIONS = {f.lower(): f for f in dir(nn)}
    FUNCTIONS['rsoftmax'] = RadixSoftmax

    def __init__(self, activation, *args, **kwargs):
        super().__init__()
        self.args, self.kwargs = tuple(), {}
        if isinstance(activation, str):
            name = activation.lower()
            if name in self.FUNCTIONS:
                if isinstance(self.FUNCTIONS[name], str):
                    activation = getattr(nn, self.FUNCTIONS[name])
                else:
                    activation = self.FUNCTIONS[name]
            else:
                raise ValueError('Unknown activation', activation)

        if activation is None:
            self.activation = None
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            # check if activation has `in_place` parameter
            has_inplace = 'inplace' in inspect.getfullargspec(activation).args
            if has_inplace:
                kwargs['inplace'] = True
            self.activation = activation(*args, **kwargs)
        elif callable(activation):
            self.activation = activation
            self.args = args
            self.kwargs = kwargs
        else:
            raise ValueError("Activation can be str, nn.Module or a callable, but given", activation)

    def forward(self, x):
        if self.activation:
            return self.activation(x, *self.args, **self.kwargs)
        return x
