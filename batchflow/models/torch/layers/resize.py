""" Modules for changing shape of tensors/sequences of tensors. """
import torch
import torch.nn as nn
from . import ConvBlock
from ..utils import get_shape



class Crop(nn.Module):
    """ Crop tensor to desired shape.

    Parameters
    ----------
    inputs
        Input tensor.
    resize_to
        Tensor or shape to resize input tensor to.
    """
    def __init__(self, inputs, resize_to):
        super().__init__()
        i_shape = get_shape(inputs)
        r_shape = get_shape(resize_to)
        self.output_shape = (*i_shape[:2], *r_shape[2:])

    def forward(self, inputs, resize_to):
        i_shape = get_shape(inputs)
        r_shape = get_shape(resize_to)
        if (i_shape[2] > r_shape[2]) or (i_shape[3] > r_shape[3]):
            # Decrease input tensor's shape by slicing desired shape out of it
            shape = [slice(None, c) for c in resize_to.size()[2:]]
            shape = tuple([slice(None, None), slice(None, None)] + shape)
            output = inputs[shape]
        elif (i_shape[2] < r_shape[2]) or (i_shape[3] < r_shape[3]):
            # Increase input tensor's shape by zero padding
            output = torch.zeros(*i_shape[:2], *r_shape[2:])
            output[:, :, :i_shape[2], :i_shape[3]] = inputs
        else:
            output = inputs
        return output


class Combine(nn.Module):
    """ Combine list of tensor into one.

    Parameters
    ----------
    inputs : sequence of torch.Tensors
        Tensors to combine.

    op : str {'concat', 'sum', 'conv'}
        If 'concat', inputs are concated along channels axis.
        If 'sum', inputs are summed.
        If 'softsum', every tensor is passed through 1x1 convolution in order to have
        the same number of channels as the first tensor, and then summed.
    """
    def __init__(self, inputs, op='concat'):
        super().__init__()

        self.op = op
        if op == 'concat':
            shape = list(get_shape(inputs[0]))
            shape[1] = int(sum([get_shape(tensor)[1] for tensor in inputs]))
            self.output_shape = shape
        elif op == 'sum':
            self.output_shape = get_shape(inputs[0])
        elif op == 'softsum':
            args = dict(layout='c', filters=get_shape(inputs[0])[1],
                        kernel_size=1)
            self.conv = [ConvBlock(get_shape(tensor), **args)
                         for tensor in inputs]
            self.output_shape = get_shape(inputs[0])
        else:
            raise ValueError('Combine `op` must be one of `concat`, `sum`, `softsum`, got {}.'.format(op))

    def forward(self, inputs):
        if self.op == 'concat':
            return torch.cat(inputs, dim=1)
        if self.op == 'sum':
            return torch.stack(inputs, dim=0).sum(dim=0)
        if self.op == 'softsum':
            inputs = [self.conv[i](tensor)
                      for i, tensor in enumerate(inputs)]
            return torch.stack(inputs, dim=0).sum(dim=0)
        raise ValueError('Combine `op` must be one of `concat`, `sum`, `softsum`, got {}.'.format(self.op))
