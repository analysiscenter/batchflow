""" Layer to combine multiple inputs into one tensor. """
import numpy as np
import torch
from torch import nn

from ..utils import get_shape, get_num_channels, get_num_dims, pad, center_crop




class Combine(nn.Module):
    """ Combine list of tensor into one.
    For each operation, we call its initialization ('*_initialization' methods) at module init,
    then use its forward ('*_forward' methods) for applying the operation.

    Parameters
    ----------
    inputs : sequence of torch.Tensors
        Tensors to combine.

    op : str or callable
        If callable, then operation to be applied to the list of inputs.
        If 'concat', 'cat', '|', then inputs are concatenated along channels axis.
        If 'sum', '+', then inputs are summed.
        If 'mul', '*', then inputs are multiplied.
        If 'mean', then inputs are averaged.
        If 'drop_path', then inputs are summed with probability:
        for each batch item, there is a chance to not add anything.
    """
    #pylint: disable=attribute-defined-outside-init
    OPS = {
        'concat': ['concat', 'cat', '|'],
        'sum': ['sum', 'plus', '+'],
        'mul': ['multi', 'mul', '*'],
        'mean': ['average', 'avg', 'mean'],
        'drop_path': ['drop_path', 'droppath', 'dp', '!']
    }
    OPS = {alias: method for method, aliases in OPS.items() for alias in aliases}

    def __init__(self, inputs=None, op='concat', force_resize=None, leading_index=0, **kwargs):
        super().__init__()
        self.name = op
        self.kwargs = kwargs
        self.idx = leading_index

        if self.idx != 0:
            inputs = inputs[:]
            inputs[0], inputs[self.idx] = inputs[self.idx], inputs[0]
        self.input_shapes, self.resized_shapes, self.output_shapes = None, None, None
        self.input_ids, self.after_ids = None, None

        if op in self.OPS:
            op_name = self.OPS[op]
            self.op_name = op_name
            if hasattr(self, f'{op_name}_initialization'):
                getattr(self, f'{op_name}_initialization')(inputs, **kwargs)

            self.op = getattr(self, f'{op_name}_forward')
            self.force_resize = force_resize if force_resize is not None else True
        elif callable(op):
            self.op_name = op.__name__
            self.op = op
            self.force_resize = force_resize if force_resize is not None else False
        else:
            raise ValueError(f'Combine op must be a callable or one from {list(self.OPS.keys())}, got {op} instead!')

    def forward(self, inputs):
        # Inputs
        self.input_ids = [id(item) for item in inputs]
        if self.idx != 0:
            inputs = inputs[:]
            inputs[0], inputs[self.idx] = inputs[self.idx], inputs[0]
        self.after_ids = [id(item) for item in inputs]
        self.input_shapes = get_shape(inputs)

        # Resize
        if self.force_resize:
            inputs = self.spatial_resize(inputs)
            self.resized_shapes = get_shape(inputs)

        # Outputs
        output = self.op(inputs)
        self.output_shapes = get_shape(output)
        return output

    def extra_repr(self):
        """ Report shapes before and after combination to a repr. """
        res = f'op={"callable " if not isinstance(self.name, str) else ""}{self.op_name}'
        res += f', leading_idx={self.idx}, force_resize={self.force_resize}'
        for key, value in self.kwargs.items():
            res += f', {key}={value}'

        if getattr(self, 'verbosity', 10) > 2:
            res += f',\n  input_shapes={self.input_shapes}'

            if self.force_resize:
                res += f',\nresized_shapes={self.resized_shapes}'

            res += f',\n output_shapes={self.output_shapes}'

            if getattr(self, 'extra', False):
                res += f',\ninput_ids={self.input_ids}'
                res += f',\nafter_ids={self.after_ids}'
        return res

    def spatial_resize(self, inputs):
        """ Force the same shapes of the inputs, if needed. """
        shape_ = get_shape(inputs[0])
        dim_ = get_num_dims(inputs[0])
        target_shape = shape_[-dim_:]

        resized = [inputs[0]]
        for item in inputs[1:]:
            shape = get_shape(item)
            dim = get_num_dims(item)
            spatial_shape = shape[-dim:]
            if dim > 0 and spatial_shape != tuple([1] * dim) and spatial_shape != target_shape:
                if any(np.array(spatial_shape) < np.array(target_shape)):
                    item = pad(item, spatial_shape, target_shape)
                    if get_shape(item)[-dim:] == target_shape:
                        resized.append(item)
                        continue
                item = center_crop(item, target_shape, dim)
            resized.append(item)
        return resized

    def concat_forward(self, inputs):
        return torch.cat(inputs, dim=1)

    def sum_forward(self, inputs):
        """ Addition with broadcasting. """
        result = 0
        for item in inputs:
            result = result + item
        return result

    def mul_forward(self, inputs):
        """ Multiplication with broadcasting. """
        result = 1
        for item in inputs:
            result = result * item
        return result

    def mean_forward(self, inputs):
        return torch.mean(inputs)


    def drop_path_initialization(self, inputs, drop_prob=0.0, scale=True, layer_scale=1e-6, **kwargs):
        """ Initializa drop path: save supplied args and create trainable parameter. """
        _ = kwargs
        self.drop_prob = drop_prob
        self.scale = scale

        if layer_scale != 0.0:
            x = inputs[1]
            channels = get_num_channels(x)
            gamma_shape = (1, channels) + (1,) * (x.ndim - 2)
            self.gamma = nn.Parameter(layer_scale * torch.ones(gamma_shape, device=x.device), requires_grad=True)
        else:
            self.gamma = None

    def drop_path_forward(self, inputs):
        """ Drop some of the batch items in the second tensor, multiply it by trainable parameter, add. """
        inputs, x = inputs

        # DropPath: drop information about some of the samples
        if self.drop_prob == 0.0 or not self.training:
            pass
        else:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep_prob)

            if self.scale:
                mask.div_(keep_prob)
            x = x * mask

        # LayerScale
        x = x if self.gamma is None else self.gamma * x

        # Residual
        x = inputs + x
        return x
