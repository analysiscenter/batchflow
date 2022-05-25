""" Test torch layers """
from contextlib import nullcontext
# pylint: disable=import-error, no-name-in-module
import pytest

import torch
import numpy as np

from batchflow.models.torch.layers.pooling import GlobalMaxPool, GlobalAvgPool, MaxPool, AvgPool
from batchflow.models.torch.layers.core import Flatten

# POOLING_TEST_DATA format: (input array,
#                    resulting array,
#                    pooling layer to use, layer parameters)
POOLING_TEST_DATA = []
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, *shape]),
                       np.array([[7], [15]]),
                       GlobalMaxPool, {})
                      for shape in [(-1,), (-1, 2), (-1, 2, 2)]]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, *shape]),
                       np.array([[3.5], [11.5]]),
                       GlobalAvgPool, {})
                      for shape in [(-1,), (-1, 2), (-1, 2, 2)]]

POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([1, 4, 7, 9, 12, 15]).reshape([2, 1, -1]),
                       MaxPool, dict(pool_size=3, pool_stride=3, padding='same'))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([2, 5, 10, 13]).reshape([2, 1, -1]),
                       MaxPool, dict(pool_size=3, pool_stride=3, padding='valid'))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([1, 5, 7, 9, 13, 15]).reshape([2, 1, -1]),
                       MaxPool, dict(pool_size=4, pool_stride=4, padding=2))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
                       np.array([3, 7, 11, 15]).reshape([2, 1, -1, 1]),
                       MaxPool, dict(pool_size=2, pool_stride=2, padding=p))
                      for p in ('same', 'valid', )]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
                       np.array([0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]).reshape([2, 1, -1, 2]),
                       MaxPool, dict(pool_size=2, pool_stride=2, padding=1))]

POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([1/3, 3, 6, 17/3, 11, 14]).reshape([2, 1, -1]),
                       AvgPool, dict(pool_size=3, pool_stride=3, padding='same'))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([1, 4, 9, 12]).reshape([2, 1, -1]),
                       AvgPool, dict(pool_size=3, pool_stride=3, padding='valid'))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
                       np.array([0.25, 3.5, 3.25, 4.25, 11.5, 7.25]).reshape([2, 1, -1]),
                       AvgPool, dict(pool_size=4, pool_stride=4, padding=2))]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
                       np.array([1.5, 5.5, 9.5, 13.5]).reshape([2, 1, -1, 1]),
                       AvgPool, dict(pool_size=2, pool_stride=2, padding=p))
                      for p in ('same', 'valid', )]
POOLING_TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
                       np.array([0, 0.25, 1.5, 2, 1.5, 1.75, 2, 2.25, 5.5, 6, 3.5, 3.75]).reshape([2, 1, -1, 2]),
                       AvgPool, dict(pool_size=2, pool_stride=2, padding=1))]



@pytest.mark.parametrize('inp, res, op, kwargs', POOLING_TEST_DATA)
def test_pooling(inp, res, op, kwargs):
    """ test for various pooling layers"""
    inp = torch.from_numpy(inp.astype(np.float32))

    out = op(inputs=inp, **kwargs)(inp)
    res = torch.from_numpy(res.astype(np.float32))

    assert torch.allclose(out, res)

# inp_shape, keepdims, out_shape
# all shapes are without batch size
FLATTEN_TEST_DATA = [((2, 1), 0, (2,)), ((3, 2, 1), 0, (6,)),
                     ((2, 1), 1, (2, 1)), ((3, 2, 1), 1, (3, 2)),
                     ((2, 1), 2, (1, 2)), ((3, 2, 1), 2, (2, 3)),
                     ((3, 2, 1), 3, (1, 6))]
# border cases
FLATTEN_TEST_DATA += [((2,), 0, (2,)), ((2,), 1, (2,))]
# add explicit batch dimension to keepdims to ensure the output is same
FLATTEN_TEST_DATA += [(i, (0, kd), o) for i, kd, o in FLATTEN_TEST_DATA]

# border cases, multiple keepdims
FLATTEN_TEST_DATA += [((2, 1), (1, 2), (2, 1)), ((3, 2, 1), (1, 2, 3), (3, 2, 1))]

# add nullcontext to mark valid data
FLATTEN_TEST_DATA = [params + (nullcontext(),) for params in FLATTEN_TEST_DATA]

# invalid data
FLATTEN_TEST_DATA += [((2, 1), 3, (1, 2), pytest.raises(ValueError))]

@pytest.mark.parametrize('inp_shape, keepdims, out_shape, expectation', FLATTEN_TEST_DATA)
@pytest.mark.parametrize('batch_size', [1, 2])
def test_flatten(inp_shape, keepdims, out_shape, expectation, batch_size):
    """ test Flatten """
    inp = torch.zeros((batch_size, ) + inp_shape)

    with expectation:
        out = Flatten(keep_dims=keepdims)(inp)

        assert out.shape[0] == inp.shape[0]
        assert out.shape[1:] == out_shape


@pytest.mark.parametrize('inp_shape', [(2,), (2, 3), (1, 2, 3)])
def test_flatten_default(inp_shape):
    """ test Flatten default parameters """
    batch_size = 2
    inp = torch.zeros((batch_size, ) + inp_shape)

    out = Flatten()(inp)

    assert out.ndim == 2
    assert out.shape == (inp.shape[0], np.prod(inp_shape))
