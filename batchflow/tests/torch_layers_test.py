""" Test torch layers """
# pylint: disable=import-error, no-name-in-module
import pytest

import torch
import numpy as np

from batchflow.models.torch.layers.pooling import GlobalMaxPool, GlobalAvgPool, \
    AdaptiveMaxPool, AdaptiveAvgPool, MaxPool, AvgPool, Pool

# TEST_DATA format: (input array,
#                    resulting array,
#                    pooling layer to use, layer parameters)
TEST_DATA = []
TEST_DATA += [(np.arange(16).reshape([2, 1, *shape]),
               np.array([[7], [15]]),
               GlobalMaxPool, {})
              for shape in [(-1,), (-1, 2), (-1, 2, 2)]]
TEST_DATA += [(np.arange(16).reshape([2, 1, *shape]),
               np.array([[3.5], [11.5]]),
               GlobalAvgPool, {})
              for shape in [(-1,), (-1, 2), (-1, 2, 2)]]

TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([3, 7, 11, 15]).reshape([2, 1, 2]),
               AdaptiveMaxPool, {'output_size': [2]})]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([6, 7, 14, 15]).reshape([2, 1, 1, 2]),
               AdaptiveMaxPool, {'output_size': [1, 2]})]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2, 2]),
               np.array([5, 7, 13, 15]).reshape([2, 1, 1, 2, 1]),
               AdaptiveMaxPool, {'output_size': [1, 2, 1]})]

TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1.5, 5.5, 9.5, 13.5]).reshape([2, 1, 2]),
               AdaptiveAvgPool,
               {'output_size': [2]})]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([3, 4, 11, 12]).reshape([2, 1, 1, 2]),
               AdaptiveAvgPool, {'output_size': [1, 2]})]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2, 2]),
               np.array([2.5, 4.5, 10.5, 12.5]).reshape([2, 1, 1, 2, 1]),
               AdaptiveAvgPool, {'output_size': [1, 2, 1]})]

TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1, 4, 7, 9, 12, 15]).reshape([2, 1, -1]),
               MaxPool, dict(pool_size=3, pool_strides=3, padding='same'))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([2, 5, 10, 13]).reshape([2, 1, -1]),
               MaxPool, dict(pool_size=3, pool_strides=3, padding='valid'))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1, 5, 7, 9, 13, 15]).reshape([2, 1, -1]),
               MaxPool, dict(pool_size=4, pool_strides=4, padding=2))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([3, 7, 11, 15]).reshape([2, 1, -1, 1]),
               MaxPool, dict(pool_size=2, pool_strides=2, padding=p))
              for p in ('same', 'valid', )]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([0, 1, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]).reshape([2, 1, -1, 2]),
               MaxPool, dict(pool_size=2, pool_strides=2, padding=1))]

TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1/3, 3, 6, 17/3, 11, 14]).reshape([2, 1, -1]),
               AvgPool, dict(pool_size=3, pool_strides=3, padding='same'))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1, 4, 9, 12]).reshape([2, 1, -1]),
               AvgPool, dict(pool_size=3, pool_strides=3, padding='valid'))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([0.25, 3.5, 3.25, 4.25, 11.5, 7.25]).reshape([2, 1, -1]),
               AvgPool, dict(pool_size=4, pool_strides=4, padding=2))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([1.5, 5.5, 9.5, 13.5]).reshape([2, 1, -1, 1]),
               AvgPool, dict(pool_size=2, pool_strides=2, padding=p))
              for p in ('same', 'valid', )]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1, 2]),
               np.array([0, 0.25, 1.5, 2, 1.5, 1.75, 2, 2.25, 5.5, 6, 3.5, 3.75]).reshape([2, 1, -1, 2]),
               AvgPool, dict(pool_size=2, pool_strides=2, padding=1))]

TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1, 4, 7, 9, 12, 15]).reshape([2, 1, -1]),
               Pool, dict(op='max', pool_size=3, pool_strides=3, padding='same'))]
TEST_DATA += [(np.arange(16).reshape([2, 1, -1]),
               np.array([1/3, 3, 6, 17/3, 11, 14]).reshape([2, 1, -1]),
               Pool, dict(op='avg', pool_size=3, pool_strides=3, padding='same'))]


@pytest.mark.parametrize('inp, res, op, kwargs', TEST_DATA)
def test_pooling(inp, res, op, kwargs):
    """ test for various pooling layers"""
    inp = torch.from_numpy(inp.astype(np.float32))

    out = op(inputs=inp, **kwargs)(inp)
    res = torch.from_numpy(res.astype(np.float32))

    assert torch.allclose(out, res)
