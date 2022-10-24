# pylint: disable=redefined-outer-name, missing-docstring
import sys

import pytest
import numpy as np

sys.path.append('..')
from batchflow import Dataset, Batch, apply_parallel, B


class MyBatch(Batch):
    components = 'images', 'masks'

    @apply_parallel
    def ap_test(self, item, param, **kwargs):
        _ = kwargs
        if isinstance(item, tuple):
            return item[0] * param + item[1] * param
        return item * param

    @apply_parallel(requires_rng=True)
    def ap_requires_rng_test(self, item, rng=None, **kwargs):
        _ = kwargs
        if isinstance(item, tuple):
            return item[0] + item[1] + rng.uniform()/CONST + CONST
        return item + rng.uniform()


BATCH_SIZE = 8
ARRAY_INIT = np.arange(BATCH_SIZE).reshape((-1, 1))
CONST = 42


@pytest.mark.parametrize('src_dst', [
    ('images', 'masks'),
    (['images', 'masks'], ['outputs1', 'outputs2']),
    (('images', 'masks'), 'outputs'),
])
def test_apply_parallel(src_dst):
    """ Check `apply_parallel` is evaluated properly """
    src, dst = src_dst

    pipeline = (Dataset(10, MyBatch)
        .pipeline()
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ap_test(src=src, dst=dst, param=CONST)
    )

    b = pipeline.next_batch(BATCH_SIZE)

    if isinstance(src, str):
        assert (getattr(b, src) * CONST == getattr(b, dst)).all()
    elif isinstance(src, list):
        for src_, dst_ in zip(src, dst):
            assert (getattr(b, src_) * CONST== getattr(b, dst_)).all()
    elif isinstance(src, tuple):
        assert (ARRAY_INIT * CONST * 2 == b.outputs).all()


@pytest.mark.parametrize('src_dst', [
    ('images', 'masks'),
    (['images', 'images'], ['outputs1', 'outputs2']),
    (('images', 'masks'), 'outputs'),
])
def test_apply_parallel_requires_rng(src_dst):
    """ Check that `rng`, supplied by applied parallel, works and
    reproduces the same results for each `src` in a list.
    """
    src, dst = src_dst

    pipeline = (Dataset(10, MyBatch)
        .pipeline()
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ap_requires_rng_test(src=src, dst=dst)
    )

    b = pipeline.next_batch(BATCH_SIZE)

    if isinstance(src, str):
        assert (getattr(b, src) != getattr(b, dst)).any()
        assert np.allclose(getattr(b, src), getattr(b, dst), atol=1)
    elif isinstance(src, list):
        assert (getattr(b, dst[0]) == getattr(b, dst[1])).all()
    elif isinstance(src, tuple):
        assert np.allclose(b.outputs, ARRAY_INIT * 2 + CONST, atol=1/CONST)

def test_apply_parallel_requires_rng_fixed_seed():
    """ Check that `shuffle`, supplied at pipeline run, produces the same results for multiple runs. """
    pipeline = (Dataset(10, MyBatch)
        .pipeline()
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ap_requires_rng_test(src='images', dst='outputs')
    )
    b = pipeline.next_batch(BATCH_SIZE, shuffle=42)
    value1 = b.random.uniform()
    outputs1 = b.outputs

    pipeline = (Dataset(10, MyBatch)
        .pipeline()
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ap_requires_rng_test(src='images', dst='outputs')
    )
    b = pipeline.next_batch(BATCH_SIZE, shuffle=42)
    value2 = b.random.uniform()
    outputs2 = b.outputs

    assert value1 == value2
    assert (outputs1 == outputs2).all()
