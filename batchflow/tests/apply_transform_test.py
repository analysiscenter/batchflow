""" Tests for Batch apply_transform method. """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=missing-docstring, redefined-outer-name
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

from batchflow import Batch, Dataset, P, R


BATCH_SIZE = 2
DATA = np.arange(3*BATCH_SIZE).reshape(BATCH_SIZE, -1)

SINGLE_CASE = ['comp1', ['comp1'], DATA, ('comp1')]
MULTI_CASE = [['comp1', 'comp2'], ('comp1', 'comp2'),
              ('comp1', DATA), ['comp1', 'comp3']]
EXPECTATION = [does_not_raise(), does_not_raise(),
               pytest.raises(RuntimeError), does_not_raise()]

def return_single_value(arr, arg1, arg2=0):
    """ Simple function that returns single value.
    """
    return arr * arg1 + arg2

def return_two_values(arr, arg1, arg2=0):
    """ Simple function that returns two values.
    """
    return arr * 2, arr * arg1 + arg2


@pytest.fixture
def batch():
    """ Prepare batch and load same DATA to comp1 and comp2 components.
    """
    dataset = Dataset(range(BATCH_SIZE), Batch)
    batch = (dataset.next_batch(BATCH_SIZE)
             .load(src=DATA, dst='comp1')
             .load(src=DATA, dst='comp2')
            )
    return batch


@pytest.mark.parametrize('src', SINGLE_CASE)
@pytest.mark.parametrize('dst,expectation', list(zip(SINGLE_CASE, EXPECTATION)))
def test_one_to_one(src, dst, expectation, batch):
    with expectation:
        batch.apply_transform(return_single_value, 3, arg2=1, src=src, dst=dst)
        assert np.all(np.equal(batch.comp1, DATA * 3 + 1))
        assert np.all(np.equal(batch.comp2, DATA))

@pytest.mark.parametrize('src', SINGLE_CASE)
@pytest.mark.parametrize('dst,expectation', list(zip(MULTI_CASE, EXPECTATION)))
def test_one_to_many(src, dst, expectation, batch):
    with expectation:
        batch.apply_transform(return_two_values, 3, arg2=1, src=src, dst=dst)
        assert np.all(np.equal(getattr(batch, dst[0]), DATA * 2))
        assert np.all(np.equal(getattr(batch, dst[1]), DATA * 3 + 1))

@pytest.mark.parametrize('src', MULTI_CASE[:2])
@pytest.mark.parametrize('dst,expectation', list(zip(SINGLE_CASE, EXPECTATION)))
def test_many_to_one(src, dst, expectation, batch):
    with expectation:
        batch.apply_transform(return_single_value, arg2=1, src=src, dst=dst)
        assert np.all(np.equal(batch.comp1, DATA * DATA + 1))
        assert np.all(np.equal(batch.comp2, DATA))

@pytest.mark.parametrize('src', MULTI_CASE[:2])
@pytest.mark.parametrize('dst,expectation', list(zip(SINGLE_CASE, EXPECTATION)))
def test_many_to_one_two_values(src, dst, expectation, batch):
    with expectation:
        batch.apply_transform(return_two_values, arg2=1, src=src, dst=dst)
        assert np.all(np.equal(batch.comp1, np.array(list(zip(DATA * 2, DATA * DATA + 1)))))
        assert np.all(np.equal(batch.comp1.shape, (len(src), BATCH_SIZE, 3)))

@pytest.mark.parametrize('dst', [MULTI_CASE[0], None])
def test_many_to_many(dst, batch):
    batch.apply_transform(return_single_value, 30000, arg2=P(R('uniform', 0, 1)),
                          src=MULTI_CASE[0], dst=dst)
    assert np.all(np.allclose(batch.comp1, DATA * 30000, atol=1.))
    assert np.all(np.allclose(batch.comp2, DATA * 30000, atol=1.))
    assert np.all(batch.comp1 - DATA * 30000 == batch.comp2 - DATA * 30000)

def test_many_to_many_two_values(batch):
    batch.apply_transform(return_two_values, 3, arg2=1,
                          src=MULTI_CASE[0], dst=MULTI_CASE[0])
    assert np.all(np.equal(batch.comp1, batch.comp2))
    assert np.all(np.equal(batch.comp1, np.array(list(zip(DATA * 2, DATA * 3 + 1)))))
    assert np.all(np.equal(batch.comp1.shape, (len(MULTI_CASE[0]), 2, 3)))

@pytest.mark.parametrize('src,expectation', list(zip(SINGLE_CASE, EXPECTATION)))
def test_one_to_one_with_default(src, expectation, batch):
    with expectation:
        batch.apply_transform(return_single_value, 3, arg2=1, src=src, dst=None)
        assert np.all(np.equal(batch.comp1, DATA * 3 + 1))
        assert np.all(np.equal(batch.comp2, DATA))
