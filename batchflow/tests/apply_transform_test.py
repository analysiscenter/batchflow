""" Tests for Batch apply_transform method. """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=missing-docstring, redefined-outer-name
from itertools import product
from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

from batchflow import Batch, Dataset, P, R


BATCH_SIZE = 2
DATA = np.arange(3*BATCH_SIZE).reshape(BATCH_SIZE, -1) + 1
SEED = 42


# Functions to apply in tests
def one2one(arr1, *args, **kwargs):
    """ Simple function.
    """
    _ = args
    addendum = kwargs.get('addendum', 0)
    result = arr1 + addendum
    return result

def two2one(arr1, arr2, **kwargs):
    """ Simple function.
    """
    addendum = kwargs.get('addendum', 0)
    result = arr1 * arr2 + addendum
    return result

def one2two(arr1, **kwargs):
    """ Simple function.
    """
    addendum = kwargs.get('addendum', 0)
    result = arr1 + addendum
    return result, result

def two2two(arr1, arr2, **kwargs):
    """ Simple function.
    """
    addendum = kwargs.get('addendum', 0)
    result = arr1 * arr2 + addendum
    return result, result


# Testing all possible combinations of SRC_COMPS and DST_COMPS
SRC_OPTS = [DATA, 'comp1', ['comp1'], ('comp1'), ('comp1', 'comp2'), ['comp1', 'comp2']]
DST_OPTS = [DATA, None, 'comp1', ['comp3'], ('comp2'), ('comp2', 'comp3'), ['comp1', 'comp3']]

SRC_COMPS, DST_COMPS = list(zip(*list(product(SRC_OPTS, DST_OPTS))))

# Test is expected to fail when dst=DATA or src=DATA and dst=None
EXPECTATION = [pytest.raises(RuntimeError), does_not_raise(), does_not_raise(),
               does_not_raise(), does_not_raise(), does_not_raise(), does_not_raise()] * 6
EXPECTATION[1] = pytest.raises(RuntimeError)
ADDENDUM = [7, P(R('uniform', seed=SEED))]

# Functions used are defined by src and dst
FUNCTIONS = ([one2one] * 5 + [one2two] * 2) * 4 + ([two2one] * 5 + [two2two] * 2) * 2
FUNCTIONS[29] = two2two
FUNCTIONS[36] = one2one
FUNCTIONS[41] = one2one


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


@pytest.mark.parametrize('addendum', ADDENDUM)
@pytest.mark.parametrize('src,dst,expectation,func', list(zip(SRC_COMPS, DST_COMPS, EXPECTATION, FUNCTIONS)))
def test_apply_transform(src, dst, expectation, func, addendum, batch,):
    """ Test checks for different types and shapes of `src` and `dst`.

    expectation
        Test is expected to fail when `dst` is an array-like, these cases are captured by `expectation`
        parameter.
    func
        Different forms of `src` and `dst` are tested with one of four types of functions: that takes one
        argument and returns one, takes one and return two, takes two - returns one and takes two returns two.
    addendum
        addendum parameter verifies that `apply_transform` correctly hanles keyword arguments and NamedExpressions.
    """
    # Arrange
    if isinstance(addendum, P):
        addendum.name.random_state.seed(seed=SEED)
    # Act
    with expectation:
        batch.apply_transform(func, addendum=addendum, src=src, dst=dst)
    # Assert
        if not isinstance(src, (list, tuple)):
            src = [src]
        if dst is None:
            dst = src
        if not isinstance(dst, (list, tuple)):
            dst = [dst]
        for dst_comp in dst:
            result = getattr(batch, dst_comp)
            func_args = [DATA for src_comp in src]
            if isinstance(addendum, P):
                addendum.name.random_state.seed(seed=SEED)
                addendum = addendum.name.get(batch).reshape(-1, 1)
            assert np.all(np.equal(result, func(*func_args, addendum=addendum)))
