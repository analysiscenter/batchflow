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

# Testing all possible combinations of SRC_COMPS and DST_COMPS
SRC_OPTS = [DATA, 'comp1', ['comp1'], ('comp1'), ('comp1', 'comp2'), ['comp1', 'comp2']]
DST_OPTS = [DATA, None, 'comp1', ['comp3'], ('comp2'), ('comp2', 'comp3'), ['comp1', 'comp3']]

SRC_COMPS, DST_COMPS = list(zip(*list(product(SRC_OPTS, DST_OPTS))))

# Test is expected to fail when dst=DATA or src=DATA and dst=None 
EXPECTATION = [pytest.raises(RuntimeError), does_not_raise(), does_not_raise(),
               does_not_raise(), does_not_raise(), does_not_raise(), does_not_raise()] * 6
EXPECTATION[1] = pytest.raises(RuntimeError)

def one2one(arr1, *args, **kwargs):
    """ Simple function.
    """
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

# Functions used are defined by src and dst. 
# Last one is one2one because it used to test same transform of each src
FUNCTIONS = ([one2one] * 5 + [one2two] * 2) * 4 + ([two2one] * 5 + [two2two] * 2) * 2
FUNCTIONS[-1] = one2one
FUNCTIONS[-6] = one2one

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


@pytest.mark.parametrize('src,dst,expectation,func', list(zip(SRC_COMPS, DST_COMPS, EXPECTATION, FUNCTIONS)))
def test_all(src, dst, expectation, func, batch):
    with expectation:
        batch.apply_transform(func, addendum=P(R('uniform', 0, 1)), src=src, dst=dst)

        if not isinstance(src, (list, tuple)):
            src = [src]
        if dst is None:
            dst = src
        if not isinstance(dst, (list, tuple)):
            dst = [dst]
        for dst_comp in dst:
            result = getattr(batch, dst_comp)
            assert np.all((result - func(*[getattr(batch, src_comp) if isinstance(src_comp, str) else src_comp for src_comp in src])) < 1)
