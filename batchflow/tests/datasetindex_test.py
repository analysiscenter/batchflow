"""Tests for DatasetIndex class.
If possible, methods are tested against DatasetIndex with length of 5.
When random values are needed, 'random_seed' is fixed to be 13.
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access
import pytest
import numpy as np

from batchflow import DatasetIndex

@pytest.mark.parametrize('constructor', [5,
                                         range(10, 20, 2),
                                         DatasetIndex(5),
                                         ['a', 'b', 'c', 'd', 'e'],
                                         (lambda: np.arange(5)[::-1])])
def test_build_index(constructor):
    """ True content of 'dsi.index' is recovered from the 'constructor'. """
    dsi = DatasetIndex(constructor)
    if isinstance(constructor, int):
        constructor = np.arange(constructor)
    elif isinstance(constructor, DatasetIndex):
        constructor = constructor.index
    elif callable(constructor):
        constructor = constructor()
    assert (dsi.index == constructor).all()

def test_build_non_unique_indices_warning():
    with pytest.warns(UserWarning):
        DatasetIndex([1, 1, 1])

def test_build_index_empty():
    with pytest.raises(ValueError):
        DatasetIndex([])

def test_build_index_multidimensional():
    with pytest.raises(TypeError):
        DatasetIndex([[1], [2]])

def test_create_batch_child():
    """ Method 'create_batch' must be type-preserving. """
    class ChildSet(DatasetIndex):
        # pylint: disable=too-few-public-methods
        pass
    dsi = ChildSet(5)
    assert isinstance(dsi.create_batch(range(5)), ChildSet)
