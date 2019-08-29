""" Tests for FilesIndex class. """
# pylint: disable=missing-docstring
# pylint: disable=protected-access
import pytest

from batchflow import FilesIndex

@pytest.mark.parametrize('path', ['', [], ['']])
def test_build_index_empty(path):
    findex = FilesIndex(path=path)
    assert len(findex) == 0
    assert findex.index == []

@pytest.mark.parametrize('path', [1, [2, 3], [None]])
def test_build_index_non_path(path):
    """ `path` should be string or list of strings """
    with pytest.raises(AttributeError):
        FilesIndex(path=path)

def test_build_no_ext():
    pass

def test_build_from_path():
    pass

def test_same_name_in_differen_folders():
    pass

def test_build_from_index():
    pass


def test_get_full_path():
    pass

def test_create_subset():
    pass
