""" Tests for FilesIndex class. """
# pylint: disable=missing-docstring
# pylint: disable=protected-access
import pytest

from batchflow import FilesIndex

@pytest.mark.parametrize('path', ['', [], ['', '']])
def test_build_index_empty(path):
    findex = FilesIndex(path=path)
    assert len(findex) == 0
    assert findex.index == []

@pytest.mark.parametrize('path,error', [(1, TypeError),
                                      ([2, 3], AttributeError),
                                      ([None], AttributeError)])
def test_build_index_non_path(path, error):
    """ `path` should be string or list of strings """
    with pytest.raises(error):
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
