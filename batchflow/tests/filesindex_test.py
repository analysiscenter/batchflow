""" Tests for FilesIndex class. """
# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
import os
import shutil

from contextlib import ExitStack as does_not_raise

import pytest
import numpy as np

from batchflow import FilesIndex, DatasetIndex


@pytest.fixture(scope='module')
def files_setup(request):
    """ Fixture that creates files for tests """
    path = 'fi_test_tmp'
    folder1 = 'folder'
    folder2 = 'other_folder'

    folders = [path, os.path.join(path, folder1), os.path.join(path, folder2)]

    for folder in folders:
        os.mkdir(folder)
        for i in range(3):
            open(os.path.join(folder, 'file_{}.txt'.format(i)), 'w').close()

    def fin():
        shutil.rmtree(path)

    request.addfinalizer(fin)
    return path, folder1, folder2


@pytest.mark.parametrize('path, expectation', [['', does_not_raise()],
                                               [[], pytest.raises(ValueError)],
                                               [['', ''], does_not_raise()]])
def test_build_index_empty(path, expectation):
    with expectation:
        findex = FilesIndex(path=path)
        assert len(findex) == 0
        assert isinstance(findex.index, np.ndarray)

@pytest.mark.parametrize('path, error', [(1, TypeError),
                                         ([2, 3], TypeError),
                                         ([None], TypeError)])
def test_build_index_non_path(path, error):
    """ `path` should be string or list of strings """
    with pytest.raises(error):
        FilesIndex(path=path)

def test_build_no_ext(files_setup):
    path, _, _ = files_setup
    path = os.path.join(path, '*')
    findex = FilesIndex(path=path, no_ext=True)
    assert len(findex) == 3
    assert os.path.splitext(findex.indices[0])[1] == ''

def test_build_dirs(files_setup):
    path, folder1, _ = files_setup
    path = os.path.join(path, '*')
    findex = FilesIndex(path=path, dirs=True, sort=True)
    assert len(findex) == 2
    assert findex.indices[0] == os.path.split(folder1)[1]

def test_same_name_in_differen_folders(files_setup):
    path, _, _ = files_setup
    path = os.path.join(path, '*', '*')
    with pytest.raises(ValueError):
        FilesIndex(path=path)

def test_build_from_index(files_setup):
    path, _, _ = files_setup
    files = ['file_{}.txt'.format(i) for i in range(3)]
    paths = dict(zip(files, [os.path.join(path, f) for f in files]))
    dsindex = DatasetIndex(files)
    findex = FilesIndex(index=dsindex, paths=paths, dirs=False)
    assert len(dsindex) == len(findex)

def test_get_full_path(files_setup):
    path, _, _ = files_setup
    findex = FilesIndex(path=os.path.join(path, '*'))
    file_name = 'file_1.txt'
    full_path = findex.get_fullpath(file_name)
    assert os.path.dirname(full_path) == path
    assert os.path.basename(full_path) == file_name

@pytest.mark.parametrize('index', [DatasetIndex(['file_1.txt']), ['file_1.txt']])
def test_create_subset(files_setup, index):
    path, _, _ = files_setup
    findex = FilesIndex(path=os.path.join(path, '*'))
    new_findex = findex.create_subset(index)
    file_name = 'file_1.txt'
    full_path = new_findex.get_fullpath(file_name)
    assert len(new_findex) == 1
    assert isinstance(new_findex.indices, np.ndarray)
    assert os.path.dirname(full_path) == path
    assert os.path.basename(full_path) == file_name
