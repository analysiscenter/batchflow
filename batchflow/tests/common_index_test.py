"""Tests for common methods of DataseIndex and FilesIndex classes.
"""
# pylint: disable=missing-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
import os
import shutil

import pytest
import numpy as np

from batchflow import DatasetIndex, FilesIndex, make_rng


@pytest.fixture(scope='module')
def files_setup(request):
    """ Fixture that creates files for tests """
    path = 'fi_test_tmp'

    folders = [path]

    for folder in folders:
        os.mkdir(folder)
        for i in range(5):
            open(os.path.join(folder, f'file_{i}.txt'), 'w').close()

    def fin():
        shutil.rmtree(path)

    request.addfinalizer(fin)
    return path, [name for name in os.listdir(path) if name.endswith('txt')]

@pytest.fixture(params=[5, ['a', 'b', 'c', 'd', 'e'], None])
def index(request, files_setup):
    if isinstance(request.param, int):
        return DatasetIndex(request.param), np.arange(request.param)
    if isinstance(request.param, list):
        return DatasetIndex(request.param), request.param
    path, files = files_setup
    return FilesIndex(path=os.path.join(path, '*')), files

@pytest.fixture(params=[DatasetIndex, FilesIndex])
def small_index(request, files_setup):
    if request.param is DatasetIndex:
        return DatasetIndex(1)
    path, _ = files_setup
    return FilesIndex(path=os.path.join(path, '*1.txt'))


def test_len(index):
    index, _ = index
    assert len(index) == 5

def test_calc_split_raise_1(index):
    index, _ = index
    with pytest.raises(ValueError):
        index.calc_split(shares=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        index.calc_split(shares=[0.5, 0.5, 0.5, 0.5])

def test_calc_split_raise_2(small_index):
    with pytest.raises(ValueError):
        small_index.calc_split(shares=[0.5, 0.3, 0.2])

def test_calc_split_correctness_1(index):
    index, _ = index
    assert sum(index.calc_split()) == 5

def test_calc_split_correctness_2(index):
    """ If 'shares' contains 2 elements, validation subset is empty. """
    index, _ = index
    left = index.calc_split(shares=[0.4, 0.6])
    right = (2, 3, 0)
    assert left == right

def test_calc_split_correctness_3(index):
    """ If 'shares' contains 3 elements, then validation subset is non-empty. """
    index, _ = index
    _, _, valid_share = index.calc_split(shares=[0.5, 0.5, 0])
    assert valid_share == 0


def test_split_correctness(index):
    """ Each element of 'index' is used.
    Constants in 'shares' are such that test does not raise errors.
    """
    index, _ = index
    shares = .3 - np.random.random(3) *.05
    index.split(shares=shares)

    assert set(index.index) == (set(index.train.index)
                                | set(index.test.index)
                                | set(index.validation.index))


def test_get_pos(index):
    index, values = index
    elem = values[4]
    assert index.get_pos(elem) == 4

def test_get_pos_slice(index):
    index, values = index
    elem = values[slice(0, 4, 2)]
    assert (index.get_pos(elem) == np.array([0, 2])).all()

def test_get_pos_iterable(index):
    index, values = index
    elem = values
    assert (index.get_pos(elem) == np.arange(len(values))).all()


def test_shuffle_bool_false(index):
    index, values = index
    right = np.arange(len(values))
    left = index.shuffle(shuffle=False)
    assert (left == right).all()

def test_shuffle_bool_true(index):
    index, values = index
    right = np.arange(len(values))
    left = index.shuffle(shuffle=42)
    assert (left != right).any()

def test_shuffle_int(index):
    index, values = index
    right = make_rng(13).permutation(np.arange(len(values)))
    left = index.shuffle(shuffle=13)
    assert (left == right).all()

def test_shuffle_randomstate(index):
    index, values = index
    right = make_rng(np.random.RandomState(13)).permutation(np.arange(len(values)))
    left = index.shuffle(shuffle=np.random.RandomState(13))
    assert (left == right).all()

def test_create_batch_pos_true(index):
    """ When 'pos' is True, method creates new batch by specified positions. """
    index, values = index
    right = values[:5]
    left = index.create_batch(range(5), pos=True).index
    assert (left == right).all()

def test_create_batch_pos_false_str(index):
    """ When 'pos' is False, method returns the same, as its first argument. """
    index, values = index
    right = np.array([values[0], values[-1]])
    left = index.create_batch(right, pos=False).index
    assert (left == right).all()


def test_next_batch_stopiter_raise(index):
    """ Iteration is blocked after end of DatasetIndex. """
    index, _ = index
    index.next_batch(5, n_epochs=1)
    with pytest.raises(StopIteration):
        index.next_batch(5, n_epochs=1)

def test_next_batch_stopiter_pass(index):
    """ When 'n_epochs' is None it is possible to iterate infinitely. """
    index, _ = index
    for _ in range(10):
        index.next_batch(1, n_epochs=None)

def test_next_batch_drop_last_false_1(index):
    """ When 'drop_last' is False 'next_batch' should cycle through index. """
    index, _ = index
    left = []
    right = list(np.concatenate([index.index, index.index]))
    for length in [3, 3, 4]:
        batch = index.next_batch(batch_size=length,
                                 n_epochs=2,
                                 drop_last=False)
        left.extend(list(batch.index))
    assert left == right

def test_next_batch_drop_last_false_2(index):
    """ When 'drop_last' is False last batch of last epoch can have smaller length. """
    index, _ = index
    left = []
    right = [2]*7 + [1] # first seven batches have length of 2, last contains one item
    for _ in range(8):
        batch = index.next_batch(batch_size=2,
                                 n_epochs=3,
                                 drop_last=False)
        left.append(len(batch))
    assert left == right

def test_next_batch_drop_last_true(index):
    """ Order and contents of generated batches is same at every epoch.
    'shuffle' is False, so dropped indices are always the same.
    """
    index, _ = index
    for _ in range(10):
        batch_1 = index.next_batch(batch_size=2,
                                   n_epochs=None,
                                   drop_last=True,
                                   shuffle=False)
        batch_2 = index.next_batch(batch_size=2,
                                   n_epochs=None,
                                   drop_last=True,
                                   shuffle=False)
        assert (batch_1.index == index.index[:2]).all()
        assert (batch_2.index == index.index[2:4]).all()

def test_next_batch_smaller(index):
    """ 'batch_size' is twice as small as length DatasetIndex. """
    index, _ = index
    for _ in range(10):
        batch = index.next_batch(batch_size=2,
                                 n_epochs=None,
                                 drop_last=True)
        assert len(batch) == 2

def test_next_batch_bigger(index):
    """ When 'batch_size' is bigger than DatasetIndex's length, ValueError is raised
    """
    index, _ = index
    with pytest.raises(ValueError):
        index.next_batch(batch_size=7, n_epochs=None, drop_last=True)
