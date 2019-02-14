"""Tests for DatasetIndex class.
If needed and possible, type of DatasetIndex is specified at
the very first line of each test.
Tests that have '_baseset_' in the name use methods, inherited from Baseset class.
"""
# pylint: disable=no-self-use, missing-docstring, redefined-outer-name
# pylint: disable=protected-access

import sys
import pytest
import numpy as np
sys.path.append('..')
from batchflow import DatasetIndex


SIZE = 10 # should be divisible by 5

CONTAINER = {'int': SIZE,
             'list': range(2*SIZE, 4*SIZE, 2),
             'callable': (lambda: np.arange(SIZE)[::-1]),
             'self': DatasetIndex(SIZE),
             'big': np.random.randint(SIZE**4),
             'str': ['a', 'b', 'c', 'd', 'e']}

GROUP_TEST = ['int', 'list', 'str']


@pytest.fixture()
def index():
    """ Container for all types of DatasetIndex instances,
    that are used for different tests.

    Parameters
    ----------
    CONTAINER : dict
        Contains possible types and contents of DatasetIndex instances.

    SIZE : int
        Defines length of DatasetIndex instances, defined in 'CONTAINER'.
        Is also used whenever constant int value is needed.

    struct : bool
        Flag that defines whether true constructor of particular DatasetIndex
        is returned

    Returns
    -------
    DatasetIndex or tuple
        First element is always DatasetIndex instance.
        If 'struct' is True, then second element of the tuple
        contains true constructor of DatasetIndex instance.
    """
    def _index(name, struct=False):
        if struct:
            return DatasetIndex(CONTAINER[name]), CONTAINER[name]
        return DatasetIndex(CONTAINER[name])

    return _index


@pytest.fixture(params=GROUP_TEST)
def all_indices(request, index):
    """ Fixture to test multiple cases at once.

    Parameters
    ----------
    GROUP_TEST : list
        List of all types of DatasetIndex instances that should be tested.

    struct : bool
        Flag that defines whether true constructor of particular DatasetIndex
        is returned

    Returns
    -------
    DatasetIndex or tuple
        First element is always DatasetIndex instance.
        If 'struct' is True, then second element of the tuple
        contains true constructor of DatasetIndex instance.
    """
    def _all_indices(struct=False):
        return index(request.param, struct)

    return _all_indices


def test_baseset_len(all_indices):
    """ True length is recovered from the constructor. """
    dsi, struct = all_indices(struct=True)
    if isinstance(struct, int):
        length = struct
    elif callable(struct):
        length = len(struct())
    else:
        length = len(struct)
    assert len(dsi) == length


def test_baseset_calc_split_raise(all_indices):
    dsi = all_indices()
    with pytest.raises(ValueError):
        dsi.calc_split(shares=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        dsi.calc_split(shares=[0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        DatasetIndex(2).calc_split(shares=[0.5, 0.5, 0.5])

def test_baseset_calc_split_correctness_1(all_indices):
    dsi = all_indices()
    assert sum(dsi.calc_split()) == len(dsi)

def test_baseset_calc_split_correctness_2(all_indices):
    """ If 'shares' contains 2 elements, then validation subset can be empty. """
    dsi = all_indices()
    _, _, valid_share = dsi.calc_split(shares=[0.5, 0.5])
    assert valid_share == 0

def test_baseset_calc_split_correctness_3(all_indices):
    """ If 'shares' contains 3 elements, then validation subset is non-empty. """
    dsi = all_indices()
    _, _, valid_share = dsi.calc_split(shares=[0.5, 0.5, 0])
    assert valid_share == 1

def test_baseset_calc_split_correctness_4(all_indices):
    dsi = all_indices()
    length = len(dsi)
    left = dsi.calc_split(shares=[0.4, 0.6])
    right = (0.4*length, 0.6*length, 0)
    assert left == right


def test_build_index_empty():
    with pytest.raises(ValueError):
        DatasetIndex([])

def test_build_index_multidimensional():
    with pytest.raises(TypeError):
        DatasetIndex([[1], [2]])

@pytest.mark.parametrize('dsi_type', list(CONTAINER.keys()))
def test_build_index(dsi_type, index):
    """ True contents of 'dsi.index' are recovered from the constructor. """
    dsi, struct = index(dsi_type, struct=True)
    if isinstance(struct, int):
        struct = np.arange(struct)
    elif isinstance(struct, DatasetIndex):
        struct = struct.index
    elif callable(struct):
        struct = struct()
    assert (dsi.index == struct).all()


def test_get_pos_int(index):
    dsi = index("int")
    assert dsi.get_pos(SIZE-1) == SIZE-1

def test_get_pos_slice(index):
    dsi = index("int")
    assert dsi.get_pos(slice(0, SIZE-1, 2)) == slice(0, SIZE-1, 2)

def test_get_pos_str(index):
    dsi = index("str")
    assert dsi.get_pos('c') == 2

def test_get_pos_iterable(index):
    dsi = index("int")
    assert (dsi.get_pos(np.arange(SIZE)) == np.arange(SIZE)).all()


def test_shuffle_bool_false(all_indices):
    dsi = all_indices()
    left = dsi._shuffle(shuffle=False)
    right = np.arange(len(dsi))
    assert (left == right).all()

def test_shuffle_bool_true(all_indices):
    dsi = all_indices()
    left = dsi._shuffle(shuffle=True)
    right = np.arange(len(dsi))
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_int(all_indices):
    dsi = all_indices()
    left = dsi._shuffle(shuffle=SIZE)
    right = np.arange(len(dsi))
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_randomstate(all_indices):
    dsi = all_indices()
    left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
    right = np.arange(len(dsi))
    assert (left != right).any()
    assert set(left) == set(right)

def test_shuffle_bool_cross(all_indices):
    dsi = all_indices()
    left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
    right = dsi._shuffle(shuffle=SIZE)
    assert (left == right).all()

def test_shuffle_bool_callable(all_indices):
    """ Callable 'shuffle' should return order. """
    dsi = all_indices()
    left = dsi._shuffle(shuffle=(lambda _: np.arange(len(dsi))))
    right = np.arange(len(dsi))
    assert (left == right).all()


@pytest.mark.parametrize("repeat_time", [1]*1)
def test_split_correctness(repeat_time, all_indices):
    """ Constants in 'shares' are such that test does not raise errors. """
    dsi = all_indices()
    shares = .3 - np.random.random(3) *.05 *repeat_time
    dsi.split(shares=shares)
    assert len(dsi) == (len(dsi.train)
                        + len(dsi.test)
                        + len(dsi.validation))
    assert set(dsi.index) == (set(dsi.train.index)
                              | set(dsi.test.index)
                              | set(dsi.validation.index))


def test_create_batch_child():
    """ Method 'create_batch' must be type-preserving. """
    class ChildSet(DatasetIndex):
        # pylint: disable=too-few-public-methods
        pass
    dsi = ChildSet(2*SIZE)
    assert isinstance(dsi.create_batch(range(SIZE)), ChildSet)

def test_create_batch_pos_true(index):
    """ When 'pos' is True, method creates new batch by specified positions. """
    dsi, struct = index("list", struct=True)
    length = len(dsi)
    left = dsi.create_batch(range(length), pos=True).index
    assert (left == struct).all()

def test_create_batch_pos_false(all_indices):
    """ When 'pos' is False, method returns the same, as its first argument. """
    dsi = all_indices()
    length = len(dsi)
    left = dsi.create_batch(range(length), pos=False).index
    right = range(length)
    assert (left == right).all()

def test_create_batch_type(all_indices):
    """ Method 'create_batch' must be type-preserving. """
    dsi = all_indices()
    length = len(dsi)
    assert isinstance(dsi.create_batch(range(length)), DatasetIndex)


def test_next_batch_stopiter_raise(all_indices):
    """ Iteration is blocked after end of DatasetIndex. """
    dsi = all_indices()
    length = len(dsi)
    dsi.next_batch(length, n_epochs=1)
    with pytest.raises(StopIteration):
        dsi.next_batch(length, n_epochs=1)

def test_next_batch_stopiter_pass(all_indices):
    """ When 'n_epochs' is None it is possible to iterate infinitely. """
    dsi = all_indices()
    left = set()
    right = set(dsi.index)
    for _ in range(SIZE**2):
        n_b = dsi.next_batch(1, n_epochs=None)
        left = left | set(n_b.index)
    assert left == right

def test_next_batch_drop_last_false_1(all_indices):
    """ When 'drop_last' is False 'next_batch' should cycle through index. """
    dsi = all_indices()
    lengths = [int(0.6*len(dsi)), int(0.6*len(dsi)), int(0.8*len(dsi))]
    left = []
    right = list(np.concatenate([dsi.index, dsi.index]))
    for item in lengths:
        n_b = dsi.next_batch(batch_size=item,
                             n_epochs=2,
                             drop_last=False)
        left.extend(list(n_b.index))
    assert left == right

def test_next_batch_drop_last_false_2(all_indices):
    """ When 'drop_last' is False last batch of last epoch can have smaller length. """
    dsi = all_indices()
    length = len(dsi)
    left = []
    right = [int(0.4*length)]*7 + [int(0.2*length)]
    for _ in range(8):
        n_b = dsi.next_batch(batch_size=int(0.4*length),
                             n_epochs=3,
                             drop_last=False)
        left.append(len(n_b))
    assert left == right

def test_next_batch_drop_last_true_1(all_indices):
    """ Order and contents of generated batches is same at every epoch.
    'shuffle' is False, so dropped indices are always the same.
    """
    dsi = all_indices()
    length = len(dsi)
    right_1 = list(dsi.index[:int(0.4*length)])
    right_2 = list(dsi.index[int(0.4*length):int(0.8*length)])
    for _ in range(SIZE**2):
        n_b_1 = dsi.next_batch(batch_size=int(0.4*length),
                               n_epochs=None,
                               drop_last=True,
                               shuffle=False)
        n_b_2 = dsi.next_batch(batch_size=int(0.4*length),
                               n_epochs=None,
                               drop_last=True,
                               shuffle=False)
        assert list(n_b_1.index) == right_1
        assert list(n_b_2.index) == right_2

def test_next_batch_drop_last_true_2(all_indices):
    """ Order and contents of generated batches may differ at different epochs.
    'shuffle' is True, so dropped indices are different at every epoch.
    """
    dsi = all_indices()
    length = len(dsi)
    left = set()
    right = set(dsi.index)
    for _ in range(SIZE**3):
        n_b = dsi.next_batch(batch_size=int(0.4*length),
                             n_epochs=None,
                             drop_last=True,
                             shuffle=True)
        left = left | set(n_b.index)
    assert left == right

def test_next_batch_smaller(all_indices):
    """ 'batch_size' is twice as small as length DatasetIndex. """
    dsi = all_indices()
    length = len(dsi)
    for _ in range(SIZE**2):
        n_b = dsi.next_batch(batch_size=length//2,
                             n_epochs=None,
                             drop_last=True)
        assert len(n_b) == length//2

@pytest.mark.xfail(reason='fails because batch_size > len(dsindex)')
def test_next_batch_bigger(all_indices):
    """ When 'batch_size' is bigger than length of DatasetIndex, the
    behavior is unstable.
    """
    dsi = all_indices()
    length = len(dsi)
    for _ in range(SIZE**2):
        n_b = dsi.next_batch(batch_size=length*2,
                             n_epochs=None,
                             drop_last=True)
        assert len(n_b) == length*2
