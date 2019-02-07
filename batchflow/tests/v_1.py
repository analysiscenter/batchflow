"""Tests for DatasetIndex class"""
# pylint: disable=no-self-use, missing-docstring, redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=too-few-public-methods, useless-super-delegation
import sys
import pytest
import numpy as np
sys.path.append('../..')
from batchflow.dsindex import DatasetIndex

np.random.seed(0)

SIZE = 10

CONTAINER = {'int': SIZE,
             'list': range(2*SIZE, 4*SIZE),
             'callable': (lambda: np.random.random(SIZE)),
             'self': DatasetIndex(SIZE),
             'big': SIZE**3,
             'str': ['a', 'b', 'c', 'd', 'e', 'f'],
             'small': 2,
             'empty': [],
             '2-dimensional': np.arange(SIZE).reshape(2, -1)}

GROUP_TEST = ['int', 'list', 'big', 'callable', 'self', 'str']


def struct_eval(struct):
    """ Calculate true contents of DatasetIndex.index"""
    np.random.seed(0)
    if isinstance(struct, int):
        return np.arange(struct)
    if isinstance(struct, DatasetIndex):
        return struct.index
    if callable(struct):
        return struct()
    return struct


@pytest.fixture()
def ds_indices():
    """ Containter for all types of DatasetIndex instances,
    that are used for different tests.

    Parameters
    ----------
    CONTAINER : dict
        Contains possible types and contents of DatasetIndex instances.

    SIZE : int
        Defines length of DatasetIndex instances, defined in 'CONTAINER'.

    Returns
    -------
    tuple
        First element of the tuple contains DatasetIndex instance.
        Second element of the tuple contains true constructor of DatasetIndex instance.
    """
    def _ds_indices(name):
        np.random.seed(0)
        return DatasetIndex(CONTAINER[name]), CONTAINER[name]

    return _ds_indices


@pytest.fixture(params=GROUP_TEST)
def all_ds_indices(request, ds_indices):
    return ds_indices(request.param)


class TestSingle:
    """ Contains tests that are applied only to single DatasetIndex
    instance. Instance is spicified at the very first line of each test.
    Every method uses 'ds_indices' fixture.
    """
    def test_build_index_empty(self, ds_indices):
        with pytest.raises(ValueError):
            ds_indices("empty")

    def test_build_index_multidimensional(self, ds_indices):
        with pytest.raises(TypeError):
            ds_indices("2-dimensional")

    def test_get_pos_int(self, ds_indices):
        dsi, _ = ds_indices("int")
        assert dsi.get_pos(SIZE-1) == SIZE-1

    def test_get_pos_slice(self, ds_indices):
        dsi, _ = ds_indices("int")
        assert dsi.get_pos(slice(0, SIZE-1, 2)) == slice(0, SIZE-1, 2)

    def test_get_pos_str(self, ds_indices):
        dsi, _ = ds_indices("str")
        assert dsi.get_pos('a') == 0

    def test_get_pos_iterable(self, ds_indices):
        dsi, _ = ds_indices("int")
        assert (dsi.get_pos(np.arange(SIZE)) == np.arange(SIZE)).all()

    def test_create_batch_pos_true(self, ds_indices):
        dsi, struct = ds_indices("list")
        length = len(dsi)
        left = dsi.create_batch(range(length), pos=True).index
        right = struct_eval(struct)
        assert (left == right).all()

    def test_create_batch_child(self):
        class ChildSet(DatasetIndex):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        dsi = ChildSet(2*SIZE)
        assert isinstance(dsi.create_batch(range(SIZE)), ChildSet)

    def test_next_batch_stopiter_pass(self, ds_indices):
        dsi, _ = ds_indices("small")
        dsi.reset_iter()
        dsi.next_batch(2, n_epochs=None)
        dsi.next_batch(2, n_epochs=None)

    def test_next_batch_stopiter_raise(self, ds_indices):
        dsi, _ = ds_indices("small")
        dsi.reset_iter()
        with pytest.raises(StopIteration):
            dsi.next_batch(2, n_epochs=1)
            dsi.next_batch(2, n_epochs=1)

    def test_next_batch_smaller(self, ds_indices):
        dsi, _ = ds_indices("big")
        dsi.reset_iter()
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=len(dsi)//2,
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == len(dsi)//2

    @pytest.mark.xfail(reason='fails because batch_size > len(dsindex)')
    def test_next_batch_bigger(self, ds_indices):
        dsi, _ = ds_indices("big")
        dsi.reset_iter()
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=int(len(dsi)*1.2),
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == int(len(dsi)*1.2)

class TestMultiple:
    """ Contains tests that are applied to multiple possible
    DatasetIndex instances.
    Every method uses 'all_ds_indices' fixture.

    Tests that have '_baseset_' in the name use methods, inherited
    from Baseset.
    """
    def test_len(self, all_ds_indices):
        dsi, struct = all_ds_indices
        if isinstance(struct, int):
            length = struct
        elif callable(struct):
            length = len(struct())
        else:
            length = len(struct)
        assert len(dsi) == length

    def test_baseset_calc_split_shares(self, all_ds_indices):
        dsi, _ = all_ds_indices
        with pytest.raises(ValueError):
            dsi.calc_split(shares=[0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            dsi.calc_split(shares=[0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            DatasetIndex(2).calc_split(shares=[0.5, 0.5, 0.5])

    def test_baseset_calc_split_correctness_1(self, all_ds_indices):
        dsi, _ = all_ds_indices
        assert sum(dsi.calc_split()) == len(dsi)

    @pytest.mark.xfail(reason='default value is 1, not 0. line 94 of batchflow/base.py')
    def test_baseset_calc_split_correctness_2(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi.calc_split(shares=[0.5, 0.5])
        right = dsi.calc_split(shares=[0.5, 0.5, 0])
        assert left == right

    def test_baseset_calc_split_correctness_3(self, all_ds_indices):
        dsi, _ = all_ds_indices
        length = len(dsi)
        left = dsi.calc_split(shares=[0.5, 0.5])
        right = (0.5*length, 0.5*length, 0)
        assert left == right

    def test_build_index(self, all_ds_indices):
        dsi, struct = all_ds_indices
        struct = struct_eval(struct)
        assert (dsi.index == struct).all()

    def test_shuffle_bool_false(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=False)
        right = np.arange(len(dsi))
        assert (left == right).all()

    def test_shuffle_bool_true(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=True)
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_int(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=SIZE)
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_randomstate(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_cross(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
        right = dsi._shuffle(shuffle=SIZE)
        assert (left == right).all()

    def test_shuffle_bool_callable(self, all_ds_indices):
        dsi, _ = all_ds_indices
        left = dsi._shuffle(shuffle=(lambda _: np.arange(len(dsi))))
        right = np.arange(len(dsi))
        assert (left == right).all()

    @pytest.mark.parametrize("repeat_time", [1]*1)
    def test_split_correctness(self, repeat_time, all_ds_indices):
        dsi, _ = all_ds_indices
        shares = np.random.random(3) *.3 * repeat_time
        dsi.split(shares=shares, shuffle=True)
        assert len(dsi) == (len(dsi.train)
                            + len(dsi.test)
                            + len(dsi.validation))

    def test_create_batch_pos_false(self, all_ds_indices):
        dsi, _ = all_ds_indices
        length = len(dsi)
        left = dsi.create_batch(range(length), pos=False).index
        right = range(length)
        assert (left == right).all()

    def test_create_batch_type(self, all_ds_indices):
        dsi, _ = all_ds_indices
        length = len(dsi)
        assert isinstance(dsi.create_batch(range(length)), DatasetIndex)

    def test_next_batch_drop_last(self, all_ds_indices):
        dsi, _ = all_ds_indices
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=3,
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == 3
