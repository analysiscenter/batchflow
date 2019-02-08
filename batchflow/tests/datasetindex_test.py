"""Tests for DatasetIndex class"""
# pylint: disable=no-self-use, missing-docstring, redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=too-few-public-methods, useless-super-delegation
import sys
import pytest
import numpy as np
sys.path.append('..')
from batchflow import DatasetIndex


SIZE = 10

CONTAINER = {'int': SIZE,
             'list': range(2*SIZE, 4*SIZE),
             'callable': (lambda: np.arange(SIZE)[::-1]),
             'self': DatasetIndex(SIZE),
             'big': SIZE**3,
             'str': ['a', 'b', 'c', 'd', 'e', 'f'],
             'small': 2,
             'empty': [],
             '2-dimensional': np.arange(SIZE).reshape(2, -1)}

GROUP_TEST = ['int', 'list', 'big', 'callable', 'self', 'str']


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


class TestSingle:
    """ Contains tests that are applied only to single DatasetIndex
    instance. If possible, instance is specified at the very first line of each test.
    Every method uses 'index' fixture.
    """
    def test_build_index_empty(self, index):
        with pytest.raises(ValueError):
            index("empty")

    def test_build_index_multidimensional(self, index):
        with pytest.raises(TypeError):
            index("2-dimensional")

    def test_get_pos_int(self, index):
        dsi = index("int")
        assert dsi.get_pos(SIZE-1) == SIZE-1

    def test_get_pos_slice(self, index):
        dsi = index("int")
        assert dsi.get_pos(slice(0, SIZE-1, 2)) == slice(0, SIZE-1, 2)

    def test_get_pos_str(self, index):
        dsi = index("str")
        assert dsi.get_pos('a') == 0

    def test_get_pos_iterable(self, index):
        dsi = index("int")
        assert (dsi.get_pos(np.arange(SIZE)) == np.arange(SIZE)).all()

    def test_create_batch_pos_true(self, index):
        """ When 'pos' is True, method creates new batch by specified positions. """
        dsi, struct = index("list", struct=True)
        length = len(dsi)
        left = dsi.create_batch(range(length), pos=True).index
        assert (left == struct).all()

    def test_create_batch_child(self):
        """ Method 'create_batch' must be type-preserving. """
        class ChildSet(DatasetIndex):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        dsi = ChildSet(2*SIZE)
        assert isinstance(dsi.create_batch(range(SIZE)), ChildSet)

    def test_next_batch_stopiter_pass(self, index):
        """ When 'n_epochs' is None it is possible to iterate infinitely. """
        dsi = index("small")
        dsi.reset_iter()
        dsi.next_batch(2, n_epochs=None)
        dsi.next_batch(2, n_epochs=None)

    def test_next_batch_stopiter_raise(self, index):
        """ Iteration is blocked after end of DatasetIndex. """
        dsi = index("small")
        dsi.reset_iter()
        dsi.next_batch(2, n_epochs=1)
        with pytest.raises(StopIteration):
            dsi.next_batch(2, n_epochs=1)

    def test_next_batch_smaller(self, index):
        """ Batch_size is twice as small as length DatasetIndex. """
        dsi = index("big")
        dsi.reset_iter()
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=len(dsi)//2,
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == len(dsi)//2

    @pytest.mark.xfail(reason='fails because batch_size > len(dsindex)')
    def test_next_batch_bigger(self, index):
        """ When 'batch_size' is bigger than length of DatasetIndex, the
        behavior is unstable.
        """
        dsi = index("big")
        dsi.reset_iter()
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=int(len(dsi)*1.2),
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == int(len(dsi)*1.2)


class TestMultiple:
    """ Contains tests that are applied to multiple possible
    DatasetIndex instances.
    Every method uses 'all_indices' fixture.

    Tests that have '_baseset_' in the name use methods, inherited
    from Baseset.
    """
    def test_baseset_len(self, all_indices):
        """ True length is recovered from the constructor. """
        dsi, struct = all_indices(struct=True)
        if isinstance(struct, int):
            length = struct
        elif callable(struct):
            length = len(struct())
        else:
            length = len(struct)
        assert len(dsi) == length

    def test_baseset_calc_split_shares(self, all_indices):
        dsi = all_indices()
        with pytest.raises(ValueError):
            dsi.calc_split(shares=[0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            dsi.calc_split(shares=[0.5, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            DatasetIndex(2).calc_split(shares=[0.5, 0.5, 0.5])

    def test_baseset_calc_split_correctness_1(self, all_indices):
        dsi = all_indices()
        assert sum(dsi.calc_split()) == len(dsi)

    def test_baseset_calc_split_correctness_2(self, all_indices):
        """ If 'shares' contains 2 elements, then validation subset can be empty. """
        dsi = all_indices()
        _, _, valid_share = dsi.calc_split(shares=[0.5, 0.5])
        assert valid_share == 0

    def test_baseset_calc_split_correctness_3(self, all_indices):
        """ If 'shares' contains 3 elements, then validation subset is non-empty. """
        dsi = all_indices()
        _, _, valid_share = dsi.calc_split(shares=[0.5, 0.5, 0])
        assert valid_share == 1

    def test_baseset_calc_split_correctness_4(self, all_indices):
        dsi = all_indices()
        length = len(dsi)
        left = dsi.calc_split(shares=[0.5, 0.5])
        right = (0.5*length, 0.5*length, 0)
        assert left == right

    def test_build_index(self, all_indices):
        """ True contents of 'dsi.index' are recovered from the constructor. """
        dsi, struct = all_indices(struct=True)
        if isinstance(struct, int):
            struct = np.arange(struct)
        elif isinstance(struct, DatasetIndex):
            struct = struct.index
        elif callable(struct):
            struct = struct()
        assert (dsi.index == struct).all()

    def test_shuffle_bool_false(self, all_indices):
        dsi = all_indices()
        left = dsi._shuffle(shuffle=False)
        right = np.arange(len(dsi))
        assert (left == right).all()

    def test_shuffle_bool_true(self, all_indices):
        dsi = all_indices()
        left = dsi._shuffle(shuffle=True)
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_int(self, all_indices):
        dsi = all_indices()
        left = dsi._shuffle(shuffle=SIZE)
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_randomstate(self, all_indices):
        dsi = all_indices()
        left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
        right = np.arange(len(dsi))
        assert (left != right).any()
        assert set(left) == set(right)

    def test_shuffle_bool_cross(self, all_indices):
        dsi = all_indices()
        left = dsi._shuffle(shuffle=np.random.RandomState(SIZE))
        right = dsi._shuffle(shuffle=SIZE)
        assert (left == right).all()

    def test_shuffle_bool_callable(self, all_indices):
        """ Callable 'shuffle' should return order. """
        dsi = all_indices()
        left = dsi._shuffle(shuffle=(lambda _: np.arange(len(dsi))))
        right = np.arange(len(dsi))
        assert (left == right).all()

    @pytest.mark.parametrize("repeat_time", [1]*1)
    def test_split_correctness(self, repeat_time, all_indices):
        """ Constants in 'shares' are such that test does not raise errors. """
        dsi = all_indices()
        shares = .3 - np.random.random(3) *.05 *repeat_time
        dsi.split(shares=shares)
        assert len(dsi) == (len(dsi.train)
                            + len(dsi.test)
                            + len(dsi.validation))

    def test_create_batch_pos_false(self, all_indices):
        """ When 'pos' is False, method returns the same, as its first argument. """
        dsi = all_indices()
        length = len(dsi)
        left = dsi.create_batch(range(length), pos=False).index
        right = range(length)
        assert (left == right).all()

    def test_create_batch_type(self, all_indices):
        """ Method 'create_batch' must be type-preserving. """
        dsi = all_indices()
        length = len(dsi)
        assert isinstance(dsi.create_batch(range(length)), DatasetIndex)

    def test_next_batch_drop_last(self, all_indices):
        """ When 'n_epochs' is None it is possible to iterate infinitely. """
        dsi = all_indices()
        for _ in range(SIZE*SIZE):
            n_b = dsi.next_batch(batch_size=3,
                                 n_epochs=None,
                                 drop_last=True)
            assert len(n_b) == 3
