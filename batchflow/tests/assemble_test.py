""" Test :meth:`.Batch._assemble` """
# pylint: disable=protected-access
import numpy as np

import pytest

from batchflow import Batch


class FakeBatch(Batch):
    components = ('c1', 'c2')


def assert_arrays_equal(item, true_val):
    assert np.all(item == np.asarray(true_val)) if true_val is not None else (item is None)


def test_handle_exceptions():
    """ Exception is raised if `all_results` contains an Exception object """
    batch = FakeBatch(np.arange(2))

    with pytest.raises(RuntimeError) as err:
        batch._assemble(np.asarray([Exception('Fake exception'), 0]))

    assert 'Could not assemble the batch' in str(err.value)


@pytest.mark.parametrize('all_results, kwargs, res', [
    ([0, 1],           dict(dst='c1'),         dict(c1=[0, 1], c2=None)),
    ([[0, 2], [1, 3]], dict(dst=['c1', 'c2']), dict(c1=[0, 1], c2=[2, 3])),

    # dst_default checking
    ([0, 1],           dict(src='c1'),                 dict(c1=[0, 1], c2=None)),
    ([[0, 2], [1, 3]], dict(dst_default='components'), dict(c1=[0, 1], c2=[2, 3])),
])
def test_assemble(all_results, kwargs, res):
    """ ensure that values from `all_results` are put into components properly """
    batch = FakeBatch(np.arange(2))
    batch._assemble(np.asarray(all_results), **kwargs)

    assert_arrays_equal(batch.c1, res['c1'])
    assert_arrays_equal(batch.c2, res['c2'])
