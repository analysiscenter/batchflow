""" Test rebatch action """
import numpy as np

import pytest

from batchflow import Dataset, Pipeline, Batch, B


class MyBatch(Batch):
    components = ('dummy', )


DATASET_SIZE = 60
PARAMETERS = [(20, 30), (1, 10), (10, 60), (1, 60), (15, 40), (13, 17)]
PARAMETERS = PARAMETERS + [(b, a) for a, b in PARAMETERS]


def check_batch_lengths(batch_lengths, batch_size):
    """
    check that list of batch lengths agrees with `DATASET_SIZE` and `batch_size`
    """
    expected_iterations = np.ceil(DATASET_SIZE / batch_size)
    assert len(batch_lengths) == expected_iterations

    expected_last_batch_len = DATASET_SIZE % batch_size
    if expected_last_batch_len == 0:
        expected_last_batch_len = batch_size
    assert batch_lengths[-1] == expected_last_batch_len

    if expected_iterations > 1:
        assert all(item == batch_size for item in batch_lengths[:-1])


@pytest.mark.parametrize('batch_size, rebatch_size', PARAMETERS)
def test_rebatch(batch_size, rebatch_size):
    """ checks that rebatch produces batches of expected lengths (and doesn't crash)"""
    data = np.vstack([np.array([i, i]) for i in range(DATASET_SIZE)])
    data = (data,)
    dataset = Dataset(index=DATASET_SIZE,
                      batch_class=MyBatch,
                      preloaded=data)

    # workaround for pipeline variables getting lost after rebatch
    batch_lengths = {'before': [], 'after': []}

    def get_batch_len(batch, dump):
        batch_lengths[dump].append(batch.size)

    p = (Pipeline()
         .call(get_batch_len, B(), 'before')
         .rebatch(rebatch_size)
         .call(get_batch_len, B(), 'after')
         ) << dataset

    p.run(batch_size=batch_size, n_epochs=1, notifier=True)

    check_batch_lengths(batch_lengths['before'], batch_size)
    check_batch_lengths(batch_lengths['after'], rebatch_size)


@pytest.mark.parametrize('merge_factor', [1, 3])
def test_merge(merge_factor):
    """ merge `merge_factor` instances of a batch and check result's shape"""
    data = np.vstack([np.array([i, i]) for i in range(DATASET_SIZE)])
    data = (data,)
    b = MyBatch(index=np.arange(DATASET_SIZE), preloaded=data)
    merged, rest = MyBatch.merge([b] * merge_factor)

    assert merged.dummy.shape[0] == b.dummy.shape[0] * merge_factor
    assert merged.dummy.shape[1] == b.dummy.shape[1]
    assert rest is None
