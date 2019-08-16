""" Test rebatch action """
import numpy as np

import pytest

from batchflow import Dataset, Pipeline, Batch


class MyBatch(Batch):
    components = ('dummy', )


DATASET_SIZE = 60
PARAMETERS = [(20, 30), (1, 10), (10, 60), (1, 60), (15, 40), (13, 17)]
PARAMETERS = PARAMETERS + [(b, a) for a, b in PARAMETERS]


def check_batch_lengths(batch_lengths, batch_size):
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

    data = np.vstack([np.array([i, i]) for i in range(DATASET_SIZE)])
    dataset = Dataset(index=DATASET_SIZE,
                      batch_class=MyBatch,
                      preloaded=data)

    # workaround for pipeline variables getting lost after rebatch
    batch_lengths = {'before': [], 'after': []}

    def get_batch_len(batch, dump):
        batch_lengths[dump].append(batch.size)

    p = (Pipeline()
         .call(get_batch_len, args=('before', ))
         .rebatch(rebatch_size)
         .call(get_batch_len, args=('after', ))
         ) << dataset

    p.run(batch_size=batch_size, n_epochs=1, bar=True)

    check_batch_lengths(batch_lengths['before'], batch_size)
    check_batch_lengths(batch_lengths['after'], rebatch_size)


if __name__ == '__main__':
    test_rebatch(15, 40)
