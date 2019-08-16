""" Test rebatch action """
import numpy as np

import pytest

from batchflow import Dataset, Pipeline, Batch


class MyBatch(Batch):
    components = ('dummy', )


PARAMETERS = [(20, 30), (1, 10), (10, 60), (1, 60)]  # (40, 15)]
PARAMETERS = PARAMETERS + [(b, a) for a, b in PARAMETERS]


@pytest.mark.parametrize('batch_size, rebatch_size', PARAMETERS)
def test_rebatch(batch_size, rebatch_size):
    dataset_size = 60
    batch_shape = (dataset_size, 10)
    data = np.random.random(batch_shape)

    dataset = Dataset(index=dataset_size,
                      batch_class=MyBatch,
                      preloaded=data)

    p = Pipeline().rebatch(rebatch_size) << dataset

    p.run(batch_size=batch_size, n_epochs=1, bar=True)


if __name__ == '__main__':
    test_rebatch(15, 40)
