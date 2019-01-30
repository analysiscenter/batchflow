import sys
import pytest
import numpy as np


from batchflow import Dataset

from batchflow import Baseset
from batchflow import Batch
from batchflow import DatasetIndex
from batchflow import Pipeline


@pytest.fixture
def dataset():
    index = DatasetIndex(np.arange(100))
    batch_class = Batch
    return Dataset(index, batch_class)


class TestDataset:
    def test_from_dataset(self, dataset):
        new_index = DatasetIndex(np.arange(25))
        new_ds = Dataset.from_dataset(dataset, new_index)
        assert isinstance(new_ds, Dataset)
        assert (new_ds.index.index == new_index.index).all()

    def test_build_index(self, dataset):
        new_index = Dataset.build_index(np.arange(25))
        assert isinstance(new_index, DatasetIndex)
        assert len(new_index.index) > 0

    def test_create_subset(self, dataset):
        new_index = DatasetIndex(np.arange(25))
        new_ds = dataset.create_subset(new_index)
        assert isinstance(new_ds, Dataset)
        assert np.isin(new_ds.index.index, dataset.index.index).all()

    def test_create_batch(self, dataset):
        target_index = DatasetIndex(np.arange(5))
        new_batch = dataset.create_batch(target_index)
        assert isinstance(new_batch, dataset.batch_class)

    def test_split(self, dataset):
        assert dataset.train is None
        dataset.split()
        assert dataset.train is not None
        assert dataset.test is not None