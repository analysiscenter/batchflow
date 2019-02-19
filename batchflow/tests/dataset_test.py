# pylint: disable=missing-docstring, redefined-outer-name

import pytest
import numpy as np

from batchflow import Dataset, Batch, ImagesBatch, DatasetIndex, Pipeline


@pytest.fixture
def dataset():
    index = DatasetIndex(np.arange(100))
    return Dataset(index, Batch)


class TestDataset:
    def test_from_dataset(self, dataset):
        new_index = DatasetIndex(np.arange(25))
        new_ds = Dataset.from_dataset(dataset, new_index)
        assert isinstance(new_ds, Dataset)
        assert new_ds.batch_class == dataset.batch_class
        assert (new_ds.index.index == new_index.index).all()

    def test_from_dataset_new_batch_class(self, dataset):
        new_index = DatasetIndex(np.arange(25))
        new_ds = Dataset.from_dataset(dataset, new_index, batch_class=ImagesBatch)
        assert isinstance(new_ds, Dataset)
        assert new_ds.batch_class == ImagesBatch

    def test_build_index(self):
        new_index = Dataset.build_index(np.arange(25))
        assert isinstance(new_index, DatasetIndex)

    def test_create_subset(self, dataset):
        new_index = DatasetIndex(np.arange(25))
        new_ds = dataset.create_subset(new_index)
        assert isinstance(new_ds, dataset.__class__)
        assert np.isin(new_ds.indices, dataset.indices).all()

    def test_create_subset_wrong_index(self, dataset):
        """
        This method checks for correct execution create_subset method in case of
        there should be exception during calling this method with index which is not in
        source dataset index range.
        Now we raises and handle AssertError
        """
        wrong_index = DatasetIndex(np.arange(200, 225))
        with pytest.raises(AssertionError) as excinfo:
            new_ds = dataset.create_subset(wrong_index)
            assert np.isin(new_ds.indices, dataset.indices).all()

    def test_create_batch(self, dataset):
        target_index = DatasetIndex(np.arange(5))
        new_batch = dataset.create_batch(target_index)
        assert isinstance(new_batch, dataset.batch_class)
        assert len(new_batch.indices) == len(target_index.indices)

    def test_pipeline(self, dataset):
        pipeline_config = {}
        new_pipeline = dataset.pipeline(pipeline_config)
        assert isinstance(new_pipeline, Pipeline)

    def test_rshift(self, dataset):
        pipeline_config = {}
        new_pipeline = dataset.pipeline(pipeline_config)
        train_pipeline = (new_pipeline << dataset)
        assert isinstance(train_pipeline, Pipeline)

    def test_split(self, dataset):
        assert dataset.train is None
        dataset.split()
        assert dataset.train is not None
        assert dataset.test is not None
