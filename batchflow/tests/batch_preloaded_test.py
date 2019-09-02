# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
import pandas as pd

from batchflow import Pipeline, Dataset, Batch, B, L, V


DATASET_SIZE = 100
IMAGE_SHAPE = 10, 10


def get_batch(data, pipeline, batch_class=Batch):
    dataset = Dataset(DATASET_SIZE, preloaded=data, batch_class=batch_class)

    template_pipeline = (
        Pipeline()
        .update(V('dummy'), B.data)     # touch batch data to fire preloading
    )

    if isinstance(pipeline, Pipeline):
        template_pipeline = pipeline + template_pipeline

    source = (dataset >> template_pipeline) if pipeline is not False else dataset

    #skip 2 batches
    source.next_batch(10)
    source.next_batch(10)
    batch = source.next_batch(10)

    return batch


@pytest.mark.parametrize('pipeline', [False, True])
class TestBatchPreloadedNoComponents:
    def test_array(self, pipeline):
        data = np.arange(DATASET_SIZE) + 100

        batch = get_batch(data, pipeline)

        assert (batch.data == np.arange(120, 130)).all()

    def test_tuple(self, pipeline):
        data = np.arange(DATASET_SIZE) + 100, np.arange(DATASET_SIZE) + 1000

        batch = get_batch(data, pipeline)

        assert (batch.data[0] == np.arange(120, 130)).all()
        assert (batch.data[1] == np.arange(1020, 1030)).all()

    def test_dict(self, pipeline):
        data = dict(comp1=np.arange(DATASET_SIZE) + 100, comp2=np.arange(DATASET_SIZE) + 1000)

        batch = get_batch(data, pipeline)

        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()

    def test_df(self, pipeline):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'comp1': comp1, 'comp2': comp2}, index=index)

        batch = get_batch(data, pipeline)

        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()


class MyBatch(Batch):
    components = "images", "nodata1", "labels", "nodata2"


@pytest.mark.parametrize('pipeline', [False, True])
class TestBatchPreloadedComponents:
    def test_tuple(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        # we cannot omit data for nodata1 component, so we pass None
        data = images, None, labels

        batch = get_batch(data, pipeline, MyBatch)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None

    def test_dict(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)

        batch = get_batch(data, pipeline, MyBatch)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None

    def test_df(self, pipeline):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'images': comp1, 'labels': comp2, 'nodata1': None}, index=index)

        batch = get_batch(data, pipeline, MyBatch)

        assert (batch.images == np.arange(120, 130)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        # since nodata1 is a pd.Series of None
        assert (batch.nodata1.to_numpy() == batch.array_of_nones).all()
        assert batch.nodata2 is None


class TestBatchPreloadedAddComponents:
    def test_none(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        pipeline = Pipeline().add_components('new')
        batch = get_batch(data, pipeline, MyBatch)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
        assert batch.new is None

    def test_array(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        pipeline = Pipeline().add_components('new', L(np.arange)(B.size) + B.indices)
        batch = get_batch(data, pipeline, MyBatch)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
        assert (batch.new == np.arange(20, 40, 2)).all()


@pytest.mark.parametrize('pipeline', [False, True])
class TestFromBatchPreloadedComponents:
    def test_tuple(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        # we cannot omit data for nodata1 component, so we pass None
        data = images, None, labels + 1000

        batch = get_batch(data, pipeline, MyBatch)
        batch = batch.from_batch(batch)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
