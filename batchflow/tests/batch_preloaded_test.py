# pylint: disable=missing-docstring, redefined-outer-name

import pytest
import numpy as np
import pandas as pd

from batchflow import Dataset, Batch, B, L


DATASET_SIZE = 100
IMAGE_SHAPE = 10, 10


class TestBatchPreloadedNoComponents:
    def test_array(self):
        data = np.arange(DATASET_SIZE) + 100
        dataset = Dataset(DATASET_SIZE, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.data == np.arange(120, 130)).all()

    def test_tuple(self):
        data = np.arange(DATASET_SIZE) + 100, np.arange(DATASET_SIZE) + 1000
        dataset = Dataset(DATASET_SIZE, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.data[0] == np.arange(120, 130)).all()
        assert (batch.data[1] == np.arange(1020, 1030)).all()

    def test_dict(self):
        data = dict(comp1=np.arange(DATASET_SIZE) + 100, comp2=np.arange(DATASET_SIZE) + 1000)
        dataset = Dataset(DATASET_SIZE, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()

    def test_df(self):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'comp1': comp1, 'comp2': comp2}, index=index)
        dataset = Dataset(index, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()


class MyBatch(Batch):
    components = "images", "nodata1", "labels", "nodata2"


class TestBatchPreloadedComponents:
    def test_tuple(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = images, None, labels
        dataset = Dataset(DATASET_SIZE, batch_class=MyBatch, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert (batch.nodata1 is None)
        assert (batch.nodata2 is None)

    def test_dict(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)
        dataset = Dataset(DATASET_SIZE, batch_class=MyBatch, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)
        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert (batch.nodata1 is None)
        assert (batch.nodata2 is None)

    def test_df(self):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'images': comp1, 'labels': comp2}, index=index)
        dataset = Dataset(index, batch_class=MyBatch, preloaded=data)

        #skip 2 batches
        dataset.next_batch(10)
        dataset.next_batch(10)

        batch = dataset.next_batch(10)

        assert (batch.images == np.arange(120, 130)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert (batch.nodata1 is None)
        assert (batch.nodata2 is None)


class TestBatchPreloadedAddComponents:
    def test_none(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)
        dataset = Dataset(DATASET_SIZE, batch_class=MyBatch, preloaded=data)

        pipeline = dataset.p.add_components('new')

        #skip 2 batches
        pipeline.next_batch(10)
        pipeline.next_batch(10)

        batch = pipeline.next_batch(10)
        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert (batch.nodata1 is None)
        assert (batch.nodata2 is None)
        assert (batch.new is None)

    def test_array(self):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)
        dataset = Dataset(DATASET_SIZE, batch_class=MyBatch, preloaded=data)

        pipeline = dataset.p.add_components('new', L(np.arange)(B.size) + B.indices)

        #skip 2 batches
        pipeline.next_batch(10)
        pipeline.next_batch(10)

        batch = pipeline.next_batch(10)
        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert (batch.nodata1 is None)
        assert (batch.nodata2 is None)
        assert (batch.new == np.arange(20, 40, 2)).all()
