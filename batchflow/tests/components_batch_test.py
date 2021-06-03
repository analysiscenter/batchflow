# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
import pandas as pd

from batchflow import Pipeline, Dataset, Batch, B, F, V


DATASET_SIZE = 100
IMAGE_SHAPE = 10, 10


def get_batch(data, pipeline, index=DATASET_SIZE, batch_class=Batch, skip=2, dst=False):
    """
    Parameters
    ----------
    data
        data to use
    pipeline : bool or Pipeline
        whether to get a batch from a dataset or a pipeline

    index : DatasetIndex

    batch_class : type

    skip : int
        how many batches to skip

    dst : bool or list of str
        preload data when False or load to components given
    """

    if dst is False:
        dataset = Dataset(index, preloaded=data, batch_class=batch_class)
    else:
        dataset = Dataset(index, batch_class=batch_class)

    template_pipeline = (
        Pipeline()
        .init_variable('dummy')
        .update(V('dummy'), B.data)     # touch batch data to fire preloading
    )

    if isinstance(pipeline, Pipeline):
        template_pipeline = pipeline + template_pipeline

    source = (dataset >> template_pipeline) if pipeline is not False else dataset

    #skip K batches
    for _ in range(skip + 1):
        batch = source.next_batch(10)

    if dst is not False:
        batch = batch.load(src=data, dst=dst)

    return batch


# preload when dst=False, otherwise load to dst
@pytest.mark.parametrize('dst', [False, None])
@pytest.mark.parametrize('pipeline', [False, True])
class TestNoComponents:
    def test_array(self, pipeline, dst):
        data = np.arange(DATASET_SIZE) + 100

        batch = get_batch(data, pipeline, skip=2, dst=dst)

        assert (batch.data == np.arange(120, 130)).all()

    def test_tuple(self, pipeline, dst):
        data = np.arange(DATASET_SIZE) + 100, np.arange(DATASET_SIZE) + 1000

        batch = get_batch(data, pipeline, skip=2, dst=dst)

        assert (batch.data[0] == np.arange(120, 130)).all()
        assert (batch.data[1] == np.arange(1020, 1030)).all()

    def test_dict(self, pipeline, dst):
        data = dict(comp1=np.arange(DATASET_SIZE) + 100, comp2=np.arange(DATASET_SIZE) + 1000)

        batch = get_batch(data, pipeline, skip=2, dst=dst)

        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()

    def test_df(self, pipeline, dst):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'comp1': comp1, 'comp2': comp2}, index=index)

        batch = get_batch(data, pipeline, index=index, skip=2, dst=dst)

        assert (batch.data['comp1'] == np.arange(120, 130)).all()
        assert (batch.data['comp2'] == np.arange(1020, 1030)).all()


class MyBatch1(Batch):
    components = ("images",)


@pytest.mark.parametrize('pipeline', [False, True])
class TestOneComponent:
    def test_array_fail(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        data = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        dst = False

        # the pipeline accesses batch data and fails when preloading the dataset
        if pipeline is True:
            with pytest.raises(AttributeError) as execinfo:
                batch = get_batch(data, pipeline, batch_class=MyBatch1, skip=2, dst=dst)
            assert "data not found in class" in str(execinfo.value)
            return

        batch = get_batch(data, pipeline=False, batch_class=MyBatch1, skip=2, dst=dst)
        # Batch tries to read `data.images` and fails at preload
        with pytest.raises(AttributeError) as execinfo:
            _ = batch.images
        assert "data not found in class" in str(execinfo.value)

    def test_array(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        data = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        dst = 'images'

        batch = get_batch(data, pipeline, batch_class=MyBatch1, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()

    @pytest.mark.parametrize('dst', [False, ('images',)])
    def test_tuple(self, pipeline, dst):
        labels = np.arange(DATASET_SIZE)
        data = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = (data,)

        batch = get_batch(data, pipeline, batch_class=MyBatch1, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()

    @pytest.mark.parametrize('dst', [False, ('images',)])
    def test_dict(self, pipeline, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)

        batch = get_batch(data, pipeline, batch_class=MyBatch1, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()

    @pytest.mark.parametrize('dst', [False, ('images',)])
    def test_dict_with_index(self, pipeline, dst):
        index = (np.arange(100)+ 1000).astype('str')
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=dict(zip(index, images)), labels=labels)

        batch = get_batch(data, pipeline, index=index, batch_class=MyBatch1, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()

    @pytest.mark.parametrize('dst', [False, ('images',)])
    def test_df(self, pipeline, dst):
        index = (np.arange(100)+ 1000).astype('str')
        labels = np.arange(DATASET_SIZE)
        images = np.arange(DATASET_SIZE)
        data = dict(images=images, labels=labels)
        data = pd.DataFrame(data, index=index)

        batch = get_batch(data, pipeline, index=index, batch_class=MyBatch1, skip=2, dst=dst)

        assert (batch.images == np.arange(20, 30)).all()


class MyBatch4(Batch):
    components = "images", "nodata1", "labels", "nodata2"


@pytest.mark.parametrize('pipeline', [False, True])
class TesManyComponents:
    def test_tuple(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        # we cannot omit data for nodata1 component, so we pass None
        data = images, None, labels

        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None

    def test_dict(self, pipeline):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels)

        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(20, 30)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None

    def test_df(self, pipeline):
        index = (np.arange(100)+ 1000).astype('str')
        comp1 = np.arange(DATASET_SIZE) + 100
        comp2 = np.arange(DATASET_SIZE) + 1000
        data = pd.DataFrame({'images': comp1, 'labels': comp2, 'nodata1': None}, index=index)

        batch = get_batch(data, pipeline, index=index, batch_class=MyBatch4, skip=2)

        assert (batch.images == np.arange(120, 130)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        # since nodata1 is a pd.Series of None
        assert (batch.nodata1 == batch.array_of_nones).all()
        assert batch.nodata2 is None


class TestAddComponents:
    @pytest.mark.parametrize('dst', [False, ('images', 'labels')])
    def test_none(self, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        pipeline = Pipeline().add_components('new')
        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
        assert batch.new is None

    @pytest.mark.parametrize('dst', [False, ('images', 'labels')])
    def test_array(self, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        pipeline = Pipeline().add_components('new', F(np.arange)(B.size) + B.indices)
        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2, dst=dst)

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
        assert (batch.new == np.arange(20, 40, 2)).all()


    @pytest.mark.parametrize('dst', [False, ('images', 'labels')])
    def test_load_from_array(self, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        pipeline = Pipeline().add_components('new')
        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2, dst=dst)
        batch = batch.load(src=np.arange(DATASET_SIZE) + 5000, dst='new')

        assert (batch.images[:, 0, 0] == np.arange(20, 30)).all()
        assert (batch.labels == np.arange(1020, 1030)).all()
        assert batch.nodata1 is None
        assert batch.nodata2 is None
        assert (batch.new == np.arange(5020, 5030)).all()


@pytest.mark.parametrize('pipeline', [False, True])
class TestItems:
    @pytest.mark.parametrize('dst', [False, None])
    def test_tuple(self, pipeline, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = (images,)

        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2, dst=dst)

        item = batch[25]

        assert (item.images == 25).all()
        assert item.labels is None

    @pytest.mark.parametrize('dst', [False, None])
    def test_dict(self, pipeline, dst):
        labels = np.arange(DATASET_SIZE)
        images = np.ones((DATASET_SIZE,) + IMAGE_SHAPE) * labels.reshape(-1, 1, 1)
        data = dict(images=images, labels=labels+1000)

        batch = get_batch(data, pipeline, batch_class=MyBatch4, skip=2, dst=dst)

        item = batch[25]

        assert (item.images == 25).all()
        assert (item.labels == 1025).all()
