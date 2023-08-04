# pylint: disable=missing-docstring, redefined-outer-name

import sys
import pytest

import PIL
import numpy as np

sys.path.insert(0, '../../')
from batchflow.opensets import (PascalSegmentation, PascalClassification,
                                COCOSegmentation, MNIST, CIFAR10, CIFAR100, Imagenette)
from batchflow.opensets.ade import ADESegmentation


@pytest.fixture()
def pascal_segmentation():
    return PascalSegmentation()

@pytest.fixture()
def ade_segmentation():
    return ADESegmentation()

@pytest.fixture()
def pascal_classification():
    return PascalClassification()

@pytest.fixture()
def coco_segmentation():
    return COCOSegmentation()

@pytest.fixture()
def cifar10():
    return CIFAR10()

@pytest.fixture()
def cifar100():
    return CIFAR100()

@pytest.fixture()
def mnist():
    return MNIST()

@pytest.fixture()
def imagenette():
    return Imagenette()


class TestOpensets:
    @pytest.mark.slow
    def test_pascal_segmentation(self, pascal_segmentation):
        batch = pascal_segmentation.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.JpegImagePlugin.JpegImageFile)
        assert isinstance(batch.labels[0], PIL.PngImagePlugin.PngImageFile)

    @pytest.mark.slow
    def test_ade_segmentation(self, ade_segmentation):
        batch = ade_segmentation.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.JpegImagePlugin.JpegImageFile)
        assert isinstance(batch.labels[0], PIL.PngImagePlugin.PngImageFile)

    @pytest.mark.slow
    def test_pascal_classification(self, pascal_classification):
        batch = pascal_classification.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.JpegImagePlugin.JpegImageFile)
        assert isinstance(batch.labels[0], np.ndarray)

    @pytest.mark.slow
    def test_cifar10(self, cifar10):
        batch = cifar10.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.Image.Image)
        assert isinstance(batch.labels[0], np.int64)

    @pytest.mark.slow
    def test_cifar100(self, cifar100):
        batch = cifar100.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.Image.Image)
        assert isinstance(batch.labels[0], np.int64)

    @pytest.mark.slow
    def test_mnist(self, mnist):
        batch = mnist.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.Image.Image)
        assert isinstance(batch.labels[0], np.uint8)

    @pytest.mark.slow
    def test_imagenette(self, imagenette):
        batch = imagenette.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.JpegImagePlugin.JpegImageFile)
        assert isinstance(batch.labels[0], np.int64)

    @pytest.mark.skip
    @pytest.mark.slow
    def test_coco_segmentation(self, coco_segmentation):
        batch = coco_segmentation.next_batch(batch_size=10)
        assert isinstance(batch.images[0], PIL.JpegImagePlugin.JpegImageFile)
        assert isinstance(batch.labels[0], PIL.PngImagePlugin.PngImageFile)