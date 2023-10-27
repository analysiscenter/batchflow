# pylint: disable=missing-docstring, redefined-outer-name

import sys
import pytest

from PIL.Image import Image
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
import numpy as np

sys.path.insert(0, '../../')
from batchflow.opensets import (PascalSegmentation, PascalClassification, ADESegmentation,
                                COCOSegmentation, MNIST, CIFAR10, CIFAR100, Imagenette)
from batchflow import FilesIndex


class TestOpensets:

    parameters = [
        (PascalSegmentation, JpegImageFile, PngImageFile),
        (PascalClassification, JpegImageFile, np.ndarray),
        (ADESegmentation, JpegImageFile, PngImageFile),
        (CIFAR10, Image, np.int64),
        (CIFAR100, Image, np.int64),
        (MNIST, Image, np.uint8),
        (Imagenette, JpegImageFile, np.int64),
        (COCOSegmentation, JpegImageFile, PngImageFile)
    ]

    @pytest.mark.parametrize(("openset", "image_type", "label_type"), parameters)
    @pytest.mark.slow
    def test_openset(self, openset, image_type, label_type):
        if issubclass(openset, COCOSegmentation):
            batch = openset().next_batch(batch_size=10)
            batch = batch.load(fmt='image', dst='images')
            batch = batch.load(
                src=FilesIndex(path=openset().path_train_masks + '/*', no_ext=True),
                fmt='image',
                dst='labels'
            )
        else:
            batch = openset().next_batch(batch_size=10)

        assert isinstance(batch.images[0], image_type)
        assert isinstance(batch.labels[0], label_type)
