""" Contains MNIST dataset """

import os
import tempfile
import urllib
import gzip
import numpy as np

from ..jointdataset import JointDataset
from ..dataset import Dataset
from ..dsindex import DatasetIndex
from ..batch import Batch, ImagesBatch, ArrayBatch
from ..decorators import parallel, any_action_failed, action



class MNIST_Batch(Batch):
    """ A batch of MNIST images and labels """
    @property
    def images(self):
        """ Images of digits """
        return self.data[0]

    @property
    def labels(self):
        """ Labels for images """
        return self.data[1]

    @action
    def load(self, src, fmt=None):
        """ Load data from a preloaded dataset """
        self._data = src[0][self.indices], src[1][self.indices]


#
#  _read32, extract_images, extract_labels are taken from tensorflow
#
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _extract_labels(f):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      labels: a 1D uint8 numpy array.
    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


class BaseMNIST:
    """ MNIST dataset """
    TRAIN_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    ALL_URLS = [TRAIN_IMAGES_URL, TRAIN_LABELS_URL, TEST_IMAGES_URL, TEST_LABELS_URL]

    def __init__(self):
        train, test = self.download()
        self._train_images, self._train_labels = train
        self._test_images, self._test_labels = test

    @property
    def _get_from_urls(self):
        """ List of URLs and type of content (0 - images, 1 - labels) """
        return [[self.ALL_URLS[i], i % 2] for i in range(len(self.ALL_URLS))]

    def _gather_data(self, all_res):
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)
        else:
            train = all_res[0], all_res[1]
            test = all_res[2], all_res[3]
        return train, test

    @parallel(init='_get_from_urls', post='_gather_data')
    def download(self, url, content):
        tmpdir = tempfile.gettempdir()
        filename = os.path.basename(url)
        localname = os.path.join(tmpdir, filename)
        if not os.path.isfile(localname):
            urllib.request.urlretrieve(url, localname)
            print("Downloaded", filename)

        with open(localname, 'rb') as f:
            data = _extract_images(f) if content == 0 else _extract_labels(f)
        return data



class MNIST(BaseMNIST):
    """ MNIST as a Dataset """
    def __init__(self, batch_class=None):
        super().__init__()

        batch_class = batch_class if batch_class is not None else MNIST_Batch

        train_index = DatasetIndex(np.arange(len(self._train_images)))
        self.train = Dataset(train_index, batch_class, preloaded=(self._train_images, self._train_labels))

        test_index = DatasetIndex(np.arange(len(self._test_images)))
        self.test = Dataset(test_index, batch_class, preloaded=(self._test_images, self._test_labels))


class JointMNIST(BaseMNIST):
    """ MNIST as a JointDataset of images and labels """
    def __init__(self, batch_class=None):
        super().__init__()

        if batch_class is not None:
            if isinstance(batch_class, Batch):
                batch_class = tuple([batch_class])
            if isinstance(batch_class, tuple):
                if len(batch_class) == 2:
                    images_batch_class, labels_batch_class = batch_class
                elif len(batch_class) == 1:
                    images_batch_class, labels_batch_class = batch_class, batch_class
                elif len(batch_class) == 0:
                    batch_class = None
                else:
                    raise ValueError("batch_class should not have more than 2 items, but it has %s: %s" % (len(batch_class), batch_class))
        if batch_class is None:
            images_batch_class, labels_batch_class = ImagesBatch, ArrayBatch
        self._check_batch_class(images_batch_class)
        self._check_batch_class(labels_batch_class)

        train_index = DatasetIndex(np.arange(len(self._train_images)))
        train_images_ds = Dataset(train_index, batch_class=images_batch_class, preloaded=self._train_images)
        train_labels_ds = Dataset(train_index, batch_class=labels_batch_class, preloaded=self._train_labels)
        self.train = JointDataset(datasets=(train_images_ds, train_labels_ds), align='same')

        test_index = DatasetIndex(np.arange(len(self._test_images)))
        test_images_ds = Dataset(train_index, batch_class=images_batch_class, preloaded=self._test_images)
        test_labels_ds = Dataset(train_index, batch_class=labels_batch_class, preloaded=self._test_labels)
        self.test = JointDataset(datasets=(test_images_ds, test_labels_ds), align='same')

        @staticmethod
        def _check_batch_class(batch_class):
            if isinstance(batch_class, Batch):
                raise TypeError("batch_class should be a subclass of Batch, not %s" % type(batch_class))
