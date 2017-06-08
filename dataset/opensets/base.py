""" Contains MNIST dataset """

import numpy as np

from .. import Dataset, DatasetIndex, ImagesBatch


class Openset:
    """ The base class for open datasets """
    def __init__(self, batch_class):
        self.batch_class = batch_class
        self.train = None
        self.test = None
        self._data = self.download()
        self.create_datasets(preloaded=self._data is not None)

    def download(self):
        """ Download a dataset from the source web-site """
        return None

    def create_datasets(self, preloaded=False):
        """ Create train, test and other sub-datasets """
        raise NotImplementedError()


class ImagesOpenset(Openset):
    """ The base class for open datasets with images """
    def __init__(self, batch_class=ImagesBatch):
        super().__init__(batch_class)

    def download(self):
        """ Download a dataset from the source web-site """
        return None, None

    def create_datasets(self, preloaded=False):
        """ Create train and test datasets """
        if isinstance(self._data, tuple) and len(self._data) == 2:
            train_data, test_data = self._data   # pylint:disable=unpacking-non-sequence
            train_index = DatasetIndex(np.arange(len(train_data[0])))
            train_preloaded = train_data if preloaded else None
            self.train = Dataset(train_index, self.batch_class, preloaded=train_preloaded)

            test_index = DatasetIndex(np.arange(len(test_data[0])))
            test_preloaded = test_data if preloaded else None
            self.test = Dataset(test_index, self.batch_class, preloaded=test_preloaded)
