""" Contains the base class for open datasets """
import numpy as np

from .. import Dataset
from .. import ImagesBatch


class Openset(Dataset):
    """ The base class for open datasets """
    def __init__(self, index=None, batch_class=None, train_test=False, path=None, preloaded=None, **kwargs):
        self.masks_directory = None
        self.train_test = train_test
        self._train_index, self._test_index = None, None
        if index is None:
            preloaded, index = self.download(path=path)

            if preloaded is not None:
                if train_test: # temporary solution for MNIST, CIFAR, ...
                    preloaded = tuple(np.concatenate(i) for i in np.array(preloaded).T)

        super().__init__(index, batch_class=batch_class, preloaded=preloaded, **kwargs)
        if self._train_index and self._test_index:
            self.train = type(self).from_dataset(self, self._train_index, batch_class=batch_class,
                                                 masks_directory=self.masks_directory, **kwargs)
            self.test = type(self).from_dataset(self, self._test_index, batch_class=batch_class,
                                                masks_directory=self.masks_directory, **kwargs)

    @staticmethod
    def uild_index(index):
        """ Create an index """
        if index is not None:
            return super().build_index(index)
        return None

    def download(self, path):
        """ Download a dataset from the source web-site """
        _ = path
        return None


class ImagesOpenset(Openset):
    """ The base class for open datasets with images """
    def __init__(self, index=None, batch_class=ImagesBatch, *args, **kwargs):
        super().__init__(index, batch_class, *args, **kwargs)
