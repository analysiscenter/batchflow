""" Contains the base class for open datasets """
import numpy as np

from .. import Dataset
from .. import ImagesBatch


class Openset(Dataset):
    """ The base class for open datasets """
    def __init__(self, index=None, batch_class=None, path=None, preloaded=None, **kwargs):
        if preloaded is None and index is None:

            preloaded, index, train_index, test_index = self.download(path=path)
            if preloaded is not None and train_test:
                preloaded = tuple(np.concatenate(i) for i in np.array(preloaded).T)

            if train_index and test_index:
                self.train = type(self)(train_index, batch_class=batch_class, preloaded=preloaded)
                self.test = type(self)(test_index, batch_class=batch_class, preloaded=preloaded)

        super().__init__(index, batch_class=batch_class, preloaded=preloaded, **kwargs)

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
