""" Contains the base class for open datasets """
import numpy as np

from .. import Dataset, DatasetIndex
from .. import ImagesBatch


class Openset(Dataset):
    """ The base class for open datasets """
    def __init__(self, index=None, batch_class=None, path=None, preloaded=None, **kwargs):
        self._train_index, self._test_index = None, None
        if index is None:
            preloaded, index, self._train_index, self._test_index = self.download(path=path)
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, **kwargs)

        if self._train_index and self._test_index:
            self.train = type(self).from_dataset(self, self._train_index, batch_class=batch_class, **kwargs)
            self.test = type(self).from_dataset(self, self._test_index, batch_class=batch_class, **kwargs)

    def download(self, path):
        """ Download a dataset from the source web-site.

        Returns
        -------
        tuple of np.arrays or None
            Preloaded data components, i.e. images and labels. Return None in case using FilesIndex.
        DatasetIndex or FilesIndex
            An index for the dataset.
        DatasetIndex or FilesIndex or None
            An index for the train part of the dataset. Return None if there is no train/test split.
        DatasetIndex or FilesIndex or None
            An index for the test part of the dataset. Return None if there is no train/test split.
        """
        _ = path
        return None, None, None, None

    def _infer_train_test_index(self, train_len, test_len):
        total_len = train_len + test_len
        index = DatasetIndex(list(range(total_len)))
        train_index = DatasetIndex(list(range(train_len)))
        test_index = DatasetIndex(list(range(train_len, total_len)))
        return index, train_index, test_index

    def create_array(self, images):
        """ Create numpy array of objects. """
        array = np.empty(len(images), dtype=object)
        for i, image in enumerate(images):
            array[i] = image
        return array


class ImagesOpenset(Openset):
    """ The base class for open datasets with images """
    def __init__(self, index=None, batch_class=ImagesBatch, *args, **kwargs):
        super().__init__(index, batch_class, *args, **kwargs)
