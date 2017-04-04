""" Dataset """

import numpy as np
from .base import Baseset
from .dsindex import DatasetIndex
from .preprocess import Preprocessing


class Dataset(Baseset):
    """ Dataset """
    def __init__(self, index, batch_class=None, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.batch_class = batch_class


    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None):
        """ Create Dataset from another dataset with new index
            (usually subset of the source dataset index)
        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and cls._is_same_index(index, dataset.index):
            return dataset
        else:
            bcl = batch_class if batch_class is not None else dataset.batch_class
            return cls(index, batch_class=bcl)

    @staticmethod
    def build_index(index):
        """ Create index """
        if isinstance(index, DatasetIndex):
            return index
        else:
            return DatasetIndex(index)

    @staticmethod
    def _is_same_index(index1, index2):
        return isinstance(index1, type(index2)) and \
                          index1.index.shape == index2.index.shape and \
                          np.all(index1.index == index2.index)


    def create_subset(self, index):
        """ Create a dataset based on the given subset of indices """
        return Dataset.from_dataset(self, index)


    def create_batch(self, batch_indices, pos=False, *args, **kwargs):
        """ Create a batch from given indices
            if pos is False then batch_indices contains the value of indices
            which should be included in the batch
            otherwise batch_indices contains positions in the index
        """
        batch_ix = self.index.create_batch(batch_indices, pos, *args, **kwargs)
        return self.batch_class(batch_ix, *args, **kwargs)


    def workflow(self):
        """ Start a data processing workflow """
        return Preprocessing(self)
