""" Dataset """

import numpy as np
from .base import Baseset
from .dsindex import DatasetIndex
from .pipeline import Pipeline


class Dataset(Baseset):
    """ Dataset

    Attributes
    ----------
    index
    indices
    is_split
    """
    def __init__(self, index, batch_class=None, preloaded=None, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.batch_class = batch_class
        self.preloaded = preloaded

    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None):
        """ Create Dataset from another dataset with new index
        (usually a subset of the source dataset index)
        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and cls._is_same_index(index, dataset.index):
            return dataset
        bcl = batch_class if batch_class is not None else dataset.batch_class
        return cls(index, batch_class=bcl, preloaded=dataset.preloaded)

    @staticmethod
    def build_index(index):
        """ Create an index """
        if isinstance(index, DatasetIndex):
            return index
        return DatasetIndex(index)

    @staticmethod
    def _is_same_index(index1, index2):
        return (isinstance(index1, type(index2)) or isinstance(index2, type(index1))) and \
               index1.indices.shape == index2.indices.shape and \
               np.all(index1.indices == index2.indices)

    def create_subset(self, index):
        """ Create a dataset based on the given subset of indices """
        return type(self).from_dataset(self, index)

    def create_batch(self, batch_indices, pos=False, *args, **kwargs):
        """ Create a batch from given indices.

            if `pos` is `False`, then `batch_indices` should contain the indices
            that should be included in the batch
            otherwise `batch_indices` should contain their positions in the current index
        """
        batch_ix = self.index.create_batch(batch_indices, pos, *args, **kwargs)
        return self.batch_class(batch_ix, preloaded=self.preloaded, **kwargs)

    def pipeline(self, config=None):
        """ Start a new data processing workflow """
        return Pipeline(self, config=config)

    @property
    def p(self):
        """:class:`dataset.Pipeline` : a short alias for `pipeline()` """
        return self.pipeline()

    def __rshift__(self, other):
        if not isinstance(other, Pipeline):
            raise TypeError("Pipeline is expected, but got %s. Use as dataset >> pipeline" % type(other))
        new_p = other.from_pipeline(other)
        new_p.dataset = self
        return new_p
