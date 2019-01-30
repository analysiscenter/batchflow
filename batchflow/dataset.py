""" Dataset """

import numpy as np
from .base import Baseset
from .batch import Batch
from .dsindex import DatasetIndex
from .dsindex import FilesIndex
from .pipeline import Pipeline


class Dataset(Baseset):
    """
    The Dataset holds an index of all data items
    (e.g. customers, transactions, etc)
    and a specific action class to process a small subset of data (batch).
    ...

    Attributes
    ----------
    index : DatasetIndex or FilesIndex

    indices : class:`numpy.ndarray`
        an array with the indices

    is_split: bool
        True if dataset has been split into train / test / validation subsets

    Methods
    -------

    """
    def __init__(self, index, batch_class=Batch, preloaded=None, *args, **kwargs):
        """ Create Dataset

            Parameters
            ----------
            index : DatasetIndex or FilesIndex
                Stores an index for a dataset

            batch_class : Batch or inherited-from-Batch
                Batch class holds the data and contains processing functions

            preloaded : data-type
                For smaller dataset it might be convenient to preload all data at once
                As a result, all created batches will contain a portion of some_data.

        """
        super().__init__(index, *args, **kwargs)
        self.batch_class = batch_class
        self.preloaded = preloaded

    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None):
        """ Create Dataset from another dataset with a new index
            (usually a subset of the source dataset index)

            Parameters
            ----------
            dataset : Dataset
                Source dataset

            index : DatasetIndex or FilesIndex

            batch_class : Batch or inherited-from-Batch

            Returns
            -------
            cls : Dataset
                The new object of class Dataset

        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and cls._is_same_index(index, dataset.index):
            return dataset
        bcl = batch_class if batch_class is not None else dataset.batch_class
        return cls(index, batch_class=bcl, preloaded=dataset.preloaded)

    @staticmethod
    def build_index(index):
        """ Create a DatasetIndex object from array-like data

            Parameters
            ----------
            index : array-like

            Returns
            -------
            index : DatasetIndex
                DatasetIndex class object which was created from
        """
        if isinstance(index, DatasetIndex):
            return index
        return DatasetIndex(index)

    @staticmethod
    def _is_same_index(index1, index2):
        """ Check if index1 and index2 are equals

            Parameters
            ----------
            index1 : array-like

            index2 : array-like

            Returns
            -------
            The result of two indices comparison 

        """
        return (isinstance(index1, type(index2)) or isinstance(index2, type(index1))) and \
               index1.indices.shape == index2.indices.shape and \
               np.all(index1.indices == index2.indices)

    def create_subset(self, index):
        """ Create a dataset based on the given subset of indices

            Parameters
            ----------
            index : DatasetIndex or Files
        """
        return type(self).from_dataset(self, index)

    def create_batch(self, batch_indices, pos=False, *args, **kwargs):
        """ Create a batch from given indices.

            Parameters
            ----------
            batch_indices : DatasetIndex or FilesIndex

            pos : bool

            Returns
            -------
            batch : Batch or inherited-from-Batch

            Notes
            -----
            if `pos` is `False`, then `batch_indices` should contain the indices
            that should be included in the batch
            otherwise `batch_indices` should contain their positions in the current index
        """
        if not isinstance(batch_indices, DatasetIndex):
            batch_indices = self.index.create_batch(batch_indices, pos, *args, **kwargs)
        return self.batch_class(batch_indices, preloaded=self.preloaded, **kwargs)

    def pipeline(self, config=None):
        """ Start a new data processing workflow

            Parameters
            ----------
            config : Config or dict

            Returns
            -------
            pipeline : Pipeline
                The new Pipeline class object
        """
        return Pipeline(self, config=config)

    @property
    def p(self):
        """A short alias for `pipeline()` """
        return self.pipeline()

    def __rshift__(self, other):
        """
            Parameters
            ----------
            other : Pipeline

            Returns
            -------

            Raises
            ------
            TypeError
                If the type of other is not a Pipeline
        """
        if not isinstance(other, Pipeline):
            raise TypeError("Pipeline is expected, but got %s. Use as dataset >> pipeline" % type(other))
        return other << self
