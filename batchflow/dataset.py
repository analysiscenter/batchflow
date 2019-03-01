""" Dataset """
import copy
import numpy as np
from .base import Baseset
from .batch import Batch
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
    def __init__(self, index, batch_class=Batch, preloaded=None, *args, **kwargs):
        if batch_class is not Batch or not issubclass(batch_class, Batch):
            raise TypeError("batch_class should be a subclass of Batch", batch_class)

        super().__init__(index, *args, **kwargs)
        self.batch_class = batch_class
        self.preloaded = preloaded

    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None, copy=False):
        """ Create Dataset from another dataset with new index
        (usually a subset of the source dataset index)
        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and cls._is_same_index(index, dataset.index):
            if not copy:
                return dataset
        bcl = batch_class if batch_class is not None else dataset.batch_class
        return cls(index, batch_class=bcl, preloaded=dataset.preloaded)

    def __copy__(self):
        return self.from_dataset(self, self.index, copy=True)

    def __getattr__(self, name):
        if name[:2] == 'cv' and name[2:].isdigit():
            raise AttributeError("To access cross-validation call cv_split() first.")

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
        return type(self).from_dataset(self, self.index.create_subset(index))

    def create_batch(self, batch_indices, pos=False, *args, **kwargs):
        """ Create a batch from given indices.

            if `pos` is `False`, then `batch_indices` should contain the indices
            that should be included in the batch
            otherwise `batch_indices` should contain their positions in the current index
        """
        if not isinstance(batch_indices, DatasetIndex):
            batch_indices = self.index.create_batch(batch_indices, pos, *args, **kwargs)
        return self.batch_class(batch_indices, preloaded=self.preloaded, **kwargs)

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
        return other << self

    def cv_split(self, method='kfold', n_splits=5, shuffle=False):
        """ Create datasets for cross-validation

        Datasets are available as `cv0`, `cv1` and so on.
        They already split into train and test parts.

        Parameters
        ----------
        method : {'kfold'}
            a plitting method (only `kfold` is supported)

        n_splits : int
            a number of folds

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            specifies the order of items, could be:

            - bool - if `False`, items go sequentionally, one after another as they appear in the index.
                if `True`, items are shuffled randomly before each epoch.

            - int - a seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance.

            - callable - a function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

        Examples
        --------

        ::

            dataset = Dataset(10)
            dataset.cv_split(n_splits=3)
            print(dataset.cv0.test.indices) # [0, 1, 2, 3]
            print(dataset.cv1.test.indices) # [4, 5, 6]
            print(dataset.cv2.test.indices) # [7, 8, 9]
        """
        order = self.index.shuffle(shuffle)

        if method == 'kfold':
            splits = self._split_kfold(n_splits, order)
        else:
            raise ValueError("Unknown split method:", method)

        for i in range(n_splits):
            test_indices = splits[i]
            train_splits = list(set(range(n_splits)) - {i})
            train_indices = np.concatenate(np.asarray(splits)[train_splits])

            setattr(self, 'cv'+str(i), copy.copy(self))
            cv_dataset = getattr(self, 'cv'+str(i))
            cv_dataset.train = self.create_subset(train_indices)
            cv_dataset.test = self.create_subset(test_indices)


    def _split_kfold(self, n_splits, order):
        split_sizes = np.full(n_splits, len(order) // n_splits, dtype=np.int)
        split_sizes[:len(order) % n_splits] += 1
        current = 0
        splits = []
        for split_size in split_sizes:
            start, stop = current, current + split_size
            splits.append(self.indices[order[start:stop]])
            current = stop
        return splits


