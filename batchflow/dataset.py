""" Dataset """
import copy as cp
import numpy as np

from .base import Baseset
from .batch import Batch
from .dsindex import DatasetIndex
from .named_expr import F
from .pipeline import Pipeline
from .components import create_item_class


class Dataset(Baseset):
    """
    The Dataset holds an index of all data items
    (e.g. customers, transactions, etc)
    and a specific action class to process a small subset of data (batch).

    Attributes
    ----------
    batch_class : Batch

    index : DatasetIndex or FilesIndex

    indices : class:`numpy.ndarray`
        an array with the indices

    p : Pipeline
        Actions which will be applied to this dataset

    preloaded : data-type
        For small dataset it could be convenient to preload data at first

    train : Dataset
        The train part of this dataset. It appears after splitting

    test : Dataset
        The test part of this dataset. It appears after splitting

    validation : Dataset
        The validation part of this dataset. It appears after splitting
    """
    def __init__(self, index, batch_class=Batch, *args, preloaded=None, cast_to_array=True, copy=False, **kwargs):
        """ Create Dataset

            Parameters
            ----------
            index : DatasetIndex or FilesIndex or int
                Stores an index for a dataset

            batch_class : Batch or inherited-from-Batch
                Batch class holds the data and contains processing functions

            preloaded : data-type
                For smaller dataset it might be convenient to preload all data at once.
                As a result, all created batches will contain a portion of preloaded.

            cast_to_array : bool
                whether to cast preloaded data to array when creating components data

            copy : bool
                whether to copy data from `preloaded` when creating a batch to alow for in-place transformations

            **kwargs : dict
                additional dataset attributes or `cv_split` parameters
        """
        if batch_class is not Batch and not issubclass(batch_class, Batch):
            raise TypeError("batch_class should be inherited from Batch", batch_class)

        super().__init__(index, *args)
        self.cast_to_array = cast_to_array
        self.batch_class = batch_class
        self.preloaded = preloaded
        self._data_named = None
        self._attrs = None
        kwargs['_copy'] = kwargs.get('_copy', copy)
        self.n_splits = None

        cv_kwargs = {item: kwargs.pop(item) for item in ['method', 'n_splits', 'shuffle'] if item in kwargs}
        if cv_kwargs.get('n_splits') is not None:
            self.cv_split(**cv_kwargs)

        self.create_attrs(**kwargs)

    def create_attrs(self, **kwargs):
        """ Create attributes from kwargs """
        self._attrs = list(kwargs.keys())
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def get_attrs(self):
        """ Return additional attrs as kwargs """
        if self._attrs is None:
            return {}
        return {attr: getattr(self, attr, None) for attr in self._attrs}

    @property
    def data(self):
        """ Return preloaded data """
        if self.preloaded is None:
            return None
        if self.batch_class.components is not None and self._data_named is None:
            self._data_named = create_item_class(self.batch_class.components, source=self.preloaded,
                                                 cast_to_array=self.cast_to_array)
        if self._data_named is not None:
            return self._data_named
        return self.preloaded

    def __getattr__(self, name):
        if name[:2] == 'cv' and name[2:].isdigit():
            raise AttributeError("To access cross-validation call cv_split() first.")
        if 'batch_class' in dir(self) and \
           self.batch_class.components is not None and name in self.batch_class.components:
            return getattr(self.data, name)
        raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None, copy=False, **kwargs):
        """ Create a Dataset object from another dataset with a new index
            (usually a subset of the source dataset index)

            Parameters
            ----------
            dataset : Dataset
                Source dataset

            index : DatasetIndex
                Set of items from source dataset which should be in the new Dataset

            batch_class : type
                a subclass of Batch class

            copy : bool
                whether to create a copy of the dataset or use the same instance wherever possible

            Returns
            -------
            Dataset
        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and cls._is_same_index(index, dataset.index):
            if not copy:
                return dataset
        if copy:
            index = cp.copy(index)
        bcl = batch_class if batch_class is not None else dataset.batch_class
        return cls(index, batch_class=bcl, preloaded=dataset.preloaded, **{**dataset.get_attrs(), **kwargs})

    def __copy__(self):
        return self.from_dataset(self, self.index, copy=True)

    def copy(self):
        """ Make a shallow copy of the dataset object """
        return cp.copy(self)

    @staticmethod
    def build_index(index, *args, **kwargs):
        """ Check if instance of the index is DatasetIndex
            if it is not - create DatasetIndex from inputs

            Parameters
            ----------
            index : DatasetIndex or any

            Returns
            -------
            DatasetIndex
        """
        if isinstance(index, DatasetIndex):
            return index
        return DatasetIndex(index, *args, **kwargs)

    @staticmethod
    def _is_same_index(index1, index2):
        """ Check if index1 and index2 are equals

            Parameters
            ----------
            index1 : DatasetIndex

            index2 : DatasetIndex

            Returns
            -------
            bool
        """
        return (isinstance(index1, type(index2)) or isinstance(index2, type(index1))) and \
               index1.indices.shape == index2.indices.shape and \
               np.all(index1.indices == index2.indices)

    def create_subset(self, index):
        """ Create a dataset based on the given subset of indices

            Parameters
            ----------
            index : DatasetIndex or np.array

            Returns
            -------
            Dataset

            Raises
            ------
            IndexError
                When a user wants to create a subset from source dataset it is necessary to be confident
                that the index of new subset lies in the range of source dataset's index.
                If the index lies out of the source dataset index's range, the IndexError is raised.

        """
        indices = index.indices if isinstance(index, DatasetIndex) else index
        if not np.isin(indices, self.indices).all():
            raise IndexError
        return type(self).from_dataset(self, self.index.create_subset(index))

    def create_batch(self, index, pos=False, *args, **kwargs):
        """ Create a batch from given indices.

            Parameters
            ----------
            index : DatasetIndex
                Indices of dataset objects that should be included into batch

            pos : bool
                Whether `index` contains elements positions. Defaults to False

            Returns
            -------
            Batch

            Notes
            -----
            If `pos` is `False`, then `index` should contain the indices
            that should be included in the batch, otherwise `index`
            should contain their positions in current index.
        """
        if not isinstance(index, DatasetIndex):
            index = self.index.create_batch(index, pos, *args, **kwargs)
        return self.batch_class(index, dataset=self, preloaded=self.data, copy=self._copy, **kwargs)

    def pipeline(self, config=None):
        """ Start a new data processing workflow

            Parameters
            ----------
            config : Config or dict
                Config lets you initialize variables in the Pipeline object, e.g. for the augmentation task
                https://analysiscenter.github.io/batchflow/intro/pipeline.html#initializing-a-variable

            Returns
            -------
            Pipeline
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
            Pipeline
                Pipeline object which now has Dataset object as attribute

            Raises
            ------
            TypeError
                If the type of other is not a Pipeline
        """
        if not isinstance(other, Pipeline):
            raise TypeError("Pipeline is expected, but got %s. Use as dataset >> pipeline" % type(other))
        return other << self

    def cv(self, n):
        """ Return a dataset which corresponds to n-th CV split """
        if n > self.n_splits - 1:
            raise ValueError("The dataset has been split into fewer splits than %d" % n)
        return  getattr(self, 'cv' + str(n))

    def CV(self, expr):
        """ Return a dataset which corresponds to the fold defined as NamedExpression """
        return  F(self.cv)(expr)

    def cv_split(self, method='kfold', n_splits=5, shuffle=False):
        """ Create datasets for cross-validation

        Datasets are available as `cv0`, `cv1` and so on. And they are already split into train and test parts.

        Another way to access these splits is `train.cv0`, `train.cv1`, ..., `test.cv0`, `test.cv1`, ...

        Note that each pair (e.g. `cv0.train` and `train.cv0`) refers to the very same instance of a dataset,
        i.e. if you change `train.cv0`, `cv0.train` will also change.

        Parameters
        ----------
        method : {'kfold'}
            a splitting method (only `kfold` is supported)

        n_splits : int
            a number of folds

        shuffle
            specifies the order of items (see :meth:`~.DatasetIndex.shuffle`)

        Examples
        --------

        ::

            dataset = Dataset(10)
            dataset.cv_split(n_splits=3)
            print(dataset.cv0.test.indices) # [0, 1, 2, 3]
            print(dataset.cv1.test.indices) # [4, 5, 6]
            print(dataset.cv2.test.indices) # [7, 8, 9]
            print(dataset.test.cv0.indices) # [0, 1, 2, 3]
            print(dataset.test.cv1.indices) # [4, 5, 6]
            print(dataset.test.cv2.indices) # [7, 8, 9]
        """
        # pylint: disable=access-member-before-definition
        if self.n_splits is not None:
            for i in range(self.n_splits):
                cv_attr = 'cv'+str(i)
                delattr(self, cv_attr)
                if self.train is not None:
                    delattr(self.train, cv_attr)
                if self.test is not None:
                    delattr(self.test, cv_attr)

        self.n_splits = n_splits

        order = self.index.shuffle(shuffle)

        if method == 'kfold':
            splits = self._split_kfold(n_splits, order)
        else:
            raise ValueError("Unknown split method:", method)

        self.train = self.copy()
        self.test = self.copy()

        self.train.n_splits = self.n_splits
        self.test.n_splits = self.n_splits

        for i in range(n_splits):
            test_indices = splits[i]
            train_splits = list(set(range(n_splits)) - {i})
            train_indices = np.concatenate(np.asarray(splits)[train_splits])

            setattr(self, 'cv'+str(i), self.copy())
            cv_dataset = getattr(self, 'cv'+str(i))
            cv_dataset.train = self.create_subset(train_indices)
            cv_dataset.test = self.create_subset(test_indices)
            setattr(self.train, 'cv'+str(i), cv_dataset.train)
            setattr(self.test, 'cv'+str(i), cv_dataset.test)

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

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for k, v in state.items():
            # this warrants that all hidden objects are reconstructed upon unpickling
            setattr(self, k, v)
