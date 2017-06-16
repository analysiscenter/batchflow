""" Contains basic Batch classes """

import os
from collections import namedtuple

try:
    import blosc
except ImportError:
    pass
import numpy as np
try:
    import pandas as pd
except ImportError:
    pass
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    pass

from .dsindex import DatasetIndex
from .decorators import action, inbatch_parallel
from .dataset import Dataset
from .batch_base import BaseBatch


class Batch(BaseBatch):
    """ The core Batch class """
    def __init__(self, index, preloaded=None):
        super().__init__(index)
        self._preloaded = preloaded
        comps = self.components
        self._components = dict(zip(comps, range(len(comps)))) if comps is not None else None

    @classmethod
    def from_data(cls, data):
        """ Create batch from a given dataset """
        # this is roughly equivalent to self.data = data
        return cls(np.arange(len(data)), preloaded=data)

    def as_dataset(self, dataset=None):
        """ Makes a new dataset from batch data
        Args:
            dataset: could be a dataset or a Dataset class
        Output:
            an instance of a class specified by `dataset` arg, preloaded with this batch data
        """
        if dataset is None:
            dataset_class = Dataset
        elif isinstance(dataset, Dataset):
            dataset_class = dataset.__class__
        elif isinstance(dataset, type):
            dataset_class = dataset
        else:
            raise TypeError("dataset should be an instance of some Dataset class or some Dataset class or None")
        return dataset_class(self.index, preloaded=self.data)

    @property
    def indices(self):
        """ Return an array-like with the indices """
        if isinstance(self.index, DatasetIndex):
            return self.index.indices
        else:
            return self.index

    def __len__(self):
        return len(self.index)

    @property
    def data(self):
        """ Return batch data """
        if self._data is None and self._preloaded is not None:
            # load data the first time it's requested
            self.load(self._preloaded)
        return self._data if self._data is not None else self._empty_data

    @property
    def components(self):
        """ Return data components names """
        return None

    def get_pos(self, component, index):
        """ Return a position in data for a given index

        Parameters:
            components: could be one of [None, int or string]
                None: data has no components (i.e. self.data is just an array)
                int: a position of a data component, when components names are not defined (see components)
                str: a name of a data component
            index: an index
        Returns:
            int - a position in a batch data where an item with a given index is stored
        It is used to read / write data in a given component:
            batch_data = data.component[pos]
            data.component[pos] = new_data

        Examples:
            if data holds a numpy array, then get_pos(None, index) should just return self.index.get_pos(index)
            if data.images contains BATCH_SIZE images as a numpy array,
                then get_pos('images', index) should return self.index.get_pos(index)

            if data.labels is a dict {index: label}, then get_pos('labels', index) should return index

            A more complicated example of data:
                - batch represent small crops of large images
                - data.source holds a few large images (e.g just 5 items)
                - data.coords holds coordinates for crops (e.g. it contains 100 items)
                - data.source_pos holds n arrays of source image id for each crop (so it also contains 100 items)
            then get_pos('source', index) should return data.source_pos[self.index.get_pos(index)]
        """
        _ = component
        return self.index.get_pos(index)

    @property
    def _item_class(self):
        if self._components is not None:
            item_class = namedtuple(self.__class__.__name__ + 'Item', self.components)
            item_class.__new__.__defaults__ = (None,) * len(self.components)
            return item_class
        else:
            raise AttributeError('components are not defined')

    @property
    def _empty_data(self):
        return None if self._components is None else self._item_class()

    def __getattr__(self, name):
        if self._components is not None and name in self._components:
            pos = self._components[name]
            return self.data[pos]
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def __setattr__(self, name, value):
        if self._components is not None and name in self._components:
            arg = {name: value}
            data = self._item_class(*self.data)._replace(**arg)  # pylint:disable=no-member
            self._data = tuple(data)
        else:
            super().__setattr__(name, value)

    def put_into_data(self, data, items):
        """ Loads data into _data property """
        if self.components is None:
            _src = data
        else:
            _src = data if isinstance(data, tuple) else tuple([data])
        self._data = self.get_items(items, _src)

    def get_items(self, index, data=None):
        """ Return one or several data items from a data source """
        if data is None:
            data = self.data
            get_pos = self.get_pos
        else:
            get_pos = lambda component, index: index

        if isinstance(data, tuple):
            comps = self.components if self.components is not None else range(len(data))
            res = tuple(data_item[get_pos(comp, index)] if data_item is not None else None for comp, data_item in zip(comps, data))
            if self.components is not None:
                res = self._item_class(*res)
        else:
            res = data[get_pos(None, index)]
        return res

    def __getitem__(self, item):
        return self.get_items(item)

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    @property
    def items(self):
        """ Init function for batch items parallelism """
        return [[self[ix]] for ix in self.indices]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    def infer_dtype(self, data=None):
        """ Detect dtype of batch data """
        if data is None:
            data = self.data
        return np.asarray(data).dtype.name

    def get_dtypes(self):
        """ Return dtype for batch data """
        if isinstance(self.data, tuple):
            return tuple(self.infer_dtype(item) for item in self.data)
        else:
            return self.infer_dtype(self.data)

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def load(self, src, fmt=None):
        """ Load data from a source """
        if fmt is None:
            self.put_into_data(src, self.indices)
        else:
            raise ValueError("Unknown format:", fmt)
        return self

    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to disk """
        return self

    @action
    @inbatch_parallel(init='indices')
    def apply_transform(self, ix, dst, src, func, *args, **kwargs):
        """ Apply a function to each item in the batch """
        if src is None:
            _args = args
        else:
            src_attr = getattr(self[ix], src)
            _args = tuple([src_attr, *args])

        dst_attr = getattr(self, dst)
        pos = self.get_pos(dst, ix)
        dst_attr[pos] = func(*_args, **kwargs)

    @action
    def apply_transform_all(self, dst, src, func, *args, **kwargs):
        """ Apply a function the whole batch at once """
        if src is None:
            _args = args
        else:
            src_attr = getattr(self, src)
            _args = tuple([src_attr, *args])
        setattr(self, dst, func(*_args, **kwargs))
        return self


class ArrayBatch(Batch):
    """ Base Batch class for array-like datasets """

    @staticmethod
    def _read_file(path, attr):
        with open(path, 'r' + attr) as file:
            data = file.read()
        return data


    @staticmethod
    def _write_file(path, attr, data):
        with open(path, 'w' + attr) as file:
            file.write(data)


    @action
    def load(self, src, fmt=None):
        """ Load data from another array or a file """

        # Read the whole source
        if fmt is None:
            _data = src
        elif fmt == 'blosc':
            packed_array = self._read_file(src, 'b')
            _data = blosc.unpack_array(packed_array)
        else:
            raise ValueError("Unknown format " + fmt)

        # But put into this batch only part of it (defined by index)
        try:
            # this creates a copy of the source data
            self._data = _data[self.indices]
        except TypeError:
            raise TypeError('Source is expected to be array-like')

        return self


    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to a file or into another array """
        filename = self.make_filename()
        fullname = os.path.join(dst, filename + '.' + fmt)

        if fmt is None:
            # think carefully when dumping to an array
            dst[self.indices] = self.data
        elif fmt == 'blosc':
            packed_array = blosc.pack_array(self.data)
            self._write_file(fullname, 'b', packed_array)
        else:
            raise ValueError("Unknown format " + fmt)
        return self


class DataFrameBatch(Batch):
    """ Base Batch class for datasets stored in pandas DataFrames """

    @action
    def load(self, src, fmt=None, *args, **kwargs):
        """ Load batch from a dataframe """
        # pylint: disable=no-member
        # Read the whole source
        if fmt is None:
            dfr = src
        elif fmt == 'feather':
            dfr = feather.read_dataframe(src, *args, **kwargs)
        elif fmt == 'hdf5':
            dfr = pd.read_hdf(src, *args, **kwargs) # pylint: disable=redefined-variable-type
        elif fmt == 'csv':
            dfr = pd.read_csv(src, *args, **kwargs)
        else:
            raise ValueError('Unknown format %s' % fmt)

        # But put into this batch only part of it (defined by index)
        if isinstance(dfr, pd.DataFrame):
            self._data = dfr.loc[self.indices]
        elif isinstance(dfr, dd.DataFrame):
            # dask.DataFrame.loc supports advanced indexing only with lists
            self._data = dfr.loc[list(self.indices)].compute()
        else:
            raise TypeError("Unknown DataFrame. DataFrameBatch supports only pandas and dask.")

        return self


    @action
    def dump(self, dst, fmt='feather', *args, **kwargs):
        """ Save batch data to disk
            dst should point to a directory where all batches will be stored
            as separate files named 'batch_id.format', e.g. '6a0b1c35.csv', '32458678.csv', etc.
        """
        filename = self.make_filename()
        fullname = os.path.join(dst, filename + '.' + fmt)

        if fmt == 'feather':
            feather.write_dataframe(self.data, fullname, *args, **kwargs)
        elif fmt == 'hdf5':
            self.data.to_hdf(fullname, *args, **kwargs)   # pylint:disable=no-member
        elif fmt == 'csv':
            self.data.to_csv(fullname, *args, **kwargs)   # pylint:disable=no-member
        else:
            raise ValueError('Unknown format %s' % fmt)
        return self
