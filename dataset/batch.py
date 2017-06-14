""" Contains basic Batch classes """

import os
from binascii import hexlify

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
from .decorators import action
from .dataset import Dataset


class Batch:
    """ Base Batch class """
    def __init__(self, index, preloaded=None):
        self.index = index
        self._data = None
        self._preloaded = preloaded

    @classmethod
    def from_data(cls, data):
        """ Create batch from given dataset """
        # this is equiv to self.data = data[:]
        return cls(np.arange(len(data)), preloaded=data)

    def as_dataset(self, dataset_class=Dataset):
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
            self.load(self._preloaded)
        return self._data

    def __getitem__(self, item):
        if isinstance(self.data, tuple):
            res = tuple(data_item[item] if data_item is not None else None for data_item in self.data)
        else:
            res = self.data[item]
        return res

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    @staticmethod
    def make_filename():
        """ Generate unique filename for the batch """
        random_data = np.random.uniform(0, 1, size=10) * 123456789
        # probability of collision is around 2e-10.
        filename = hexlify(random_data.data)[:8]
        return filename.decode("utf-8")

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
        """ Load data from a file or another data source """
        raise NotImplementedError()

    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to disk """
        raise NotImplementedError()

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)


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
            self.data.to_hdf(fullname, *args, **kwargs)
        elif fmt == 'csv':
            self.data.to_csv(fullname, *args, **kwargs)
        else:
            raise ValueError('Unknown format %s' % fmt)
        return self
