""" Contains basic Batch classes """

import os

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
from .decorators import action, inbatch_parallel, ModelDecorator
from .dataset import Dataset
from .batch_base import BaseBatch
from .components import MetaComponentsTuple


class Batch(BaseBatch):
    """ The core Batch class """
    def __init__(self, index, preloaded=None):
        self._item_class = self._make_item_class()
        super().__init__(index)
        self._preloaded = preloaded

    @classmethod
    def from_data(cls, index, data):
        """ Create batch from a given dataset """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

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
        res = self._data if self._components is None else self._data_named
        return res if res is not None else self._empty_data

    @property
    def components(self):
        """ Return data components names """
        return None

    @property
    def _components(self):
        """ Return data components names """
        comps = self.components
        return dict(zip(comps, range(len(comps)))) if comps is not None else None

    def get_pos(self, data, component, index):
        """ Return a position in data for a given index

        Parameters:
            data: if None, get_pos should return a position in self.data
            components: could be one of [None, int or string]
                None: data has no components (e.g. just an array or pandas.DataFrame)
                int: a position of a data component, when components names are not defined
                str: a name of a data component
            index: an index
        Returns:
            int - a position in a batch data where an item with a given index is stored
        It is used to read / write data in a given component:
            batch_data = data.component[pos]
            data.component[pos] = new_data

        Examples:
            if self.data holds a numpy array, then get_pos(None, None, index) should
            just return self.index.get_pos(index)
            if self.data.images contains BATCH_SIZE images as a numpy array,
                then get_pos(None, 'images', index) should return self.index.get_pos(index)
            if self.data.labels is a dict {index: label}, then get_pos(None, 'labels', index) should return index.

            if data is not None, then you need to know in advance how to get a position for a given index.
            For instance, data is a large numpy array, a batch is a subset of this array and
            batch.index holds row numbers from a large arrays.
            Thus, get_pos(data, None, index) should just return index.

            A more complicated example of data:
            - batch represent small crops of large images
            - self.data.source holds a few large images (e.g just 5 items)
            - self.data.coords holds coordinates for crops (e.g. it contains 100 items)
            - self.data.image_no holds an array of image numbers for each crop (so it also contains 100 items)
            then get_pos(None, 'source', index) should return self.data.image_no[self.index.get_pos(index)].
            Whilst, get_pos(data, 'source', index) should return data.image_no[index].
        """
        _ = component
        if data is None:
            return self.index.get_pos(index)
        else:
            return index

    def _make_item_class(self):
        if self._components is None:
            return None
        else:
            return MetaComponentsTuple(self.__class__.__name__ + 'Components', components=self.components)

    @property
    def _empty_data(self):
        return None if self._components is None else self._item_class()

    def __getattr__(self, name):
        if self._components is not None and name in self._components:
            return getattr(self.data, name)
        else:
            raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

    def __setattr__(self, name, value):
        if self._components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                self._data_named = self._item_class(data=self._data)
            elif name in self._components:
                setattr(self._data_named, name, value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def put_into_data(self, items, data):
        """ Loads data into _data property """
        if self._components is None:
            _src = data
        else:
            _src = data if isinstance(data, tuple) else tuple([data])
        self._data = self.get_items(items, _src)

    def get_items(self, index, data=None):
        """ Return one or several data items from a data source """
        if data is None:
            _data = self.data
        else:
            _data = data

        if self._item_class is not None and isinstance(_data, self._item_class):
            pos = [self.get_pos(None, comp, index) for comp in self.components]   # pylint: disable=not-an-iterable
            res = self._item_class(data=_data, pos=pos)
        elif isinstance(_data, tuple):
            comps = self.components if self.components is not None else range(len(_data))
            res = tuple(data_item[self.get_pos(data, comp, index)] if data_item is not None else None
                        for comp, data_item in zip(comps, _data))
        elif isinstance(_data, dict):
            res = dict(zip(_data.keys(), (_data[comp][self.get_pos(data, comp, index)] for comp in _data)))
        else:
            res = _data[self.get_pos(data, None, index)]
        return res

    def get(self, item=None, component=None):
        """ Return an item from the batch or the component """
        if item is None:
            if component is None:
                raise ValueError("item and component cannot be both None")
            return getattr(self, component)
        else:
            if component is None:
                res = self[item]
            else:
                res = self[item]
                res = getattr(res, component)
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

    def get_model_by_name(self, model_name):
        """ Return a model specification given its name """
        return ModelDecorator.get_model_by_name(self, model_name)

    def get_all_model_names(self):
        """ Return all model names in the batch class """
        return ModelDecorator.get_all_model_names(self)

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def do_nothing(self, *args, **kwargs):
        """ An empty action (might be convenient in complicated pipelines) """
        _ = args, kwargs
        return self

    @action
    def load(self, src, fmt=None):
        """ Load data from a source """
        if fmt is None:
            self.put_into_data(self.indices, src)
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
        """ Apply a function to each item in the batch

        Args:
            dst: string - a destination component name, e.g. 'images' or 'masks'
            src: string - a source component name
            func: a callable - a function to apply to each item in the source component

        apply_transform does the following:
            for item in batch:
                self.dst[item] = func(self.src[item], *args, **kwargs)
        """
        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least of of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(ix, src)
            else:
                pos = self.get_pos(None, dst, ix)
                src_attr = src[pos]
            _args = tuple([src_attr, *args])

        if isinstance(dst, str):
            dst_attr = self.get(component=dst)
            pos = self.get_pos(None, dst, ix)
        else:
            dst_attr = dst
            pos = self.get_pos(None, src, ix)
        dst_attr[pos] = func(*_args, **kwargs)

    @action
    def apply_transform_all(self, dst, src, func, *args, **kwargs):
        """ Apply a function the whole batch at once

        Args:
            dst: string - a destination component name, e.g. 'images' or 'masks'
            src: string - a source component name
            func: a callable - a function to apply to each item in the source component

        apply_transform_all does the following:
            self.dst = func(self.src, *args, **kwargs)
        """

        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least of of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = self.get(component=src)
            else:
                src_attr = src
            _args = tuple([src_attr, *args])

        tr_res = func(*_args, **kwargs)
        if isinstance(dst, str):
            setattr(self, dst, tr_res)
        else:
            dst[:] = tr_res
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
            dfr = pd.read_csv(src, *args, **kwargs) # pylint: disable=redefined-variable-type
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
