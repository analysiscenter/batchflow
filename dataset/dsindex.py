""" DatasetIndex """

import os
import glob
import numpy as np
from .base import Baseset


class DatasetIndex(Baseset):
    """ Stores an index for a dataset
    The index should be 1-d array-like, e.g. numpy array, pandas Series, etc.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos = self.build_pos()

    @classmethod
    def from_index(cls, *args, **kwargs):
        """Create index from another index """
        return cls(*args, **kwargs)

    @staticmethod
    def build_index(index):
        """ Check index type and structure """
        if callable(index):
            _index = index()
        else:
            _index = index

        if isinstance(_index, DatasetIndex):
            _index = _index.index
        else:
            # index should allow for advance indexing (i.e. subsetting)
            try:
                _ = _index[[0]]
            except TypeError:
                _index = np.asarray(_index)
            except IndexError:
                raise ValueError("Index cannot be empty")

        if len(_index.shape) > 1:
            raise TypeError("Index should be 1-dimensional")

        return _index

    def build_pos(self):
        """ Create a dictionary with positions in the index """
        pos_dict = dict()
        pos = 0
        for item in self.indices:
            pos_dict.update({item: pos})
            pos += 1
        return pos_dict

    def get_pos(self, index):
        """ Return position of an item in the index """
        return self._pos[index]

    def subset_by_pos(self, pos):
        """ Return subset of index by given positions in the index """
        return self.index[pos]

    def create_subset(self, index):
        """ Return a new index object based on the subset of indices given """
        return type(self)(index)

    def cv_split(self, shares=0.8, shuffle=False):
        """ Split index into train, test and validation subsets
        Shuffles index if necessary.
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           di.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           di.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           di.cv_split([0.5, 0.3, 0.2])
        """
        _, test_share, valid_share = self.calc_cv_split(shares)

        # TODO: make a view not copy if not shuffled
        order = np.arange(len(self))
        if shuffle:
            np.random.shuffle(order)

        if valid_share > 0:
            validation_pos = order[:valid_share]
            self.validation = self.create_subset(self.subset_by_pos(validation_pos))
        if test_share > 0:
            test_pos = order[valid_share : valid_share + test_share]
            self.test = self.create_subset(self.subset_by_pos(test_pos))
        train_pos = order[valid_share + test_share:]
        self.train = self.create_subset(self.subset_by_pos(train_pos))


    def next_batch(self, batch_size, shuffle=False, one_pass=False):
        """ Return next batch """
        num_items = len(self)

        # TODO: make a view not copy whenever possible
        if self._order is None:
            self._order = np.arange(num_items)
            if shuffle:
                np.random.shuffle(self._order)

        rest_items = None
        if self._start_index + batch_size >= num_items:
            rest_items = np.copy(self._order[self._start_index:])
            rest_of_batch = self._start_index + batch_size - num_items
            self._start_index = 0
            self._n_epochs += 1
            if shuffle:
                np.random.shuffle(self._order)
        else:
            rest_of_batch = batch_size

        new_items = self._order[self._start_index : self._start_index + rest_of_batch]
        # TODO: concat not only numpy arrays
        if rest_items is None:
            batch_items = new_items
        else:
            batch_items = np.concatenate((rest_items, new_items))

        if one_pass and rest_items is not None:
            return self.create_batch(rest_items, pos=True)
        else:
            self._start_index += rest_of_batch
            return self.create_batch(batch_items, pos=True)


    def gen_batch(self, batch_size, shuffle=False, one_pass=False):
        """ Generate batches """
        self._start_index = 0
        self._order = None
        _n_epochs = self._n_epochs
        while True:
            if one_pass and self._n_epochs > _n_epochs:
                raise StopIteration()
            else:
                yield self.next_batch(batch_size, shuffle, one_pass)


    def create_batch(self, batch_indices, pos=True, as_array=False):   # pylint: disable=arguments-differ
        """ Create a batch from given indices
        if pos is False then batch_indices contains the value of indices
        which should be included in the batch (so expected batch is just the very same batch_indices)
        otherwise batch_indices contains positions in the index
        """
        if isinstance(batch_indices, DatasetIndex):
            _batch_indices = batch_indices.index
        else:
            _batch_indices = batch_indices
        if pos:
            batch = self.subset_by_pos(_batch_indices)
        else:
            batch = _batch_indices
        if not as_array:
            batch = self.create_subset(batch)
        return batch


class FilesIndex(DatasetIndex):
    """ Index with the list of files or directories with the given path pattern

        Usage:
        Create sorted index of files in a directory:
        fi = FilesIndex('/path/to/data/files/*', sort=True)
        Create unsorted index of directories through all subdirectories:
        fi = FilesIndex('/path/to/data/archive*/patient*', dirs=True)
    """
    def __init__(self, *args, **kwargs):
        self._paths = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_index(cls, index, paths):   # pylint: disable=arguments-differ
        """Create index from another FilesIndex """
        return cls(index=index, path=None, paths=paths)

    def build_index(self, index=None, path=None, *args, **kwargs):     # pylint: disable=arguments-differ
        """ Build index from a path string or an index given """
        if path is None:
            return self.build_from_index(index, *args, **kwargs)
        else:
            return self.build_from_path(path, *args, **kwargs)

    def build_from_index(self, index, paths):
        """ Build index from another index for indices given """
        self._paths = dict((file, paths[file]) for file in index)
        return index

    def build_from_path(self, path, dirs=False, no_ext=False, sort=False):
        """ Build index from a path/glob or a sequence of paths/globs """
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        _all_index = None
        _all_paths = dict()
        for one_path in paths:
            _index, _paths = self.build_from_one_path(one_path, dirs, no_ext, sort)
            if _all_index is None:
                _all_index = _index
            else:
                _all_index = np.concatenate((_all_index, _index))
            _all_paths.update(_paths)
        self._paths = _all_paths
        return _all_index

    def build_from_one_path(self, path, dirs=False, no_ext=False, sort=False):
        """ Build index from a path/glob """
        check_fn = os.path.isdir if dirs else os.path.isfile
        pathlist = glob.iglob(path)
        _full_index = np.asarray([self.build_key(fname, no_ext) for fname in pathlist if check_fn(fname)])
        if sort:
            _order = np.argsort(_full_index[:, 0])
        else:
            _order = slice(None, None)
        _index = _full_index[_order, 0]
        _paths = _full_index[_order, 1]
        _paths = dict(zip(_index, _paths))
        return _index, _paths

    @staticmethod
    def build_key(fullpathname, no_ext=False):
        """ Create index item from full path name """
        if no_ext:
            key_name = '.'.join(os.path.basename(fullpathname).split('.')[:-1])
        else:
            key_name = os.path.basename(fullpathname)
        return key_name, fullpathname

    def get_fullpath(self, key):
        """ Return the full path name for an item in the index """
        return self._paths[key]

    def create_subset(self, index):
        """ Return a new FilesIndex based on the subset of indices given """
        return FilesIndex.from_index(index, self._paths)
