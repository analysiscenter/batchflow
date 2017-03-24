""" DatasetIndex """

import os
import glob
import numpy as np


class DatasetIndex:
    """ Stores an index for a dataset
    The index should be 1-d array-like, e.g. numpy array, pandas Series, etc.
    """
    def __init__(self, *args, **kwargs):
        _index = self.build_index(*args, **kwargs)
        self.index = self.check_index(_index)
        self.train = None
        self.test = None
        self.validation = None
        self._start_index = 0
        self._order = None
        self._n_epochs = 0


    @staticmethod
    def build_index(index):
        """ Return index. Child classes should generate index from the arguments given """
        return index

    @staticmethod
    def check_index(index):
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
            raise TypeError("index should be 1-dimensional")

        return _index


    def _subset_by_pos(self, pos):
        """ Return subset of index by given positions in the index """
        return self.index[pos]


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
        _shares = np.array(shares).ravel() # pylint: disable=no-member

        if _shares.shape[0] > 3:
            raise ValueError("shares must have no more than 3 elements")
        if _shares.sum() > 1:
            raise ValueError("shares must sum to 1")

        if _shares.shape[0] == 3:
            if not np.allclose(1. - _shares.sum(), 0.):
                raise ValueError("shares must sum to 1")
            train_share, test_share, valid_share = _shares
        elif _shares.shape[0] == 2:
            train_share, test_share, valid_share = _shares[0], _shares[1], 1 - _shares.sum()
        else:
            train_share, test_share, valid_share = _shares[0], 1 - _shares[0], 0.

        n_items = len(self.index)
        train_share, test_share, valid_share = \
            np.round(np.array([train_share, test_share, valid_share]) * n_items).astype('int')
        train_share = n_items - test_share - valid_share

        # TODO: make a view not copy if not shuffled
        order = np.arange(n_items)
        if shuffle:
            np.random.shuffle(order)

        if valid_share > 0:
            validation_pos = order[:valid_share]
            self.validation = DatasetIndex(self._subset_by_pos(validation_pos))
        if test_share > 0:
            test_pos = order[valid_share : valid_share + test_share]
            self.test = DatasetIndex(self._subset_by_pos(test_pos))
        train_pos = order[valid_share + test_share:]
        self.train = DatasetIndex(self._subset_by_pos(train_pos))


    @property
    def is_splitted(self):
        """ True if dataset was splitted into train / test / validation sub-datasets """
        return self.train is not None


    def next_batch(self, batch_size, shuffle=False, one_pass=False):
        """ Return next batch """
        num_items = len(self.index)

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
            return self.index[rest_items]
        else:
            self._start_index += rest_of_batch
            return self.index[batch_items]


    def gen_batch(self, batch_size, shuffle=False, one_pass=False):
        """ Generate one batch """
        self._start_index = 0
        self._order = None
        _n_epochs = self._n_epochs
        while True:
            if one_pass and self._n_epochs > _n_epochs:
                raise StopIteration()
            else:
                yield self.next_batch(batch_size, shuffle, one_pass)


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

    def build_index(self, path, dirs=False, no_ext=False, sort=False):    # pylint: disable=arguments-differ
        """ Generate index from path """
        check_fn = os.path.isdir if dirs else os.path.isfile
        pathlist = glob.iglob(path)
        _full_index = np.asarray([self.build_key(fname, no_ext) for fname in pathlist if check_fn(fname)])
        if sort:
            _order = np.argsort(_full_index[:, 0])
        else:
            _order = slice(None, None)
        _index = _full_index[_order, 0]
        _paths = _full_index[_order, 1]
        self._paths = dict(zip(_index, _paths))
        return _index

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
