""" Base class """
import numpy as np


class Baseset:
    def __init__(self, *args, **kwargs):
        self._index = self.build_index(*args, **kwargs)

        self.train = None
        self.test = None
        self.validation = None

        self._start_index = 0
        self._order = None
        self._n_epochs = 0
        self.batch_generator = None        


    @staticmethod
    def build_index(index):
        """ Create the index. Child classes should generate index from the arguments given """
        return index

    @property
    def index(self):
        """ Return the index """
        return self._index

    def __len__(self):
        return len(self.index)

    @property
    def is_splitted(self):
        """ True if dataset was splitted into train / test / validation sub-datasets """
        return self.train is not None

    def calc_cv_split(self, shares=0.8, shuffle=False):
        """ Calculate split into train, test and validation subsets

        Return: a tuple which contains number of items in train, test and validation subsets

        Usage:
           # split into train / test in 80/20 ratio
           bs.calc_cv_split()
           # split into train / test / validation in 60/30/10 ratio
           bs.calc_cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           bs.calc_cv_split([0.5, 0.3, 0.2])
        """
        _shares = np.array(shares).ravel() # pylint: disable=no-member

        if _shares.shape[0] > 3:
            raise ValueError("Shares must have no more than 3 elements")
        if _shares.sum() > 1:
            raise ValueError("Shares must sum to 1")

        if _shares.shape[0] == 3:
            if not np.allclose(1. - _shares.sum(), 0.):
                raise ValueError("Shares must sum to 1")
            train_share, test_share, valid_share = _shares
        elif _shares.shape[0] == 2:
            train_share, test_share, valid_share = _shares[0], _shares[1], 1 - _shares.sum()
        else:
            train_share, test_share, valid_share = _shares[0], 1 - _shares[0], 0.

        n_items = len(self)
        train_share, test_share, valid_share = \
            np.round(np.array([train_share, test_share, valid_share]) * n_items).astype('int')
        train_share = n_items - test_share - valid_share

        return train_share, test_share, valid_share


    def gen_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Generate batches """
        for ix_batch in self.index.gen_batch(batch_size, shuffle, one_pass):
            batch = self.create_batch(ix_batch, *args, **kwargs)
            yield batch

    def next_batch(self, batch_size, shuffle=False, one_pass=False, *args, **kwargs):
        """ Return a tuple of batches from all source datasets """
        if self.batch_generator is None:
            self.batch_generator = self.gen_batch(batch_size, shuffle=shuffle, one_pass=one_pass, *args, **kwargs)
        batch = next(self.batch_generator)
        return batch

    def create_batch(self, batch_indices, pos=True):
        raise NotImplementedError("create_batch should be defined in child classes")
