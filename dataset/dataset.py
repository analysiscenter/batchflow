""" Dataset """

from .base import Baseset
from .dsindex import DatasetIndex


class Dataset(Baseset):
    """ Dataset """
    def __init__(self, index, batch_class=None, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.batch_class = batch_class


    @classmethod
    def from_dataset(cls, dataset, index, batch_class=None):
        """ Create Dataset from another dataset with new index
            (usually subset of the source dataset index)
        """
        if (batch_class is None or (batch_class == dataset.batch_class)) and (index == dataset.index):
            return dataset
        else:
            bcl = batch_class if batch_class is not None else dataset.batch_class
            return cls(index, batch_class=bcl)

    @staticmethod
    def build_index(index):
        """ Create index """
        if isinstance(index, DatasetIndex):
            return index
        else:
            return DatasetIndex(index)


    def cv_split(self, shares=0.8, shuffle=False):
        """ Split the dataset into train, test and validation sub-datasets
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           ds.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           ds.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           ds.cv_split([0.5, 0.3, 0.2])
        """
        self.index.cv_split(shares, shuffle)

        self.train = Dataset.from_dataset(self, self.index.train)
        if self.index.test is not None:
            self.test = Dataset.from_dataset(self, self.index.test)
        if self.index.validation is not None:
            self.validation = Dataset.from_dataset(self, self.index.validation)


    def create_batch(self, batch_indices, pos=False, *args, **kwargs):
        """ Create a batch from given indices 
            if pos is False then batch_indices contains the value of indices 
            which should be included in the batch
            otherwise batch_indices contains positions in the index
        """
        batch_ix = self.index.create_batch(batch_indices, pos, *args, **kwargs)
        return self.batch_class(batch_ix, *args, **kwargs)


class JointDataset(Baseset):
    """ Dataset comprising several Datasets """
    def __init__(self, datasets, *args, **kwargs):
        if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
            raise TypeError("Expected a non-empty list-like with instances of Dataset or Preprocessing.")
        else:
            index_len = None
            for dataset in datasets:
                if not isinstance(dataset, (Dataset, Preprocessing)):
                    raise TypeError("Dataset or Preprocessing is expected, but instead %s was given." % type(dataset))
                ds_ilen = len(dataset.index)
                if index_len is None:
                    index_len = ds_ilen
                elif index_len != ds_ilen:
                    raise TypeError("All datasets should have indices of the same length.")

        self.datasets = datasets
        super().__init__(*args, **kwargs)
        self.batch_generator = None


    @staticmethod
    def build_index(*args, **kwargs):
        return DatasetIndex(np.arange(len(self.datasets[0])))


    def create_subset(self, index):
        """ Create new JointDataset """
        ds_set =list()
        for dataset in self.datasets:
            ds_set.append(Dataset.from_dataset(dataset, dataset.index.subset_by_pos(index)))
        return JointDataset(ds_set)


    def cv_split(self, shares=0.8, shuffle=False):
        """ Split the dataset into train, test and validation sub-datasets
        Subsets are available as .train, .test and .validation respectively

        Usage:
           # split into train / test in 80/20 ratio
           ds.cv_split()
           # split into train / test / validation in 60/30/10 ratio
           ds.cv_split([0.6, 0.3])
           # split into train / test / validation in 50/30/20 ratio
           ds.cv_split([0.5, 0.3, 0.2])
        """
        self.index.cv_split(shares, shuffle)

        self.train = self.create_subset(self.index.train)
        if self.index.test is not None:
            self.test =self.create_subset(self.index.test)
        if self.index.validation is not None:
            self.validation = self.create_subset(self.index.validation)


    def create_batch(self, batch_indices, pos=True, *args, **kwargs):
        """ Create a list of batches from all source datasets """
        ds_batches =list()
        for dataset in self.datasets:
            ds_batches.append(dataset.create_batch(ix_batch, pos, *args, **kwargs))
        return ds_batches


class FullDataset(JointDataset):
    """ Dataset which include data and target sub-Datasets """
    def __init__(self, data, target):
        super.__init__((data, target))

    @property
    def data(self):
        return self.datasets[0]

    @property
    def target(self):
        return self.datasets[1]
