""" FullDataset """
import numpy as np
from .dataset import Dataset
from .preprocess import Preprocessing


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
    def build_index(datasets, *args, **kwargs):
        return DatasetIndex(np.arange(len(datasets[0])))


    def create_subset(self, index):
        """ Create new JointDataset """
        ds_set = list()
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
            self.test = self.create_subset(self.index.test)
        if self.index.validation is not None:
            self.validation = self.create_subset(self.index.validation)


    def create_batch(self, batch_indices, pos=True, *args, **kwargs):
        """ Create a list of batches from all source datasets """
        ds_batches = list()
        for dataset in self.datasets:
            ds_batches.append(dataset.create_batch(ix_batch, pos, *args, **kwargs))
        return ds_batches


class FullDataset(JointDataset):
    """ Dataset which include data and target sub-Datasets """
    def __init__(self, data, target):
        super.__init__((data, target))

    @property
    def data(self):
        """ Data datset """
        return self.datasets[0]

    @property
    def target(self):
        """ Target datset """
        return self.datasets[1]
