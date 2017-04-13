""" Dataset module implements Dataset, DatasetIndex, Preprocess and Batch classes"""
import sys

from .base import Baseset
from .batch import Batch, ArrayBatch, DataFrameBatch
from .dataset import Dataset
from .jointdataset import JointDataset, FullDataset
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel

if sys.version_info < (3, 4):
    raise ImportError("Dataset module requires Python 3.4 or higher")
