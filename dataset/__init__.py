""" Dataset enables a fast processing of large dataset using flexible pipelines """
import sys

from .base import Baseset
from .batch import Batch, ArrayBatch, DataFrameBatch
from .dataset import Dataset
from .pipeline import Pipeline
from .named_expr import B, C, F, V
from .jointdataset import JointDataset, FullDataset
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, model
from .exceptions import SkipBatchException


if sys.version_info < (3, 5):
    raise ImportError("Dataset module requires Python 3.5 or higher")
