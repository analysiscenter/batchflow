""" Dataset enables a fast processing of large dataset using flexible pipelines """

import sys

if sys.version_info < (3, 5):
    raise ImportError("Dataset module requires Python 3.5 or higher")

from .base import Baseset
from .batch import Batch, ArrayBatch, DataFrameBatch
from .batch_image import ImagesBatch
from .config import Config
from .dataset import Dataset
from .pipeline import Pipeline
from .named_expr import B, C, F, L, V, R, W, P
from .jointdataset import JointDataset, FullDataset
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, mjit
from .exceptions import SkipBatchException
from .sampler import Sampler, ConstantSampler as CS
from .sampler import NumpySampler as NS, TfSampler as TFS, HistoSampler as HS, ScipySampler as SS

__version__ = '0.3.0'

m = sys.modules[__name__]
m.BEST_PRACTICE = {}

def enable_best_practice(option='enable'):
    """ Sets a best practice option """
    m.BEST_PRACTICE.update({option: True})

def is_best_practice(option='enable'):
    """ Check if a best practice option is enabled """
    return m.BEST_PRACTICE.get(option, False)
