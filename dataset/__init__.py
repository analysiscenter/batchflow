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
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, mjit
from .exceptions import SkipBatchException
from .sampler import Sampler, ConstantSampler, NumpySampler, HistoSampler, ScipySampler


__version__ = '0.3.0'


m = sys.modules[__name__]
m.BEST_PRACTICE = {}

def _setup_best_practice_options(options, flag):
    if options is not None:
        if isinstance(options, str):
            options = [options]
        for i in options:
            if flag is None:
                if i in m.BEST_PRACTICE:
                    del m.BEST_PRACTICE[i]
            else:
                m.BEST_PRACTICE.update({i: flag})

def setup_best_practice(enable=None, disable=None, clear=None):
    """ Enables and disables best practice options """
    _setup_best_practice_options(enable, True)
    _setup_best_practice_options(disable, False)
    _setup_best_practice_options(clear, None)

def is_best_practice(option='all'):
    """ Check if a best practice option is enabled """
    value = m.BEST_PRACTICE.get(option)
    return value if value is not None else m.BEST_PRACTICE.get('all', False)
