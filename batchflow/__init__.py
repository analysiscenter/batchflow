""" BatchFlow enables a fast processing of large dataset using flexible pipelines """

import sys

if sys.version_info < (3, 5):
    raise ImportError("BatchFlow module requires Python 3.5 or higher")

from .base import Baseset
from .batch import Batch
from .batch_image import ImagesBatch
from .config import Config
from .dataset import Dataset
from .pipeline import Pipeline
from .monitor import *
from .notifier import Notifier
from .named_expr import NamedExpression, B, C, F, L, V, M, D, R, W, P, I
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, mjit, deprecated, apply_parallel
from .exceptions import SkipBatchException, EmptyBatchSequence
from .sampler import Sampler, ConstantSampler, NumpySampler, HistoSampler, ScipySampler
from .utils import save_data_to
from .utils_notebook import in_notebook, get_notebook_path, get_notebook_name, pylint_notebook


__version__ = '0.3.0'
