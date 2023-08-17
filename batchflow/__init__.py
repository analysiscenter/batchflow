""" BatchFlow enables a fast processing of large dataset using flexible pipelines """
import sys

if sys.version_info < (3, 5):
    raise ImportError("BatchFlow module requires Python 3.5 or higher")

from importlib.metadata import version, PackageNotFoundError

from .base import Baseset
from .batch import Batch
from .batch_image import ImagesBatch
from .config import Config
from .dataset import Dataset
from .pipeline import Pipeline
from .monitor import *
from .notifier import Notifier, notifier
from .named_expr import NamedExpression, B, L, C, F, V, M, D, R, W, P, PP, I, eval_expr
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, any_action_failed, mjit, deprecated, apply_parallel
from .exceptions import SkipBatchException, EmptyBatchSequence, StopPipeline
from .sampler import Sampler, ConstantSampler, NumpySampler, HistoSampler, ScipySampler
from .utils import save_data_to, read_data_from
from .utils_random import make_rng, make_seed_sequence, spawn_seed_sequence
from .utils_notebook import in_notebook, get_notebook_path, get_notebook_name, pylint_notebook,\
                            get_available_gpus, set_gpus
from .utils_telegram import TelegramMessage
from .utils_transforms import Normalizer, Quantizer


from .utils_import import try_import
plot = try_import(module='.plotter', package=__name__, attribute='plot',
                  help='Try `pip install batchflow[image]`!')


try:
    __version__ = version('batchflow')
except PackageNotFoundError:
    # batchflow cannot be found within batchflow dev env only
    pass
