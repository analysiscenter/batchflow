""" BatchFlow enables a fast processing of large dataset using flexible pipelines """
import sys
import os
import re

if sys.version_info < (3, 8):
    raise ImportError("BatchFlow module requires Python 3.8 or higher")

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
from .utils_telegram import TelegramMessage
from .utils_transforms import Normalizer, Quantizer


from .utils_import import try_import, make_delayed_import
plot = try_import(module='.plotter', package=__name__, attribute='plot',
                  help='Try `pip install batchflow[image]`!')

pylint_notebook = make_delayed_import(module='.utils_notebook', package=__name__, attribute='pylint_notebook')
get_available_gpus = make_delayed_import(module='.utils_notebook', package=__name__, attribute='get_available_gpus')
set_gpus = make_delayed_import(module='.utils_notebook', package=__name__, attribute='set_gpus')


try:
    __version__ = version('batchflow')
except PackageNotFoundError:
    # batchflow cannot be found within batchflow dev env only
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(pyproject_path, encoding="utf-8") as f:
        __version__ = re.search(r'^version\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)
