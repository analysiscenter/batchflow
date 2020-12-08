""" Research module.

.. note::
    This module requries multiprocess package <http://multiprocess.rtfd.io/>`_.
"""
from .domain import KV, Domain, Option, ConfigAlias
from .distributor import Distributor
from .logger import BaseLogger, FileLogger, PrintLogger, TelegramLogger
from .workers import Worker, PipelineWorker
from .named_expr import ResearchNamedExpression, REU, RP, RI, RC, RR, RD, REP, RID
from .research import Research
from .results import Results
