""" Research module.

.. note::
    This module requries multiprocess package <http://multiprocess.rtfd.io/>`_.
"""
from .domain import Alias, Domain, Option, ConfigAlias
from .named_expr import E, EC, O, EP, R
from .research import Research
from .results import ResearchResults
from .experiment import Experiment, Executor
from .utils import get_metrics, convert_research_results
