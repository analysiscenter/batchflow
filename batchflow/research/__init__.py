""" Research module.

.. note::
    This module requries multiprocess package <http://multiprocess.rtfd.io/>`_.
"""
from .domain import Alias, Domain, Option, ConfigAlias
from .distributor import Distributor
from .named_expr import E, EC, O
from .research import Research
from .results import Results
from .experiment import Experiment, Executor
from .utils import transform_research_results
