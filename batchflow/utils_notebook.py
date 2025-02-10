""" Notebook utils. """
# ruff: noqa: E402; module-import-not-at-top-of-file

import warnings
warnings.warn("'pylint_notebook', 'get_available_gpus' and 'set_gpus' were moved into 'nbtools' and will be removed",
              category=DeprecationWarning, stacklevel=2)

from nbtools import pylint_notebook, get_available_gpus, set_gpus
