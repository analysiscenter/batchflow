""" Run multiple notebooks. """
# pylint: disable=import-error
import os

import warnings
from glob import glob
import pytest



NOTEBOOKS_DIR = './plot_notebooks/'
NOTEBOOKS = glob(NOTEBOOKS_DIR + '*.ipynb')

TUTORIALS_DIR = './../../examples/plot/'
TUTORIALS = glob(TUTORIALS_DIR + '*.ipynb')

PARAMETERS = []
# Run every notebook in test directory
PARAMETERS += [path for path in NOTEBOOKS]

# Run plot notebooks inside tutorials dir
PARAMETERS += [path for path in TUTORIALS]

_ = [print(item) for item in PARAMETERS]

# Some of the actions are appropriate in notebooks, but better be ignored in tests
BAD_PREFIXES = ['get_ipython']
                
@pytest.mark.parametrize('path', PARAMETERS)
def test_run_notebooks(path):
    """ There are a lot of examples in different notebooks, and all of them should be working.

    Parameters
    ----------
    path : str
        Location of notebook to run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nbconvert import PythonExporter
        code, _ = PythonExporter().from_filename(path)

    code_ = []
    for line in code.split('\n'):
        if not line.startswith('#'):
            flag = sum([name in line for name in BAD_PREFIXES])
            if flag == 0:
                code_.append(line)

    code = '\n'.join(code_)
    print(code)
    exec(code, {})
