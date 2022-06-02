""" Run multiple notebooks. """
import warnings
from glob import glob
import pytest



PLOT_NOTEBOOKS = glob('./plot_notebooks/*.ipynb')
PLOT_TUTORIALS = glob('./../../examples/plot/*.ipynb')

PATHS = PLOT_NOTEBOOKS + PLOT_TUTORIALS

# Some of the actions are appropriate in notebooks, but better be ignored in tests
BAD_PREFIXES = ['get_ipython']

@pytest.mark.parametrize('path', PATHS)
def test_run_notebooks(path):
    """ There are a lot of examples in different notebooks, and all of them should be working.

    Parameters
    ----------
    path : str
        Location of notebook to run.
    """
    # pylint: disable=exec-used
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
