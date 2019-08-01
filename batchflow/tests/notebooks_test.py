""" Run multiple notebooks. """
import warnings
from glob import glob
import pytest



BAD_PREFIXES = ['get_ipython', 'plt', 'plot', 'figure', 'ax.',]
ALLOWED_TUTORIALS = [
    '01',
    '02',   # quite long
    # '03', # very long
    '04',
    '07',
    # '10', # requires multiprocess module
]

PARAMETERS = []
PARAMETERS += [(path, mb) for path in glob('./notebooks/*.ipynb')
               for mb in [None, 4]]
PARAMETERS += [(path, None) for path in glob('./../../examples/tutorials/*.ipynb')
               if path.split('/')[-1][:2] in ALLOWED_TUTORIALS]

print('\n', PARAMETERS, '\n')


@pytest.mark.slow
@pytest.mark.parametrize('parameter', PARAMETERS)
def test_run_notebooks(parameter):
    """ There are a lot of examples in different notebooks, and all of them
    should be working.
    """
    # pylint: disable=exec-used
    path, microbatch = parameter

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
    exec(code, {'MICROBATCH': microbatch})
