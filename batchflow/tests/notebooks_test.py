""" Run multiple notebooks. """
# pylint: disable=import-error
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import warnings
from glob import glob
import pytest
from tensorflow.test import is_gpu_available



NO_GPU = pytest.mark.skipif(not is_gpu_available(), reason='No GPU')

NOTEBOOKS_DIR = './notebooks/'
NOTEBOOKS = glob(NOTEBOOKS_DIR + '*.ipynb')

TUTORIALS_DIR = './../../examples/tutorials/'
TUTORIALS = glob(TUTORIALS_DIR + '*.ipynb')
ALLOWED_TUTORIALS = [
    '01',
    '02',   # quite long
    # '03', # very long
    '04',
    '07',
    # '10', # requires `multiprocess` module
]

MICROBATCH_LIST = [None, 8] # each integer values must be a divisor of 16
DEVICE_LIST = ['CPU:0', pytest.param('GPU:*', marks=NO_GPU)] # set your own value(s) for used devices

PARAMETERS = []
# Run every notebook in test directory for every combination of microbatching
PARAMETERS += [(path, mb) for path in NOTEBOOKS
               for mb in MICROBATCH_LIST]

# Run selected notebooks inside tutorials dir without microbatching
PARAMETERS += [(path, None) for path in TUTORIALS
               if path.split('/')[-1][:2] in ALLOWED_TUTORIALS]

_ = [print(item) for item in PARAMETERS]


# Some of the actions are appropriate in notebooks, but better be ignored in tests
BAD_PREFIXES = ['get_ipython', 'plt', 'plot', 'figure', 'ax.',]


@pytest.mark.slow
@pytest.mark.parametrize('path, microbatch', PARAMETERS)
@pytest.mark.parametrize('device', DEVICE_LIST)
def test_run_notebooks(path, microbatch, device):
    """ There are a lot of examples in different notebooks, and all of them should be working.

    Parameters
    ----------
    path : str
        Location of notebook to run.

    microbatch : int or None
        If None, then no microbatch is applied.
        If int, then size of microbatch used.

    device : str or None
        If None, then default device behaviour is used.
        If str, then any option of device configuration from :class:`.tf.TFModel` is supported.

    Notes
    -----
    `device` is moved to separate parameter in order to work properly with `parametrize`.
    """
    # pylint: disable=exec-used
    if path.startswith(TUTORIALS_DIR) and 'CPU' not in device:
        pytest.skip("Tutorials don't utilize device config.")

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
    exec(code, {'MICROBATCH': microbatch, 'DEVICE': device})
