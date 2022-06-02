""" Utility functions to work with Jupyter Notebooks. """
import os
import re
import json
import warnings

import numpy as np

# Additionally imports 'requests`, 'ipykernel`, `jupyter_server`, `nbconvert`, `pylint`,
#                      'nbconvert', 'IPython' and `nvidia_smi`, if needed


def in_notebook():
    """ Return True if in Jupyter notebook and False otherwise. """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        if shell == 'TerminalInteractiveShell':
            return False
        return False
    except NameError:
        return False

def get_notebook_path():
    """ Return the full absolute path of the current Jupyter notebook,
    for example, `/path/path/path/My_notebook_title.ipynb`.

    If run outside Jupyter notebook, returns None.
    """
    if not in_notebook():
        return None

    import requests
    import ipykernel

    # Id of the current running kernel: a string uid
    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)

    # Get running servers for both JupyterLab v2.# and v3.#
    from notebook.notebookapp import list_running_servers as list_running_servers_v2
    from jupyter_server.serverapp import list_running_servers as list_running_servers_v3
    servers = list(list_running_servers_v2()) + list(list_running_servers_v3())

    for server in servers:
        root_dir = server.get('root_dir') or server.get('notebook_dir')
        response = requests.get(requests.compat.urljoin(server['url'], 'api/sessions'),
                                params={'token': server.get('token', '')})

        for params in json.loads(response.text):
            if params['kernel']['id'] == kernel_id:
                relative_path = params['notebook']['path']
                return os.path.join(root_dir, relative_path)
    raise ValueError(f'Unable to find kernel `{kernel_id}` in {len(servers)} servers!')

def get_notebook_name():
    """ Return the title of the current Jupyter notebook without base directory and extension,
    for example, `My_notebook_title`.

    If run outside Jupyter notebook, returns None.
    """
    if not in_notebook():
        return None

    return os.path.splitext(get_notebook_path())[0].split('/')[-1]


def pylint_notebook(path=None, options='', printer=print, ignore_comments=True, ignore_codes=tuple(),
                    use_pylintrc=True, keep_script=False, return_report=False):
    """ Run pylint on entire Jupyter notebook.
    Under the hood, the notebook is converted to regular `.py` script,
    special IPython commands like magics removed, and then pylint is executed.

    If run outside Jupyter notebook, returns 1.

    Parameters
    ----------
    path : str, optional
        Path to run linter on. If not provided, the callee notebook is linted.
    options : str
        Additional flags for linter execution, for example, the pylint configuration options.
    printer : callable
        Method for displaying results.
    ignore_comments : bool
        Whether to ignore markdown cells and comments in code.
    ignore_codes : sequence
        Pylint errors to ignore.
        By default, `invalid-name`, `import-error` and `wrong-import-position` are disabled.
    use_pylintrc : bool
        Whether to use the `BatchFlow` pylint configuration.
    keep_script : bool
        Whether to keep temporal `.py` file after command execution.
    return_report : bool
        If True, then this function returns the string representation of produced report.
        If False, then 0 is returned.
    """
    if not in_notebook():
        return 1

    from nbconvert import PythonExporter
    from pylint import epylint as lint

    # Parse parameters
    path = path or get_notebook_path()
    options = options if options.startswith(' ') else ' ' + options
    ignore_codes = set(ignore_codes)
    ignore_codes.update({'import-error', 'wrong-import-position', 'invalid-name',
                         'unnecessary-semicolon', 'trailing-whitespace', 'trailing-newlines'})

    # Try to add `pylintrc` configuration file to options
    if use_pylintrc and 'rcfile' not in options:
        # Locate the batchflow pylintrcfile
        # The loop converts the __file__ path, which is a combination of absolute and relative path, to an absolute
        pylintcrc_path = []
        for item in __file__.split('/')[:-2]:
            if item != '..':
                pylintcrc_path.append(item)
            else:
                pylintcrc_path.pop(-1)
        pylintcrc_path = '/' + os.path.join(*pylintcrc_path, 'pylintrc')

        if os.path.exists(pylintcrc_path):
            options += f' --rcfile {pylintcrc_path}'

    # Convert the notebook contents to raw string without outputs
    code, _ = PythonExporter().from_filename(path)

    # Unwrap code lines from line/cell magics
    code_list = []
    cell_codes, cell_counter = [], 0
    cell_code_lines, cell_code_counter = [], 1

    for line in code.split('\n'):
        # Line magics: remove autoreload
        if line.startswith('get_ipython().run_line_magic'):
            if 'autoreload' in line:
                line = ''
            else:
                line = line[line.find(',')+3:-2]

        # Cell magics: contain multiple lines
        if line.startswith('get_ipython().run_cell_magic'):
            line = line[line.find(',')+1:]
            line = line[line.find(',')+3:-2]

            lines = line.split('\\n')
        else:
            lines = [line]

        # Update all the containers
        for part in lines:
            code_list.append(part)
            cell_codes.append(cell_counter)
            cell_code_lines.append(cell_code_counter)
            cell_code_counter += 1

        if line.startswith('# In['):
            cell_counter += 1
            cell_code_counter = 0

    code = '\n'.join(code_list)

    # Create temporal file with code, run pylint on it
    temp_name = os.path.splitext(path)[0] + '.py'
    with open(temp_name, 'w') as temp_file:
        temp_file.write(code)

    pylint_stdout, pylint_stderr = lint.py_run(temp_name + options, return_std=True)

    errors = pylint_stderr.getvalue()
    report = pylint_stdout.getvalue()
    if errors:
        printer('Errors \n', errors)

    # Create a better repr of pylint report: remove markdown-related warnings
    report_ = []
    for error_line in report.split('\n'):
        if temp_name in error_line:
            error_line = error_line.replace(temp_name, 'nb')
            code_line_number = int(error_line.split(':')[1])
            code_line = code_list[code_line_number - 1]

            # Ignore markdown and comments
            if ignore_comments and code_line.startswith('#'):
                continue

            # Ignore codes
            if sum(code in error_line for code in ignore_codes):
                continue

            # Create report message
            cell_number = cell_codes[code_line_number - 1]
            cell_code_number = cell_code_lines[code_line_number - 1] - 1
            error_code = error_line[error_line.find('(')+1 : error_line.find('(')+6]
            error_msg = error_line[error_line.find(')')+2:]

            report_msg = f'Cell {cell_number}, line {cell_code_number}, error code {error_code}:'
            report_msg += f'\nPylint message: {error_msg}\nCode line   ::: {code_line}\n'

            report_.append(report_msg)

        if 'rated' in error_line:
            report_.insert(0, error_line.strip(' '))
            report_.insert(1, '-' * (len(error_line) - 1))
            report_.insert(2, '')

    printer('\n'.join(report_))

    # Cleanup
    if not keep_script:
        os.remove(temp_name)

    if return_report:
        return '\n'.join(report_)
    return 0


def get_available_gpus(n=1, min_free_memory=0.9, max_processes=2, verbose=False, raise_error=False):
    """ Select `n` gpus from available and free devices.

    Parameters
    ----------
    n : int, str
        If `max`, then use maximum number of available devices.
        If int, then number of devices to select.
    min_free_memory : float
        Minimum percentage of free memory on a device to consider it free.
    max_processes : int
        Maximum amount of computed processes on a device to consider it free.
    verbose : bool
        Whether to show individual device information.
    raise_error : bool
        Whether to raise an exception if not enough devices are available.

    Returns
    -------
    List with indices of availble GPUs
    """
    try:
        import nvidia_smi
    except ImportError as exception:
        raise ImportError('Install Python interface for nvidia_smi') from exception

    nvidia_smi.nvmlInit()
    n_devices = nvidia_smi.nvmlDeviceGetCount()

    available_devices, memory_usage = [], []
    for i in range(n_devices):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        fraction_free = info.free / info.total
        num_processes = len(nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle))

        consider_available = (fraction_free > min_free_memory) & (num_processes <= max_processes)
        if consider_available:
            available_devices.append(i)
            memory_usage.append(fraction_free)

        if verbose:
            print(f'Device {i} | Free memory: {fraction_free:4.2f} | '
                  f'Number of running processes: {num_processes:>2} | Free: {consider_available}')

    if isinstance(n, str) and n.startswith('max'):
        n = len(available_devices)

    if len(available_devices) < n:
        msg = f'Not enough free devices: requested {n}, found {len(available_devices)}'
        if raise_error:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)

    available_devices = np.array(available_devices)[np.argsort(memory_usage)[::-1]]
    return sorted(available_devices[:n])

def get_gpu_free_memory(index):
    """ Get free memory of the gpu"""
    try:
        import nvidia_smi
    except ImportError as exception:
        raise ImportError('Install Python interface for nvidia_smi') from exception

    nvidia_smi.nvmlInit()
    nvidia_smi.nvmlDeviceGetCount()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    return info.free / info.total

def set_gpus(n=1, min_free_memory=0.9, max_processes=2, verbose=False, raise_error=False):
    """ Set the `CUDA_VISIBLE_DEVICES` variable to `n` available devices.

    Parameters
    ----------
    n : int, str
        If `max`, then use maximum number of available devices.
        If int, then number of devices to select.
    min_free_memory : float
        Minimum percentage of free memory on a device to consider it free.
    max_processes : int
        Maximum amount of computed processes on a device to consider it free.
    verbose : bool or int
        Whether to show individual device information.
        If 0 or False, then no information is displayed.
        If 1 or True, then display the value assigned to `CUDA_VISIBLE_DEVICES` variable.
        If 2, then display memory and process information for each device.
    raise_error : bool
        Whether to raise an exception if not enough devices are available.
    """
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        str_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        warnings.warn(f'`CUDA_VISIBLE_DEVICES` is already set to "{str_devices}"!')
        return [int(d) for d in str_devices.split(',')]

    devices = get_available_gpus(n=n, min_free_memory=min_free_memory, max_processes=max_processes,
                                 verbose=(verbose==2), raise_error=raise_error)
    str_devices = ','.join(str(i) for i in devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = str_devices

    newline = "\n" if verbose==2 else ""
    print(f'{newline}`CUDA_VISIBLE_DEVICES` set to "{str_devices}"')
    return devices
