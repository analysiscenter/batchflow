""" Utility functions to work with Jupyter Notebooks. """
import os
import re
import json
import time
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


def extract_traceback(notebook):
    """ Extracts information about an error from the notebook.

    Parameters
    ----------
    notebook: :class:`nbformat.notebooknode.NotebookNode`
        Executed notebook to find an error traceback.

    Returns
    -------
    bool
        Whether the executed notebook has an error traceback.
    int or None
        Number of a cell with a traceback.
        If None, then the notebook doesn't contain an error traceback.
    str
        Error traceback if exists.
    """
    for cell in notebook['cells']:
        # Find a cell output with a traceback and extract the traceback
        outputs = cell.get('outputs', [])

        for output in outputs:
            traceback = output.get('traceback', [])

            if traceback:
                traceback = '\n'.join(traceback)
                return True, cell['execution_count'], traceback

    return False, None, ""

def prepare_notebook_io(notebook, nb_inputs, nb_outputs, out_path_db, nb_inputs_pos):
    """ Prepare notebook for execution: add a cell with inputs and create
    a separate notebook for outputs extraction.

    It is a helper method for the :meth:`run_notebook`.

    Parameters
    ----------
    notebook : :class:`nbformat.notebooknode.NotebookNode`
        Notebook fr execution.
    nb_inputs, nb_outputs, nb_inputs_pos
        Unchanged arguments from :meth:`run_notebook`.
    out_path_db : str
        Path to the notebook shelve database for saving inputs/outputs.

    Returns
    -------
    notebook : :class:`nbformat.notebooknode.NotebookNode`
        Provided notebook with the additional cell with the notebook parameters.
        (if `nb_inputs` is provided).
    output_notebook : :class:`nbformat.notebooknode.NotebookNode` or None
        Additional notebook with saving of required outputs from the `notebook`.
        (if `nb_outputs` is provided).
    """
    import nbformat
    import shelve
    from dill import Pickler, Unpickler
    from textwrap import dedent

    if nb_inputs or nb_outputs:
        # (Re)create a shelve database
        shelve.Pickler = Pickler
        shelve.Unpickler = Unpickler

        with shelve.open(out_path_db) as notebook_db:
            notebook_db.clear()

        # Code for work with the shelve database from notebooks
        code_header = f"""\
                       # Cell inserted during automated execution
                       import os, shelve
                       from dill import Pickler, Unpickler

                       shelve.Pickler = Pickler
                       shelve.Unpickler = Unpickler

                       out_path_db = {repr(out_path_db)}"""

        code_header = dedent(code_header)

    if nb_inputs:
        # Save `nb_inputs` in the shelve database and create a cell in the `notebook`
        # for parameters extraction
        with shelve.open(out_path_db) as notebook_db:
            notebook_db.update(nb_inputs)

        code = """\n
               with shelve.open(out_path_db) as notebook_db:
                   nb_inputs = {**notebook_db}

                   locals().update(nb_inputs)"""

        code = dedent(code)
        code = code_header + code

        notebook['cells'].insert(nb_inputs_pos, nbformat.v4.new_code_cell(code))

    if nb_outputs is not None:
        # Create a notebook to extract outputs from the main notebook
        # The `output_notebook` save locals with preferred names in the shelve database
        if isinstance(nb_outputs, str):
            nb_outputs = [nb_outputs]

        code = f"""
                # Output dict preparation
                output = {{}}
                nb_outputs = {nb_outputs}

                for value_name in nb_outputs:
                    if value_name in locals():
                        output[value_name] = locals()[value_name]

                with shelve.open(out_path_db) as notebook_db:
                    notebook_db['nb_outputs'] = output"""

        code = dedent(code)
        code = code_header + code

        output_notebook = nbformat.v4.new_notebook(metadata=notebook.metadata)
        output_notebook['cells'].append(nbformat.v4.new_code_cell(code))
    else:
        output_notebook = None

    return notebook, output_notebook


def run_notebook(path, nb_inputs=None, nb_outputs=None, nb_inputs_pos=1, timeout=-1, execute_kwargs=None,
                 save_ipynb=True, out_path_ipynb=None, save_html=False, out_path_html=None, suffix='_out',
                 add_timestamp=True, hide_input=False, display_links=True,
                 raise_exception=False, return_nb=False):
    """ Run a notebook and save the execution result.

    Additionally, allows to pass `nb_inputs` arguments, that are used as inputs for notebook execution. Under the hood,
    we place all of them into a separate cell, inserted in the notebook; hence, all of the keys must be valid Python
    names, and values should be valid for re-creating objects.
    Heavily inspired by https://github.com/tritemio/nbrun.

    Also, allows to pass `nb_outputs`, which are used as outputs for notebook execution. Under the hood, we create a
    separate notebook that saves local variables with names from the `nb_outputs`. After that, we extract
    output variables in this method and return them.

    Parameters
    ----------
    path : str
        Path to the notebook to execute.
    nb_inputs : dict, optional
        Inputs for notebook execution. Converted into a cell of variable assignments and inserted
        into the notebook on `nb_inputs_pos` place.
    nb_outputs : str or iterable of str
        List of notebook local variables that return to output.
    nb_inputs_pos : int
        Position to insert the cell with inputs into the notebook.
    timeout : int
        Maximum execution time for each cell. -1 means no constraint.
    execute_kwargs : dict, optional
        Other parameters of `:class:ExecutePreprocessor`.
    save_ipynb : bool
        Whether to save the output .ipynb file.
    out_path_ipynb : str, optional
        Path to save the output .ipynb file. If not provided and `save_ipynb` is set to True, we add `suffix` to `path`.
    save_html : bool
        Whether to convert the executed notebook to .html.
    out_path_html : str, optional
        Path to save the output .html file. If not provided and `save_html` is set to True, we add `suffix` to `path`.
    suffix : str
        Appended to output file names if paths are not explicitly provided.
    add_timestamp : bool
        Whether to add a cell with execution information at the beginning of the executed notebook.
    hide_input : bool
        Whether to hide the code cells in the executed notebook.
    display_links : bool
        Whether to display links to the executed notebook and html at execution.
    raise_exception : bool
        Whether to re-raise exceptions from the notebook.
    return_nb : bool
        Whether to return the notebook object from this function.

    Returns
    -------
    exec_res : dict
        Dictionary with the notebook execution results.
        It provides next information:
            - 'failed' : whether the execution was failed;
            - 'nb_outputs' : the notebook saved outputs;
            - 'failed cell number': an error cell execution number (if exists);
            - 'traceback': traceback message from the notebook (if exists).
    notebook :class:`nbformat.notebooknode.NotebookNode`
        Executed notebook object.
        Note that this output is provided only if `return_nb` is True.
    """
    # pylint: disable=bare-except, lost-exception
    from IPython.display import display, FileLink
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert import HTMLExporter
    import shelve

    # Prepare paths
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path {path} not found.')

    if save_ipynb and out_path_ipynb is None:
        out_path_ipynb = os.path.splitext(path)[0] + suffix + '.ipynb'
    if save_html and out_path_html is None:
        out_path_html = os.path.splitext(path)[0] + suffix + '.html'
    if nb_inputs or nb_outputs:
        # Create a path to save a shelve database for providing inputs/outputs in the notebook
        if out_path_ipynb is not None:
            out_path_db = f"{os.path.splitext(out_path_ipynb)[0]}_db"
        elif out_path_html is not None:
            out_path_db = f"{os.path.splitext(out_path_html)[0]}_db"
        else:
            out_path_db = f"{os.path.splitext(path)[0]}_db"

    # Execution arguments
    working_dir = './'
    execute_kwargs = execute_kwargs or {}
    execute_kwargs.update(timeout=timeout)

    kernel_manager = KernelManager()
    executor = ExecutePreprocessor(**execute_kwargs)

    # Read the master notebook, insert kwargs cell, and create a notebook for outputs extraction
    notebook = nbformat.read(path, as_version=4)

    if hide_input:
        notebook["metadata"].update({"hide_input": True})

    notebook, output_notebook = prepare_notebook_io(notebook=notebook,
                                                    nb_inputs=nb_inputs, nb_outputs=nb_outputs,
                                                    out_path_db=out_path_db, nb_inputs_pos=nb_inputs_pos)

    # Execute the notebook
    start_time = time.time()
    try:
        executor.preprocess(notebook, {'metadata': {'path': working_dir}}, km=kernel_manager)
        exec_failed = False
    except:
        exec_failed = True
        if raise_exception:
            raise
    finally:
        # Save nb_outputs in the shelve db
        if nb_outputs is not None:
            executor.preprocess(output_notebook, {'metadata': {'path': working_dir}}, km=kernel_manager)

        # Check that something gone wrong or not
        failed, error_cell_num, traceback_message = extract_traceback(notebook=notebook)
        failed = failed or exec_failed

        # Prepare execution results: execution state, notebook return values and error info if exists
        exec_res = {'failed': failed}

        if nb_outputs is not None:
            with shelve.open(out_path_db) as notebook_db:
                exec_res['nb_outputs'] = notebook_db['nb_outputs']

        if failed:
            exec_res.update({'failed cell number': error_cell_num, 'traceback': traceback_message})

        # Add execution info
        if add_timestamp:
            timestamp = (f"**Executed:** {time.ctime(start_time)}<br>"
                         f"**Duration:** {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}<br>"
                         f"**Autogenerated from:** [{path}]\n\n---")
            timestamp_cell = nbformat.v4.new_markdown_cell(timestamp)
            notebook['cells'].insert(0, timestamp_cell)

        # Save the executed notebook/HTML to disk
        if save_ipynb:
            with open(out_path_ipynb, 'w', encoding='utf-8') as file:
                nbformat.write(notebook, file)

            if display_links:
                display(FileLink(out_path_ipynb))

        if save_html:
            html_exporter = HTMLExporter()
            body, _ = html_exporter.from_notebook_node(notebook)

            with open(out_path_html, 'w') as f:
                f.write(body)

            if display_links:
                display(FileLink(out_path_html))

        # Remove shelve files if the notebook is successfully executed
        if (nb_inputs or nb_outputs) and not failed:
            for ext in ['bak', 'dat', 'dir']:
                os.remove(out_path_db + '.' + ext)

        if return_nb:
            return (exec_res, notebook)
        return exec_res


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
