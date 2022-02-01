""" Utility functions for running Jupyter Notebooks."""
import os
import time


def run_notebook(path, inputs=None, outputs=None, inputs_pos=1, out_path_db=None, execute_kwargs=None,
                 out_path_ipynb=None, out_path_html=None, add_timestamp=True, hide_code_cells=False, display_links=True,
                 raise_exception=False, return_notebook=False):
    """ Run a notebook and save the execution result.

    Allows to pass `inputs` arguments, that are used as inputs for notebook execution. Under the hood,
    we place all of them into a separate cell, inserted in the notebook; hence, all of the keys must be valid Python
    names, and values should be valid for re-creating objects.
    Heavily inspired by https://github.com/tritemio/nbrun.

    Also, allows to pass `outputs` parameter, which is a list of local variables that you need to return from
    the executed notebook. Under the hood, we insert a cell that saves local variables with names from the `outputs`
    in the shelve db. If the notebook failed, then the cell is executed directly. After that, we extract output
    variables in this method and return them.

    Parameters
    ----------
    path : str
        Path to the notebook to execute.
    inputs : dict, optional
        Inputs for notebook execution. Converted into a cell of variable assignments and inserted
        into the notebook on `inputs_pos` place.
    outputs : str or iterable of str
        List of notebook local variables that return to output.
    inputs_pos : int
        Position to insert the cell with inputs into the notebook.
    out_path_db : str, optional
        Path to save the shelve database files. There is no need in files extension.
        If None and `inputs` or `outputs` are provided, than `out_path_db` is created from `out_path_ipynb`.
    execute_kwargs : dict, optional
        Other parameters of `:class:ExecutePreprocessor`.
    out_path_ipynb : str, optional
        Path to save the output .ipynb file.
    out_path_html : str, optional
        Path to save the output .html file.
    add_timestamp : bool
        Whether to add a cell with execution information at the beginning of the executed notebook.
    hide_code_cells : bool
        Whether to hide the code cells in the executed notebook.
    display_links : bool
        Whether to display links to the executed notebook and html at execution.
    raise_exception : bool
        Whether to re-raise exceptions from the notebook.
    return_notebook : bool
        Whether to return the notebook object from this function.

    Returns
    -------
    exec_res : dict
        Dictionary with the notebook execution results.
        It provides next information:
        - 'failed' : bool
           Whether the execution was failed.
        - 'outputs' : dict
          The notebook saved outputs.
        - 'failed cell number': int
          An error cell execution number (if exists).
        - 'traceback' : str
          Traceback message from the notebook (if exists).
        - 'notebook' : :class:`nbformat.notebooknode.NotebookNode`
          Executed notebook object.
          Note that this output is provided only if `return_notebook` is True.
    """
    # pylint: disable=bare-except, lost-exception
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbconvert.preprocessors import ExecutePreprocessor
    import shelve
    from dill import Pickler, Unpickler
    from textwrap import dedent

    if inputs or outputs:
        # Set `out_path_db` value
        if out_path_db is None:
            if out_path_ipynb:
                out_path_db = os.path.splitext(out_path_ipynb)[0] + '_db'
            else:
                error_message = """\
                                Invalid value for `out_path_db` argument. If `inputs` or `outputs` are provided,
                                then you need to provide `out_path_db` or `out_path_ipynb` arguments."""
                error_message = dedent(error_message)
                raise ValueError(error_message)

        # (Re)create a shelve database
        shelve.Pickler = Pickler
        shelve.Unpickler = Unpickler

        with shelve.open(out_path_db) as notebook_db:
            notebook_db.clear()

    if isinstance(outputs, str):
        outputs = [outputs]

    working_dir = './'
    execute_kwargs = execute_kwargs or {'timeout': -1}
    executor = ExecutePreprocessor(**execute_kwargs)
    kernel_manager = KernelManager()

    # Notebook preparation:
    # Read the notebook, insert a cell with inputs, insert another cell for outputs extraction
    notebook = nbformat.read(path, as_version=4)

    if hide_code_cells:
        notebook["metadata"].update({"hide_input": True})

    if inputs or outputs:
        # Code for work with the shelve database from the notebook
        comment_header = "# Cell inserted during automated execution\n"
        code_header = f"""\
                       import os, shelve
                       from dill import Pickler, Unpickler

                       shelve.Pickler = Pickler
                       shelve.Unpickler = Unpickler

                       out_path_db = {repr(out_path_db)}"""

        code_header = dedent(code_header)

    if inputs:
        # Save `inputs` in the shelve database and create a cell in the notebook
        # for parameters extraction
        with shelve.open(out_path_db) as notebook_db:
            notebook_db.update(inputs)

        code = """\n
                # Inputs loading
                with shelve.open(out_path_db) as notebook_db:
                    inputs = {**notebook_db}

                    locals().update(inputs)"""

        code = dedent(code)
        code = comment_header + code_header + code

        notebook['cells'].insert(inputs_pos, nbformat.v4.new_code_cell(code))

    if outputs:
        # Create a cell to extract outputs from the notebook
        # It saves locals from the notebook with preferred names in the shelve database
        # This cell will be executed in error case too
        code = f"""\n
                # Output dict preparation
                output = {{}}
                outputs = {outputs}

                for value_name in outputs:
                    if value_name in locals():
                        output[value_name] = locals()[value_name]

                with shelve.open(out_path_db) as notebook_db:
                    notebook_db['outputs'] = output"""

        code = dedent(code)
        code = comment_header + (code_header if not inputs else "") + code

        output_cell = nbformat.v4.new_code_cell(code)
        notebook['cells'].append(output_cell)

    # Execute the notebook
    start_time = time.time()
    exec_failed = False
    try:
        executor.preprocess(notebook, {'metadata': {'path': working_dir}}, km=kernel_manager)
    except:
        exec_failed = True

        # Save notebook outputs in the shelve db
        if outputs is not None:
            executor.kc = kernel_manager.client() # For compatibility with 5.x.x version
            executor.preprocess_cell(output_cell, {'metadata': {'path': working_dir}}, -1)

        if raise_exception:
            raise
    finally:
        # Check if something went wrong
        failed, error_cell_num, traceback_message = extract_traceback(notebook=notebook)
        failed = failed or exec_failed

        # Prepare execution results: execution state, notebook outputs and error info (if exists)
        if failed:
            exec_res = {'failed': failed, 'failed cell number': error_cell_num, 'traceback': traceback_message}
        else:
            exec_res = {'failed': failed, 'failed cell number': None, 'traceback': ''}

        if outputs is not None:
            with shelve.open(out_path_db) as notebook_db:
                exec_res['outputs'] = notebook_db.get('outputs', {})

        if add_timestamp:
            timestamp = (f"**Executed:** {time.ctime(start_time)}<br>"
                         f"**Duration:** {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}<br>"
                         f"**Autogenerated from:** [{path}]\n\n---")
            timestamp_cell = nbformat.v4.new_markdown_cell(timestamp)
            notebook['cells'].insert(0, timestamp_cell)

        # Save the executed notebook/HTML to disk
        if out_path_ipynb:
            save_notebook(notebook=notebook, out_path_ipynb=out_path_ipynb, display_link=display_links)
        if out_path_html:
            notebook_to_html(notebook=notebook, out_path_html=out_path_html, display_link=display_links)

        # Remove shelve files if the notebook is successfully executed
        if out_path_db and not failed:
            for ext in ['bak', 'dat', 'dir']:
                os.remove(out_path_db + '.' + ext)

        if return_notebook:
            exec_res['notebook'] = notebook
        return exec_res

# Save notebook functions
def save_notebook(notebook, out_path_ipynb, display_link):
    """ Save notebook as ipynb file."""
    import nbformat
    from IPython.display import display, FileLink

    with open(out_path_ipynb, 'w', encoding='utf-8') as file:
        nbformat.write(notebook, file)

    if display_link:
        display(FileLink(out_path_ipynb))

def notebook_to_html(notebook, out_path_html, display_link):
    """ Save notebook as ipynb file."""
    from nbconvert import HTMLExporter
    from IPython.display import display, FileLink

    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(notebook)

    with open(out_path_html, 'w') as f:
        f.write(body)

    if display_link:
        display(FileLink(out_path_html))


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
