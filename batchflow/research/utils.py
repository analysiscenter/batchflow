""" Auxilary functions """
import os
import glob
import shutil
import logging
import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
import dill
from tqdm import tqdm_notebook
import numpy as np

def to_list(value):
    return value if isinstance(value, list) else [value]

def count_startswith(seq, name):
    return sum(1 for item in seq if item.startswith(name))

def get_metrics(pipeline, metrics_var, metrics_name, *args, agg='mean', **kwargs):
    """ Function to evaluate metrics. """
    metrics_name = metrics_name if isinstance(metrics_name, list) else [metrics_name]
    metrics = pipeline.get_variable(metrics_var).evaluate(metrics_name, *args, agg=agg, **kwargs)
    values = [metrics[name] for name in metrics_name]
    if len(values) == 1:
        return values[0]
    return values

def transform_research_results(research_name, new_name=None, bar=True):
    """ Transform old research format to the new. Only results will be transformed, old research can not be
    transformed to load. """
    # Copy research if needed
    if new_name is not None:
        shutil.copytree(research_name, new_name)
        research_name = new_name

    # Move configs from separate folder to experiment folders
    configs = {}
    for config in glob.glob(f'{research_name}/configs/*'):
        for experiment_folder in glob.glob(f'{research_name}/results/{glob.escape(os.path.basename(config))}/*'):
            exp_id = os.path.basename(experiment_folder)
            configs[exp_id] = os.path.basename(config)

    for exp_id, config in configs.items():
        src = f'{research_name}/configs/{config}'
        dst = f'{research_name}/results/{config}/{exp_id}/config.dill'
        with open(src, 'rb') as f:
            content = dill.load(f) # content is a ConfigAlias instance
            content['updates'] = content['update'] # Rename column for the new format
            content.pop_config('update')
            content['device'] = None # Add column
        with open(dst, 'wb') as f:
            dill.dump(content, f)
        with open(f'{research_name}/results/{config}/{exp_id}/config.json', 'w') as f:
            json.dump(jsonify(content.config().config), f)

    # Remove folder with configs
    shutil.rmtree(f'{research_name}/configs')

    # Remove one nested level
    initial_results = glob.glob(f'{research_name}/results/*')
    for exp_path in initial_results:
        for path in os.listdir(exp_path):
            src = os.path.join(exp_path, path)
            dst = os.path.join(os.path.dirname(exp_path), path)
            shutil.move(src, dst)
    for path in initial_results:
        shutil.rmtree(path)

    # Rename 'results' folder to 'experiments'
    shutil.move(f'{research_name}/results', f'{research_name}/experiments')

    # Move files from experiment folder to subfodlers
    for results_file in tqdm_notebook(glob.glob(f'{research_name}/experiments/*/*'), disable=(not bar)):
        filename = os.path.basename(results_file)
        content = get_content(results_file)
        if content is not None:
            content.pop('sample_index')
            iterations = content.pop('iteration')

            unit_name, iteration_in_name = filename.split('_')
            iteration_in_name = int(iteration_in_name) - 1
            dirname = os.path.dirname(results_file)
            for var in content:
                new_dict = OrderedDict()
                for i, val in zip(iterations, content[var]):
                    new_dict[i] = val
                folder_for_var = f'{dirname}/results/{unit_name}_{var}'
                if not os.path.exists(folder_for_var):
                    os.makedirs(folder_for_var)
                dst = f'{folder_for_var}/{iteration_in_name}'
                with open(dst, 'wb') as f:
                    dill.dump(new_dict, f)
            os.remove(results_file)

def get_content(path):
    """ Open research results file (if it is). """
    filename = os.path.basename(path)
    if len(filename.split('_')) != 2:
        return None
    _, iteration_in_name = filename.split('_')
    if not iteration_in_name.isdigit():
        return None
    try:
        with open(path, 'rb') as f:
            content = dill.load(f)
    except dill.UnpicklingError:
        return None
    if not isinstance(content, dict):
        return None
    if 'sample_index' not in content or 'iteration' not in content:
        return None
    return content

def jsonify(src):
    """ Transform np.arrays to lists to JSON serialize. """
    src = deepcopy(src)
    for key, value in src.items():
        if isinstance(value, np.ndarray):
            src[key] = value.tolist()
    return src

def create_logger(name, path=None, loglevel='info'):
    """ Create logger. """
    loglevel = getattr(logging, loglevel.upper())
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)

    if path is not None:
        handler = logging.FileHandler(path)
    else:
        handler = logging.StreamHandler() #TODO: filter outputs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    handler.setLevel(loglevel)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def must_execute(iteration, when, n_iters=None, last=False):
    """ Returns does unit must be executed for the current iteration. """
    if last and 'last' in when:
        return True

    frequencies = (item for item in when if isinstance(item, int) and item > 0)
    iterations = (int(item[1:]) for item in when if isinstance(item, str) and item != 'last')

    it_ok = iteration in iterations
    freq_ok = any((iteration+1) % item == 0 for item in frequencies)

    if n_iters is None:
        return it_ok or freq_ok

    return (iteration + 1 == n_iters and 'last' in when) or it_ok or freq_ok

def parse_name(name):
    """ Parse name of the form 'namespace_name.unit_name' into tuple ('namespace_name', 'unit_name'). """
    if '.' not in name:
        raise ValueError('`func` parameter must be provided or name must be "namespace_name.unit_name"')
    name_components = name.split('.')
    if len(name_components) > 2:
        raise ValueError(f'name must be "namespace_name.unit_name" but {name} were given')
    return name_components

def generate_id(config, random):
    """ Generate id for experiment. """
    name = hashlib.md5(config.alias(as_string=True).encode('utf-8')).hexdigest()[:8]
    name += ''.join(str(i) for i in random.integers(10, size=8))
    return name

def explicit_call(method, name, experiment):
    """ Add unit into research by explicit call in research-pipeline. """
    def _method(*args, **kwargs):
        return experiment.add_executable_unit(name, src=method, args=args, **kwargs)
    return _method
