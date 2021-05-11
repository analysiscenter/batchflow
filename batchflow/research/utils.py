""" Auxilary functions """
import os
import glob
import shutil
import logging


def to_list(value):
    return value if isinstance(value, list) else [value]

def count_startswith(seq, name):
    return sum(1 for item in seq if item.startswith(name))

def get_metrics(pipeline, metrics_var, metrics_name, *args, agg='mean', **kwargs):
    """ Function to evaluate metrics """
    metrics_name = metrics_name if isinstance(metrics_name, list) else [metrics_name]
    metrics = pipeline.get_variable(metrics_var).evaluate(metrics_name, *args, agg=agg, **kwargs)
    values = [metrics[name] for name in metrics_name]
    if len(values) == 1:
        return values[0]
    return values

def transform_research_results(research_name):
    """ Transform old research format (with additional nesting level) to the new. """
    configs = {}
    for config in glob.glob(f'{research_name}/configs/*'):
        for experiment_folder in glob.glob(f'{research_name}/results/{os.path.basename(config)}/*'):
            exp_id = os.path.basename(experiment_folder)
            configs[exp_id] = config
    for exp_id, path in configs.items():
        dst = os.path.join(os.path.dirname(path), exp_id)
        shutil.move(path, dst)

    initial_results = glob.glob(f'{research_name}/results/*')
    for exp_path in initial_results:
        for path in os.listdir(exp_path):
            src = os.path.join(exp_path, path)
            dst = os.path.join(os.path.dirname(exp_path), path)
            shutil.move(src, dst)
    for path in initial_results:
        shutil.rmtree(path)

def create_logger(name, path, loglevel):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)

    if path is not None:
        fh = logging.FileHandler(path)
    else:
        fh = logging.StreamHandler() #TODO: filter outputs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
