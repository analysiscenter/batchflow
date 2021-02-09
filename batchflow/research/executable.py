""" Executable class: wrapper for pipelines and functions. """

import os
import time
from copy import copy, deepcopy
import dill
from ..named_expr import eval_expr
from .. import Config, Pipeline, V, F

class PipelineStopIteration(StopIteration):
    """ Special pipeline StopIteration exception """
    pass

class Executable:
    """ Function or pipeline

    **Attributes**

    function : callable
        is None if `Executable` is a pipeline
    pipeline : Pipeline
        is None if `Executable` is a function
    root_pipeline : Pipeline
        is None if `Executable` is a function or pipeline is not divided into root and branch
    dataset : Dataset or None
        dataset for pipelines
    part : str or None
        part of dataset to use
    result : dict
        current results of the `Executable`. Keys are names of variables (for pipeline)
        or returns (for function) values are lists of variable values
    path : str
        path to the folder where results will be dumped
    exec : int, list of ints or None
    dump : int, list of ints or None
    to_run : bool
    variables : list
        variables (for pipeline) or returns (for function)
    on_root : bool

    args : list

    kwargs : dict()
    """
    def __init__(self):
        self.function = None
        self.pipeline = None
        self.name = None
        self.result = None
        self.execute = None
        self.dump = None
        self.to_run = None
        self.variables = []
        self.root_pipeline = None
        self.on_root = None
        self.args = []
        self.kwargs = dict()
        self.experiment_path = None
        self.research_path = None
        self.config = None
        self.config_from_grid = None
        self.config_from_func = None
        self.logging = None
        self.additional_config = None
        self.action = None
        self.dataset = None

        self.last_update_time = None

        self._function = None

    def add_callable(self, function, *args, name='callable', execute=1, dump='last', returns=None,
                     on_root=False, logging=False, **kwargs):
        """ Add function as an Executable Unit. """
        returns = returns or []

        if not isinstance(returns, list):
            returns = [returns]

        self.name = name
        self.function = function
        self._function = function
        self.execute = execute
        self.dump = dump
        self.variables = returns
        self.args = args
        self.kwargs = kwargs
        self.on_root = on_root
        self.logging = logging

        self.action = {
            'type': 'function',
            'name': name,
            'on_root': on_root
        }

        self._clear_result()
        self._process_iterations()

    def add_pipeline(self, root, name, branch=None, dataset=None, variables=None,
                     execute=1, dump='last', run=False, logging=False, **kwargs):
        """ Add pipeline as an Executable Unit """
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]

        if branch is None:
            pipeline = root
            root_pipeline = None
        else:
            pipeline = branch
            root_pipeline = root

        self.name = name
        self.pipeline = pipeline
        self.root_pipeline = root_pipeline
        self.dataset = dataset
        self.variables = variables
        self.execute = execute
        self.dump = dump
        self.to_run = run
        self.kwargs = kwargs
        self.logging = logging

        self.action = {
            'type': 'pipeline',
            'name': name,
            'root': root is not None,
            'run': run
        }

        self.config = None

        self.additional_config = None

        self._clear_result()
        self._process_iterations()

    def _process_iterations(self):
        if not isinstance(self.execute, list):
            self.execute = [self.execute]
        if not isinstance(self.dump, list):
            self.dump = [self.dump]

    def get_copy(self):
        """ Create copy of unit """
        new_unit = copy(self)
        if self.pipeline is not None:
            new_unit.pipeline += Pipeline()
        new_unit.result = deepcopy(new_unit.result)
        new_unit.variables = copy(new_unit.variables)
        return new_unit

    def set_dataset(self):
        """ Add dataset to root if root exists or create root pipeline on the base of dataset. """
        if self.dataset is not None:
            if self.root_pipeline is not None:
                self.root_pipeline = self.root_pipeline << self.dataset
            else:
                self.pipeline = self.pipeline << self.dataset

    def reset(self, what='iter'):
        """ Reset iterators in pipelines """
        if self.pipeline is not None:
            self.pipeline.reset(what)
        if self.root_pipeline is not None:
            self.root_pipeline.reset(what)

    def _clear_result(self):
        self.result = {var: [] for var in self.variables}
        self.result['iteration'] = []

    def set_config(self, config_from_grid, config_from_func, worker_config, branch_config, import_config):
        """ Set new config for pipeline """
        self.config_from_grid = config_from_grid
        self.config_from_func = config_from_func

        self.config = config_from_grid + config_from_func
        self.additional_config = Config(worker_config) + Config(branch_config) + Config(import_config)
        if self.pipeline is not None:
            self.pipeline.set_config(self.config.config() + self.additional_config)

    def set_research_path(self, path):
        self.research_path = path

    def dump_config(self):
        with open(os.path.join(self.research_path, 'configs', self.config.alias(as_string=True)), 'wb') as file:
            dill.dump(self.config, file)

    def next_batch(self):
        """ Next batch from pipeline """
        if self.pipeline is not None:
            try:
                batch = self.pipeline.next_batch()
            except StopIteration as e:
                raise PipelineStopIteration('{} was stopped'.format(self.name)) from e
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def run(self):
        """ Run pipeline """
        if self.pipeline is not None:
            self.pipeline.reset("iter", "vars")
            self.pipeline.run()
        else:
            raise TypeError("Executable should be pipeline, not a function")

    def reset_root_iter(self):
        """ Reset pipeline iterator """
        if self.root_pipeline is not None:
            self.root_pipeline.reset("iter")
        else:
            raise TypeError("Executable must have root")

    def next_batch_root(self):
        """ Next batch from root pipeline """
        if self.root_pipeline is not None:
            try:
                batch = self.root_pipeline.next_batch()
            except StopIteration as e:
                raise PipelineStopIteration('{} was stopped'.format(self.name)) from e
        else:
            raise TypeError("Executable should have root pipeline")
        return batch

    def execute_for(self, batch):
        """ Execute pipeline for batch from root """
        if self.pipeline is not None:
            batch = self.pipeline.execute_for(batch)
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def _call_pipeline(self, iteration):
        if self.to_run:
            self.run()
        else:
            self.next_batch()
        self.put_result(iteration)

    def _call_function(self, job, iteration, experiment):
        function = eval_expr(self.function, job=job, iteration=iteration, experiment=experiment)
        if callable(function):
            args = eval_expr(self.args, job=job, iteration=iteration, experiment=experiment, path=self.research_path)
            kwargs = eval_expr(self.kwargs, job=job, iteration=iteration, experiment=experiment,
                               path=self.research_path)
            result = function(*args, **kwargs)
        else:
            result = function
        self.put_result(iteration, result)
        return result

    def __call__(self, job, iteration, experiment):
        if self.pipeline is not None:
            self._call_pipeline(iteration)
        else:
            self._call_function(job, iteration, experiment)

    def put_result(self, iteration, result=None):
        """ Put result from pipeline to self.result """
        if len(self.variables) > 0:
            if self.pipeline is not None:
                for variable in self.variables:
                    self.result[variable].append(self.pipeline.get_variable(variable))
            else:
                if len(self.variables) == 1:
                    result = [result]
                for variable, value in zip(self.variables, result):
                    self.result[variable].append(value)
            self.result['iteration'].append(iteration)

    def dump_result(self, task_id, iteration, filename):
        """ Dump pipeline results """
        if len(self.variables) > 0:
            if not os.path.exists(os.path.join(self.research_path, self.experiment_path, task_id)):
                os.makedirs(os.path.join(self.research_path, self.experiment_path, task_id))
            self.result['sample_index'] = [task_id] * len(self.result['iteration'])
            path = os.path.join(self.research_path, self.experiment_path, task_id, filename + '_' + str(iteration))
            with open(path, 'wb') as file:
                dill.dump(self.result, file)
        self._clear_result()

    def create_folder(self):
        """ Create folder if it doesn't exist """
        self.experiment_path = os.path.join('results', self.config.alias(as_string=True))
        try:
            os.makedirs(os.path.join(self.research_path, self.experiment_path))
        except FileExistsError:
            pass

    def action_iteration(self, iteration, n_iters=None, action='execute'):
        """ Returns does Unit should be executed at that iteration """
        rule = self.execute if action == 'execute' else self.dump

        frequencies = (item for item in rule if isinstance(item, int) and item > 0)
        iterations = (int(item[1:]) for item in rule if isinstance(item, str) and item != 'last')

        it_ok = iteration in iterations
        freq_ok = any((iteration+1) % item == 0 for item in frequencies)

        if n_iters is None:
            return it_ok or freq_ok

        return (iteration + 1 == n_iters and 'last' in rule) or it_ok or freq_ok

    def set_shared_value(self, last_update_time):
        """ Set last update time """
        self.last_update_time = last_update_time
        if self.pipeline is not None:
            self.pipeline += (Pipeline()
                              .init_variable('_time', default=last_update_time)
                              .update(V('_time').value, F(time.time))
                              )
