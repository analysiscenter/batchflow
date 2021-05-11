import os
import datetime
import csv
from copy import copy, deepcopy
import itertools
import traceback
import multiprocess as mp
import hashlib
import random
import dill
import logging
from collections import OrderedDict

from .. import Config, inbatch_parallel
from ..named_expr import NamedExpression, eval_expr

from .domain import ConfigAlias
from .named_expr import E, O, EC
from .utils import create_logger

class PipelineWrapper:
    def __init__(self, pipeline, mode='generator'):
        if mode not in['generator', 'func', 'execute_for']:
            raise ValueError(f'Unknown PipelineWrapper mode: {mode}')
        self.pipeline = pipeline
        self.mode = mode
        self.config = None

    def __call__(self, config, batch=None):
        if self.config is None:
            self.config = config
            self.pipeline.set_config(self.config)
        if self.mode == 'generator':
            return self.generator()
        elif self.mode == 'func':
            return self.pipeline.run()
        else: # 'execute_for'
            return self.pipeline.execute_for(batch)

    def generator(self, **kwargs):
        self.reset()
        while True:
            yield self.pipeline.next_batch()

    def __getattr__(self, attr):
        return getattr(self.pipeline, attr)

    def reset(self):
        self.pipeline.reset('iter', 'vars')

    def __copy__(self):
        new_unit = PipelineWrapper(self.pipeline + Pipeline(), self.mode)
        return new_unit

class Namespace:
    """ Namespace to use in each experiment in research. Will be initialized at the start of the experiment.

    Parameters
    ----------
    namespace : class
        class which represents namespace.
    name : str
        name of the namespace to use in research.
    """
    def __init__(self, name, namespace, root=False, **kwargs):
        self.name = name
        self.namespace = namespace
        self.root = root
        self.kwargs = kwargs

    def __call__(self, experiment, **kwargs):
        kwargs = {**self.kwargs, **kwargs}
        kwargs = eval_expr(kwargs, experiment=experiment)
        return self.namespace(**kwargs)

class ExecutableUnit:
    def __init__(self, name, func=None, generator=None, root=False, iterations_to_execute=None, args=None, **kwargs):
        self.name = name
        self.callable = func
        self.generator = generator

        self.root = root
        self.iterations_to_execute = iterations_to_execute or 1

        if isinstance(self.iterations_to_execute, (int, str)):
            self.iterations_to_execute = [self.iterations_to_execute]
        self.kwargs = kwargs
        self.args = [] if args is None else args
        self._output = None
        self._iterator = None

    def set_unit(self, config, experiment):
        self.config = config
        self.experiment = experiment

    def get_method(self):
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        if isinstance(src, (tuple, list)):
            setattr(self, attr, getattr(self.experiment.instances[src[0]], src[1]))

    def __call__(self, iteration, n_iters, last=False):
        if self.must_execute(iteration, n_iters, last):
            self.get_method()
            self.iteration = iteration
            args = eval_expr(self.args, experiment=self.experiment)
            kwargs = eval_expr(self.kwargs, experiment=self.experiment)
            if self.callable is not None:
                self._output = self.callable(*args, **kwargs)
            else:
                if self._iterator is None:
                    self._iterator = self.generator(*args, **kwargs)
                self._output = next(self._iterator)
            return self._output

    def must_execute(self, iteration, n_iters=None, last=False):
        # TODO
        if last and 'last' in self.iterations_to_execute:
            return True

        frequencies = (item for item in self.iterations_to_execute if isinstance(item, int) and item > 0)
        iterations = (int(item[1:]) for item in self.iterations_to_execute if isinstance(item, str) and item != 'last')

        it_ok = iteration in iterations
        freq_ok = any((iteration+1) % item == 0 for item in frequencies)

        if n_iters is None:
            return it_ok or freq_ok

        return (iteration + 1 == n_iters and 'last' in self.iterations_to_execute) or it_ok or freq_ok

    def __copy__(self):
        new_unit = ExecutableUnit(self.name)
        for attr in ['callable', 'generator', 'root']:
            setattr(new_unit, attr, copy(getattr(self, attr)))

        new_unit.iterations_to_execute = copy(self.iterations_to_execute)
        new_unit.args = copy(self.args)
        new_unit.kwargs = copy(self.kwargs)
        return new_unit

    def __getattr__(self, key):
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        return getattr(src, key)

    def __getiten__(self, key):
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        return src[key]

class Experiment:
    def __init__(self, namespaces=None, actions=None):
        if namespaces is not None:
             self.namespaces = OrderedDict(namespaces)
        else:
            self.namespaces = OrderedDict()
        if actions is None:
            self.actions = OrderedDict() # unit_name : (namespace_name, attr_name)
        else:
            self.actions = actions

        self.state = ['_is_alive', '_exception_raised', 'iteration'] # TODO: explain what is it

        self._is_alive = True
        self._exception_raised = False
        self.last = False
        self.outputs = dict()
        self.results = OrderedDict()
        self.has_dump = False

    @property
    def is_alive(self):
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value):
        self._is_alive = self._is_alive and value

    @property
    def exception_raised(self):
        return self._exception_raised

    @exception_raised.setter
    def exception_raised(self, value):
        self._exception_raised = self._exception_raised or value

    def add_instance(self, name, namespace, root=False, **kwargs):
        self.namespaces[name] = Namespace(name, namespace, root, **kwargs)
        def _create_instance(experiments, item_name):
            if not isinstance(experiments, list):
                experiments = [experiments]
            instance = experiments[0].namespaces[item_name](experiments[0])
            for e in experiments:
                e.instances[item_name] = instance
        self.add_callable(f'init_{name}', _create_instance, experiments=E(all=root),
                          root=root, item_name=name, iterations_to_execute="%0")
        return self

    def add_executable_unit(self, name, src=None, mode='func', args=None, iterations_to_execute=None, save_to=None, **kwargs):
        if src is None:
            kwargs[mode] = self.parse_name(name)
        else:
            kwargs[mode] = src
        self.actions[name] = ExecutableUnit(name=name, args=args, iterations_to_execute=iterations_to_execute, **kwargs)
        if save_to is not None:
            self.save(src=O(name), dst=save_to, iterations_to_execute=iterations_to_execute, copy=False)
        return self

    def add_callable(self, name, func=None, args=None, **kwargs):
        return self.add_executable_unit(name, src=func, mode='func', args=args, **kwargs)

    def add_generator(self, name, generator=None, args=None, **kwargs):
        return self.add_executable_unit(name, src=generator, mode='generator', args=args, **kwargs)

    def add_pipeline(self, name, root_pipeline, branch_pipeline=None, run=False, **kwargs):
        if branch_pipeline is None:
            mode = 'func' if run else 'generator'
            pipeline = PipelineWrapper(root_pipeline, mode=mode)
            self.add_executable_unit(name, src=pipeline, mode=mode, config=EC(), **kwargs)
        else:
            root_pipeline = PipelineWrapper(root_pipeline, mode='generator')
            branch_pipeline = PipelineWrapper(branch_pipeline, mode='execute_for')

            self.add_generator(f'{name}_root', generator=root_pipeline, config=EC(), **kwargs)
            self.add_callable(f'{name}_branch', func=branch_pipeline, config=EC(), batch=O(f'{name}_root'), **kwargs)
        return self

    def save(self, src, dst, iterations_to_execute=None, copy=False):
        def _save_results(_src, _dst, experiment, copy): #TODO: test does copy work
            previous_values = experiment.results.get(_dst, OrderedDict())
            previous_values[experiment.iteration] = deepcopy(_src) if copy else _src
            experiment.results[_dst] = previous_values
        name = self.add_postfix('save_results')
        self.add_callable(name, _save_results,
                          iterations_to_execute=iterations_to_execute,
                          _src=src, _dst=dst, experiment=E(), copy=copy)
        return self

    def dump(self, variable=None, iterations_to_execute=['last']):
        self.has_dump = True

        def _dump_results(variable, experiment):
            if experiment.dump_results:
                values = experiment.results[variable] if variable is not None else experiment.results
                iteration = experiment.iteration
                variable_path = os.path.join(experiment.full_path, variable or 'all_results')
                if not os.path.exists(variable_path):
                    os.makedirs(variable_path)
                with open(os.path.join(variable_path, str(iteration)), 'wb') as file:
                    dill.dump(values, file)
                if variable is None:
                    experiment.results = OrderedDict() # TODO: does it removed?
                else:
                    del experiment.results[variable]

        name = self.add_postfix('dump_results')
        self.add_callable(name, _dump_results,
                          iterations_to_execute=iterations_to_execute,
                          variable=variable, experiment=E())
        return self

    def add_postfix(self, name):
        return name + '_' + str(sum([item.startswith(name) for item in self.actions]))

    def parse_name(self, name):
        if '.' not in name:
            raise ValueError('`func` parameter must be provided or name must be "namespace_name.unit_name"')
        name_components = name.split('.')
        if len(name_components) > 2:
            raise ValueError(f'name must be "namespace_name.unit_name" but {name} were given')
        return name_components

    def copy(self):
        namespaces = copy(self.namespaces)
        actions = OrderedDict([(name, copy(unit)) for name, unit in self.actions.items()])
        new_experiment = Experiment(namespaces=namespaces, actions=actions)
        new_experiment.has_dump = self.has_dump
        return new_experiment

    def copy_state(self, src):
        for attr in self.state:
            setattr(self, attr, getattr(src, attr))

    def __getitem__(self, key):
        return self.actions[key]

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        _ = memo
        return self.copy()

    def generate_id(self, alias):
        self.id = hashlib.md5(alias.encode('utf-8')).hexdigest()[:8] + str(random.getrandbits(16))

    def create_folder(self):
        # make dir using path from self.meta
        if self.dump_results:
            self.experiment_path = os.path.join('results', self.id)
            self.full_path = os.path.join(self.executor.folder, self.experiment_path)
            if not os.path.exists(self.full_path):
                os.makedirs(self.full_path)

    def create_instances(self, config, executor=None):
        self.config = config.config()
        self.executor = executor
        self.research = executor.research
        self.dump_results = self.research.dump_results
        self.generate_id(config.alias(as_string=True))
        self.create_folder()
        self.instances = OrderedDict()

        root_experiment = executor.experiments[0] if len(executor.experiments) > 0 else None

        for name in self.actions:
            if self.actions[name].root and root_experiment is not None:
                self.actions[name] = root_experiment.actions[name]
            else:
                self.actions[name].set_unit(config=config, experiment=self)

    def create_logger(self):
        name = f"{self.executor.research.name}.{self.id}"
        path = os.path.join(self.full_path, 'experiment.log') if self.dump_results else None
        loglevel = getattr(logging, self.research.loglevel.upper())

        self.logger = create_logger(name, path, loglevel)

    # def close_logger(self):
    #     self.logger.removeHandler(self.logger.handlers[0])

    def call(self, name, iteration, n_iters=None):
        if self.is_alive:
            self.last = self.last or (iteration + 1 == n_iters)
            self.iteration = iteration

            self.logger.debug(f"Execute '{name}' [{iteration}/{n_iters}]")
            # self.research.monitor.start_execution(name, self)
            try:
                # self.info(name, 'call')
                self.outputs[name] = self.actions[name](iteration, n_iters, last=self.last)
            except Exception as e:
                self.exception_raised = True
                self.last = True
                if isinstance(e, StopIteration):
                    self.logger.info(f"Stop '{name}' [{iteration}/{n_iters}]")
                    self.research.monitor.stop_iteration(name, self)
                else:
                    self.exception = e
                    ex_traceback = e.__traceback__
                    msg = ''.join(traceback.format_exception(e.__class__, e, ex_traceback))
                    self.logger.error(f"Fail '{name}' [{iteration}/{n_iters}]: Exception\n{msg}")
                    self.research.monitor.fail_execution(name, self)
            else:
                self.research.monitor.finish_execution(name, self)
            if self.exception_raised and (list(self.actions.keys())[-1] == name):
                self.is_alive = False

    def info(self, name, msg):
        print(f'Experiment: {self.id}, it: {self.iteration}, name: {name} :: {msg}')

class Executor:
    def __init__(self, experiment, research=None, configs=None, executor_config=None, branches_configs=None,
                 target='threads', n_iters=None, task_name=None, dump_results=False, folder=None):
        if configs is None:
            if branches_configs is None:
                self.n_branches = 1
            else:
                self.n_branches = len(branches_configs)
        else:
            if branches_configs is not None and len(configs) != len(branches_configs):
                raise ValueError('`configs` and `branches_configs` must be of the same length.')
            else:
                self.n_branches = len(configs)

        self.configs = configs or [Config() for _ in range(self.n_branches)]
        self.executor_config = Config(executor_config or dict())
        self.branches_configs = branches_configs or [Config() for _ in range(self.n_branches)]
        self.branches_configs = [Config(config) for config in self.branches_configs]
        self.n_iters = n_iters
        self.research = research
        self.experiment_template = experiment
        self.task_name = task_name or 'Task'

        self.folder = self.research.name if self.research is not None else (folder or '.')

        self.create_experiments()

        self.parallel_call = inbatch_parallel(init=self._parallel_init_call, target=target, _use_self=False)(self._parallel_call)

    def create_experiments(self):
        self.experiments = []
        for config, branch_config in zip(self.configs, self.branches_configs):
            config = ConfigAlias(config) + ConfigAlias(branch_config) + ConfigAlias(self.executor_config)
            experiment = self.experiment_template.copy()
            experiment.create_instances(config, self)
            experiment.create_logger()
            self.experiments.append(experiment)

    def run(self, worker):
        self.worker = worker
        self.pid = os.getpid()

        iterations = range(self.n_iters) if self.n_iters else itertools.count()
        for iteration in iterations:
            for unit_name, unit in self.experiment_template.actions.items():
                if unit.root:
                    self.call_root(iteration, unit_name)
                else:
                    self.parallel_call(iteration, unit_name)
            if not any([experiment.is_alive for experiment in self.experiments]):
                break
        self.send_results()

    def _parallel_call(self, experiment, iteration, unit_name):
        """ Parallel call of the unit 'name' """
        experiment.call(unit_name, iteration, self.n_iters)

    def _parallel_init_call(self, iteration, unit_name):
        _ = iteration, unit_name
        return self.experiments

    def call_root(self, iteration, unit_name):
        self.experiments[0].call(unit_name, iteration, self.n_iters)
        for experiment in self.experiments[1:]:
            experiment.outputs[unit_name] = self.experiments[0].outputs[unit_name]
            experiment.copy_state(self.experiments[0])

    def send_results(self):
        if self.research is not None:
            for experiment in self.experiments:
                self.research.results.put(experiment.id, experiment.results)