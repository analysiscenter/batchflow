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

from .. import Config, inbatch_parallel, Pipeline
from ..named_expr import NamedExpression, eval_expr

from .domain import ConfigAlias
from .named_expr import E, O, EC
from .utils import create_logger, to_list

class PipelineWrapper:
    """ Make callable or generator from `batchflow.pipeline`.

    Parameters
    ----------
    pipeline : Pipeline
        pipeline defined with `.run_later`
    mode : 'generator', 'func' or 'execute_for', optional
        the way to use pipeline, by default 'generator':
            - 'generator': pipeline will be used as generator of batches
            - 'func': execute pipeline with `.run`
            - 'execute_for': execute pipeline with `.execute_for` with batch
    """
    def __init__(self, pipeline, mode='generator'):
        if mode not in['generator', 'func', 'execute_for']:
            raise ValueError(f'Unknown PipelineWrapper mode: {mode}')
        self.pipeline = pipeline
        self.mode = mode
        self.config = None

    def __call__(self, config, batch=None):
        """ Execute pipeline.

        Parameters
        ----------
        config : Config
            `Config` for pipeline, defined at the first pipeline execution.
        batch : Batch, optional
            `Batch` to use with `execute_for` method, by default None

        Returns
        -------
        generator, Pipeline or Batch
            return depends on the mode:
                - 'generator': generator
                - 'func': Pipeline itself
                - 'execute_for': processed batch
        """
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
        """ Generator returns batches from pipeline. Generator will stop when StopIteration will be raised. """
        self.reset()
        while True:
            yield self.pipeline.next_batch()

    def __getattr__(self, attr):
        return getattr(self.pipeline, attr)

    def reset(self):
        self.pipeline.reset('iter', 'vars')

    def __copy__(self):
        """ Create copy of the pipeline with the same mode. """
        new_unit = PipelineWrapper(self.pipeline + Pipeline(), self.mode)
        return new_unit

class Namespace:
    """ Namespace to use in each experiment in research. Will be initialized at the start of the experiment execution.

    Parameters
    ----------
    name : str
        name of the namespace to use in research.
    namespace : class
        class which represents namespace.
    root : bool, optional
        does namespace is the same for all branches or not, by default False
    kwargs : dict
        kwargs for namespace initialization
    """
    def __init__(self, name, namespace, root=False, **kwargs):
        self.name = name
        self.namespace = namespace
        self.root = root
        self.kwargs = kwargs

    def __call__(self, experiment, **kwargs):
        """ Create instance of the namespace. """
        kwargs = {**self.kwargs, **kwargs}
        kwargs = eval_expr(kwargs, experiment=experiment)
        return self.namespace(**kwargs)

class ExecutableUnit:
    """ Class to represent callables and generators executed in experiment.

    Parameters
    ----------
    name : str

    func : callable or tuple of str, optional
        callable itself or tuple consists of namespace name and its attribute to call, by default None. `func` and
        `generator` can't be defined simultaneously.
    generator : generator, optional
        generator itself or tuple consists of namespace name and its attribute to call, by default None. `func` and
        `generator` can't be defined simultaneously.
    root : bool, optional
        [description], by default False
    iterations_to_execute : str, int or list of ints, optional
        iterations of the experiment to execute unit, by default 1.
            - If `'last'`, unit will be executed just at last iteration (if `iteration + 1 == n_iters` or `StopIteration`
              was raised).
            - If positive int, pipeline will be executed each `iterations_to_execute` iterations.
            - If str, must be `'#{it}'` or `'last'` where it is int, the pipeline will be executed at this
              iteration (zero-based).
            - If list, must be list of int or str described above.
    args, kwargs : optional
        args and kwargs for unit call, by default None.
    """
    def __init__(self, name, func=None, generator=None, root=False, iterations_to_execute=1, args=None, **kwargs):
        self.name = name
        self.callable = func
        self.generator = generator

        self.root = root
        self.iterations_to_execute = iterations_to_execute

        if isinstance(self.iterations_to_execute, (int, str)):
            self.iterations_to_execute = [self.iterations_to_execute]
        self.kwargs = kwargs
        self.args = [] if args is None else args
        self._output = None # the previous output of the unit.
        self._iterator = None

    def set_unit(self, config, experiment):
        """ Set config and experiment instance for the unit. """
        self.config = config
        self.experiment = experiment

    def get_method(self):
        """ Transform `callable` or `generator` from tuples to instance attributes. """
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        if isinstance(src, (tuple, list)):
            setattr(self, attr, getattr(self.experiment.instances[src[0]], src[1]))

    def __call__(self, iteration, n_iters, last=False):
        """ Call unit: execute callable or get the next item from generator.

        Parameters
        ----------
        iteration : int
            current iteration
        n_iters : int or None
            total number of iterations for the experiment. `None` means that experiment will be executed until
            `StopIteration` for at least one executable unit.
        last : bool, optional
            does it last iteration or not, by default False. `last` is True when StopIteration was raised for one
            of the previously executed units or `iteration + 1 == n_iters` when `n_iters` is not None.

        Returns
        -------
        object
            output of the wrapped unit
        """
        if iteration == 0:
            self.get_method()
        if self.must_execute(iteration, n_iters, last):
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
        """ Returns does unit must be executed for the current iteration. """
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
        """ Create copy of the unit. """
        attrs = ['name', 'callable', 'generator', 'root', 'iterations_to_execute', 'args']
        params = {attr if attr !='callable' else 'func': copy(getattr(self, attr)) for attr in attrs}
        new_unit = ExecutableUnit(**params, **copy(self.kwargs))
        return new_unit

    def __getattr__(self, key):
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        return getattr(src, key)

    def __getitem__(self, key):
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

        self._is_alive = True # should experiment be executed or not. Becomes False when Exceptions was raised and all
                              # units for these iterations were executed.
        self._exception_raised = False # was an exception raised or not
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

    def add_executable_unit(self, name, src=None, mode='func', args=None, iterations_to_execute=1, save_to=None, **kwargs):
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

    def save(self, src, dst, iterations_to_execute=1, copy=False):
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
                variables_to_dump = list(experiment.results.keys()) if variable is None else to_list(variable)
                for var in variables_to_dump:
                    values = experiment.results[var]
                    iteration = experiment.iteration
                    variable_path = os.path.join(experiment.full_path, var)
                    if not os.path.exists(variable_path):
                        os.makedirs(variable_path)
                    with open(os.path.join(variable_path, str(iteration)), 'wb') as file:
                        dill.dump(values, file)
                    experiment.results[var] = OrderedDict()

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
        if self.dump_results:
            self.experiment_path = os.path.join('results', self.id)
            self.full_path = os.path.join(self.name, self.experiment_path)
            if not os.path.exists(self.full_path):
                os.makedirs(self.full_path)

    def create_instances(self, index, config, executor=None):
        self.index = index
        self.config_alias = config
        self.config = config.config()
        self.executor = executor
        self.research = executor.research

        # Get attributes from research or kwargs of executor
        for attr, default in [('dump_results', False), ('loglevel', 'debug'), ('name', 'executor'), ('monitor', None)]:
            if self.research:
                value = getattr(self.research, attr)
            else:
                value = self.executor.kwargs.get(attr, default)
            setattr(self, attr, value)

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
        name = f"{self.name}." if self.name else ""
        name += f"{self.id}"
        path = os.path.join(self.full_path, 'experiment.log') if self.dump_results else None

        self.logger = create_logger(name, path, self.loglevel)

    # def close_logger(self):
    #     self.logger.removeHandler(self.logger.handlers[0])

    def call(self, name, iteration, n_iters=None):
        if self.is_alive:
            self.last = self.last or (iteration + 1 == n_iters)
            self.iteration = iteration

            self.logger.debug(f"Execute '{name}' [{iteration}/{n_iters}]")
            # self.research.monitor.start_execution(name, self)
            try:
                self.outputs[name] = self.actions[name](iteration, n_iters, last=self.last)
            except Exception as e:
                self.exception_raised = True
                self.last = True
                if isinstance(e, StopIteration):
                    self.logger.info(f"Stop '{name}' [{iteration}/{n_iters}]")
                    if self.monitor:
                        self.monitor.stop_iteration(name, self)
                else:
                    self.exception = e
                    ex_traceback = e.__traceback__
                    msg = ''.join(traceback.format_exception(e.__class__, e, ex_traceback))
                    self.logger.error(f"Fail '{name}' [{iteration}/{n_iters}]: Exception\n{msg}")
                    if self.monitor:
                        self.monitor.fail_execution(name, self)
            else:
                if self.monitor:
                    self.monitor.finish_execution(name, self)
            if self.exception_raised and (list(self.actions.keys())[-1] == name):
                self.is_alive = False

class Executor:
    def __init__(self, experiment, research=None, configs=None, executor_config=None, branches_configs=None,
                 target='threads', n_iters=None, task_name=None, **kwargs):
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

        self.kwargs = kwargs

        self.create_experiments()

        self.parallel_call = inbatch_parallel(init=self._parallel_init_call, target=target, _use_self=False)(self._parallel_call)

    def create_experiments(self):
        self.experiments = []
        for index, (config, branch_config) in enumerate(zip(self.configs, self.branches_configs)):
            config = ConfigAlias(config) + ConfigAlias(branch_config) + ConfigAlias(self.executor_config)
            experiment = self.experiment_template.copy()
            experiment.create_instances(index, config, self)
            experiment.create_logger()
            self.experiments.append(experiment)

    def run(self, worker=None):
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
        # TODO: experiment can be alive if error was in the branch after all roots
        self.experiments[0].call(unit_name, iteration, self.n_iters)
        for experiment in self.experiments[1:]:
            experiment.outputs[unit_name] = self.experiments[0].outputs[unit_name]
            for attr in ['_is_alive', '_exception_raised', 'iteration']:
                setattr(experiment, attr, getattr(self.experiments[0], attr))

    def send_results(self):
        if self.research is not None:
            for experiment in self.experiments:
                experiment.results['config'] = experiment.config_alias
                self.research.results.put(experiment.id, experiment.results)
