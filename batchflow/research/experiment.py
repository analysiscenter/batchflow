#pylint:disable=logging-fstring-interpolation
""" Experiment and corresponding classes. """

import os
import sys
from copy import copy, deepcopy
import itertools
import traceback
import contextlib
import warnings
from collections import OrderedDict
import time

from .. import Config, Pipeline, spawn_seed_sequence, make_rng, make_seed_sequence
from ..decorators import parallel
from ..named_expr import eval_expr
from ..utils import to_list

from .domain import ConfigAlias
from .named_expr import E, O, EC
from .utils import generate_id, must_execute, parse_name, MultiOut
from .profiler import ExecutorProfiler
from .storage import BaseResearchStorage

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
    variables : str, list, optional
        variables to return from call
    """
    def __init__(self, pipeline, mode='generator', variables=None):
        if mode not in['generator', 'func', 'execute_for']:
            raise ValueError(f'Unknown PipelineWrapper mode: {mode}')
        if isinstance(pipeline, str):
            pipeline = parse_name(pipeline)
        self.pipeline = pipeline
        self.mode = mode
        self.variables = to_list(variables or [])
        self.config = None

    def __call__(self, config, batch=None, **kwargs):
        """ Execute pipeline.

        Parameters
        ----------
        config : Config
            `Config` for pipeline, defined at the first pipeline execution.
        batch : Batch, optional
            `Batch` to use with `execute_for` method, by default None

        Returns
        -------
        tuple or generator
            return depends on the mode:
                - 'generator': generator
                - 'func': the pipeline object and its variables values
                - 'execute_for': the processed batch and its variables values
        """
        if self.config is None:
            self.config = {**config, **kwargs}
            self.pipeline.set_config(self.config)

        if self.mode == 'generator':
            return self.generator()
        if self.mode == 'func':
            return (self.pipeline.run(), *self._get_vars_values())
        batch = self.pipeline.execute_for(batch)
        return (batch, *self._get_vars_values()) # if self.mode == 'execute_for'

    def generator(self):
        """ Generator returns batches from pipeline. Generator will stop when StopIteration will be raised. """
        self.reset()
        while True:
            try:
                yield (self.pipeline.next_batch(), *self._get_vars_values())
            except StopIteration:
                return

    def _get_vars_values(self):
        return [self.pipeline.v(var) for var in self.variables if var is not None]

    def __getattr__(self, attr):
        return getattr(self.pipeline, attr)

    def reset(self):
        """ Reset pipeline state: variables and bacth generator. """
        self.pipeline.reset('iter', 'vars')

    def __copy__(self):
        """ Create copy of the pipeline with the same mode. """
        if isinstance(self.pipeline, (list, tuple)):
            return PipelineWrapper(self.pipeline, self.mode, self.variables)

        return PipelineWrapper(self.pipeline + Pipeline(), self.mode, self.variables)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

class InstanceCreator:
    """ Instance class to use in each experiment in research. Will be initialized at the start of
    the experiment execution.

    Parameters
    ----------
    name : str
        name of the instance to use in research.
    creator : class
        class of the instance.
    root : bool, optional
        does instance is the same for all branches or not, by default False.
    args : list
        args for instance initialization, by default None.
    kwargs, other_kwargs : dict
        kwargs for instance initialization.
    """
    def __init__(self, name, creator, root=False, args=None, kwargs=None, **other_kwargs):
        self.name = name
        self.creator = creator
        self.root = root
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.other_kwargs = other_kwargs

    def __call__(self, experiment, *args, **kwargs):
        """ Create instance of the creator. """
        args = [*self.args, *args]
        kwargs = {**self.kwargs, **kwargs}
        args = eval_expr(args, experiment=experiment)
        kwargs = eval_expr(kwargs, experiment=experiment)
        other_kwargs = eval_expr(self.other_kwargs, experiment=experiment)
        return self.creator(*args, **other_kwargs, **kwargs)

class ExecutableUnit:
    """ Class to represent callables and generators executed in experiment.

    Parameters
    ----------
    name : str
        name of the unit.
    func : callable or tuple of str, optional
        callable itself or tuple which consists of instance name and its attribute to call, by default None.
        `func` and `generator` can't be defined simultaneously.
    generator : generator or tuple of str, optional
        generator itself or tuple which consists of instance name and its attribute to call, by default None.
        `func` and `generator` can't be defined simultaneously.
    root : bool, optional
        does unit is the same for all branches or not, by default False.
    when : str, int or list of ints, optional
        iterations of the experiment to execute unit, by default 1.
            - If `'last'`, unit will be executed just at last iteration (if `iteration + 1 == n_iters` or
              `StopIteration` was raised).
            - If positive int, pipeline will be executed each `when` iterations.
            - If str, must be `'#{it}'` or `'last'` where it is int, the pipeline will be executed at this
              iteration (zero-based).
            - If list, must be list of int or str described above.
    args, kwargs : optional
        args and kwargs for unit call, by default None.
    save_to : str, list or None, optional
        names to save output from unit. Must be None if `save_output_dict` is True. By default None.
    save_output_dict : bool, optional
        if the output is a dict, use its keys as names of variables to store in results.
        If True, `save_to` must be None. By default False.
    """
    def __init__(self, name, func=None, generator=None, root=False, when=1,
                 args=None, kwargs=None, save_to=None, save_output_dict=False, **other_kwargs):
        self.name = name
        self.callable = func
        self.generator = generator

        self.root = root
        self.when = when

        if isinstance(self.when, (int, str)):
            self.when = [self.when]

        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.other_kwargs = other_kwargs

        self.config = None
        self.experiment = None
        self.output = None # the last output of the unit.
        self.iterator = None # created iterator

        self.iteration = 0

        self.save_to = save_to
        self.save_output_dict = save_output_dict

        if self.save_to is not None and self.save_output_dict:
            raise ValueError('save_to is not None and save_output_dict is True.')


    def set_unit(self, config, experiment):
        """ Set config and experiment instance for the unit. """
        self.config = config
        self.experiment = experiment

    def transform_method(self):
        """ Transform `callable` or `generator` from tuples of str to instance attributes. """
        attr = 'callable' if self.callable is not None else 'generator'
        src = getattr(self, attr)
        if isinstance(src, (tuple, list)):
            setattr(self, attr, getattr(self.experiment.instances[src[0]], src[1]))
        if isinstance(src, PipelineWrapper) and isinstance(src.pipeline, (tuple, list)):
            pipeline = src.pipeline
            src.pipeline = getattr(self.experiment.instances[pipeline[0]], pipeline[1])

    def __call__(self, iteration, n_iters, last=False):
        """ Call unit: execute callable or get the next item from generator.

        Parameters
        ----------
        iteration : int
            current iteration of the experiment.
        n_iters : int or None
            total number of iterations of the experiment. `None` means that experiment will be executed until
            `StopIteration` for at least one executable unit.
        last : bool, optional
            does it is the last iteration or not, by default False. `last` is `True` when StopIteration was raised
            for one of the previously executed units or `iteration + 1 == n_iters` when `n_iters` is not None.

        Returns
        -------
        object
            output of the wrapped unit
        """
        if iteration == 0:
            self.transform_method()

        if self.must_execute(iteration, n_iters, last):
            total = (n_iters - 1) if n_iters is not None else None
            self.experiment.logger.debug(f"Execute '{self.name}' [{iteration}/{total}]")

            self.iteration = iteration
            args = eval_expr(self.args, experiment=self.experiment)
            kwargs = eval_expr(self.kwargs, experiment=self.experiment)
            other_kwargs = eval_expr(self.other_kwargs, experiment=self.experiment)

            start_time = time.time()
            if self.callable is not None:
                self.output = self.callable(*args, **kwargs, **other_kwargs)
            else:
                if self.iterator is None:
                    start_time = time.time()
                    self.iterator = self.generator(*args, **kwargs, **other_kwargs)
                else:
                    start_time = time.time()
                self.output = next(self.iterator)

            if self.save_to or self.save_output_dict:
                self.save_output()

            eval_time = time.time() - start_time
            return self.output, eval_time

        return None, None

    def must_execute(self, iteration, n_iters=None, last=False):
        """ Returns does unit must be executed for the current iteration. """
        return must_execute(iteration, self.when, n_iters, last)

    def save_output(self):
        """ Save output of the unit. """
        if self.save_output_dict:
            dst = self.output.keys()
            src = self.output.values()
        else:
            if not isinstance(self.save_to, (list, tuple)):
                src = [self.output]
                dst = [self.save_to]
            else:
                src = self.output
                dst = self.save_to

            if len(src) != len(dst):
                raise ValueError(f'Length of src and dst must be the same but src={src} and dst={dst}')

        for _src, _dst in zip(src, dst):
            if _dst is not None:
                self.experiment.storage.update_variable(_dst, _src)

    @property
    def src(self):
        """ Return wrapped source (callable or generator) of the unit. """
        attr = 'callable' if self.callable is not None else 'generator'
        return getattr(self, attr)

    def __copy__(self):
        """ Create copy of the unit. """
        attrs = ['name', 'callable', 'generator', 'root', 'when', 'args', 'kwargs', 'save_to', 'save_output_dict']
        params = {attr if attr !='callable' else 'func': copy(getattr(self, attr)) for attr in attrs}
        new_unit = ExecutableUnit(**params, **copy(self.other_kwargs))
        return new_unit

    def __getattr__(self, key):
        return getattr(self.src, key)

    def __getitem__(self, key):
        return self.src[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

class Experiment:
    """ Experiment description which consists of lists of instances to create and actions to execute. Each action
    defines executable unit (callable or generator) and corresponding execution parameters. Actions will be executed in
    the order defined by list. Actions can be defined as attributes of some instance (e.g., see `name` of
    `:meth:.add_callable`).

    Parameters
    ----------
    instance_creators : list, optional
        list of instance_creators, by default None. Can be extended by `:meth:.add_instance`.
    actions : list, optional
        list of actions, by default None.  Can be extended by `:meth:.add_executable_unit` and other methods.
    namespaces : list, optional
        list of namespaces, by default None. If None, then global namespace will be added.
    """
    def __init__(self, instance_creators=None, actions=None, namespaces=None):
        if instance_creators is not None:
            self.instance_creators = OrderedDict(instance_creators)
        else:
            self.instance_creators = OrderedDict()
        if actions is None:
            self.actions = OrderedDict() # unit_name : (instance_name, attr_name) or callable
        else:
            self.actions = actions
        self._namespaces = namespaces if namespaces is not None else []


        self._is_alive = True # should experiment be executed or not. Becomes False when Exceptions was raised and all
                              # units for these iterations were executed.
        self._is_failed = False # was an exception raised or not

        self.last = False
        self.outputs = dict()
        self.storage = None
        self.has_dump = False # does unit has any dump actions or not
        self.name = None # name of the executor/research
        self.dump_results = None
        self.loglevel = None
        self.monitor = None
        self.id = None #pylint:disable=invalid-name
        self.index = None
        self.config_alias = None
        self.config = None
        self.executor = None
        self.research = None
        self.instances = None
        self.logger = None
        self.iteration = None
        self.random_seed = None
        self.random = None
        self.profiler = None
        self.stdout_file = None
        self.stderr_file = None

    @property
    def is_alive(self):
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value):
        self._is_alive = self._is_alive and value

    @property
    def is_failed(self):
        return self._is_failed

    @is_failed.setter
    def is_failed(self, value):
        self._is_failed = self._is_failed or value

    def add_executable_unit(self, name, src=None, mode='func', when=1,
                            save_to=None, dump=None, args=None, **kwargs):
        """ Add executable unit to experiment.

        Parameters
        ----------
        name : str
            name of unit to use inside of the research. Can be `'instance_name.attr'` to refer to instance attr.
        src : callable or generator, optional
            callable or generator to wrap into ExecutableUnit, by default None.
        mode : str, optional
            type of src ('func' or 'generator'), by default 'func'
        when : int, str or list, optional
            iterations to execute callable (see `when` of `:class:ExecutableUnit`), by default 1.
        save_to : str or list, optional
            dst to save output of the unit (if needed), by default None.
        dump : int, str or list, optional
            iterations to dump results (see `when` of `:class:ExecutableUnit`), by default 1.
        args : list, optional
            args to execute unit, by default None.
        kwargs : dict
            kwargs to execute unit.

        Returns
        -------
        Research
        """
        if not isinstance(name, str) and hasattr(name, '__name__'):
            src = name
            name = src.__name__

        if src is None:
            kwargs[mode] = parse_name(name)
        else:
            kwargs[mode] = src

        name = self.add_postfix(name)
        self.actions[name] = ExecutableUnit(name=name, args=args, when=when, save_to=save_to, **kwargs)
        if dump is not None:
            self.dump(save_to, when=dump)
        return self

    def add_callable(self, name, func=None, args=None, when=1, save_to=None, dump=None, **kwargs):
        """ Add callable to experiment.

        Parameters
        ----------
        name : str
            name of callable to use inside of the research. Can be `'instance_name.method'` to refer to instance method.
        func : callable, optional
            callable to add into experiment, by default None.
        args : list, optional
            args to execute callable, by default None.
        when : int, str or list, optional
            iterations to execute callable (see `when` of `:class:ExecutableUnit`), by default 1.
        save_to : str or list, optional
            dst to save output of the callable (if needed), by default None.
        dump : int, str or list, optional
            iterations to dump results (see `when` of `:class:ExecutableUnit`), by default 1.
        root : bool, optional
            does unit is the same for all branches or not, by default False.
        kwargs : dict
            kwargs to execute callable.

        Returns
        -------
        Research
        """
        return self.add_executable_unit(name, src=func, mode='func', when=when,
                                        save_to=save_to, dump=dump, args=args, **kwargs)

    def add_generator(self, name, generator=None, args=None, **kwargs):
        """ Add generator to experiment.

        Parameters
        ----------
        name : str
            name of generator to use inside of the research. Can be `'instance_name.method'` to refer
            to instance method.
        generator : generator, optional
            generator to add into experiment, by default None.
        args : list, optional
            args to create iterator, by default None.
        when : int, str or list, optional
            iterations to get item from generator (see `when` of `:class:ExecutableUnit`),
            by default 1.
        save_to : NamedExpression, optional
            dst to save generated item (if needed), by default None.
        root : bool, optional
            does unit is the same for all branches or not, by default False.
        kwargs : dict
            kwargs to create iterator.

        Returns
        -------
        Research
        """
        return self.add_executable_unit(name, src=generator, mode='generator', args=args, **kwargs)

    def add_instance(self, name, creator, root=False, **kwargs):
        """ Add instance of some class into research.

        Parameters
        ----------
        name : str
            instance name.
        creator : class
            class which instance will be used to get attributes.
        root : bool, optional
            does instances is the same for all branches or not, by default False.

        Returns
        -------
        Experiment
        """
        self.instance_creators[name] = InstanceCreator(name, creator, root, **kwargs)
        self.add_callable(f'init_{name}', _create_instance, experiments=E(all=root),
                          root=root, item_name=name, when="%0")
        return self

    def add_pipeline(self, name, root=None, branch=None, run=False, variables=None, dump=None, when=1, **kwargs):
        """ Add pipeline to experiment.

        Parameters
        ----------
        name : str
            name of pipeline to use inside of the research. Can be `'instance_name.attribute'` to refer to instance
            attribute.
        root : batchflow.Pipeline, optional
            a pipeline to execute, by default None. It must contain `run` action with `lazy=True` or `run_later`.
            Only if `branch` is None, `root` may contain parameters that can be defined by config.
            from domain.
        branch : Pipeline, optional
            a parallelized pipeline to execute, by default None. Several copies of branch pipeline will be executed
            in parallel per each batch received from the root pipeline. May contain parameters that can be
            defined by domain, all branch pipelines will correspond to different experiments and will have
            different configs from domain.
        run : bool, optional
            if False then `.next_batch()` will be applied to pipeline, else `.run()` , by default False.
        dump : int, str or list, optional
            iterations to dump results (see `when` of `:class:ExecutableUnit`), by default 1.
        variables : str, list or None, optional
            variables of pipeline to save.
        when : int, str or list, optional
            iterations to execute (see `when` of `:class:ExecutableUnit`), by default 1.

        Returns
        -------
        Research
        """
        save_to = [None] + to_list(variables or [])
        if branch is None:
            mode = 'func' if run else 'generator'
            pipeline = PipelineWrapper(root if root is not None else name, mode=mode, variables=variables)
            self.add_executable_unit(name, src=pipeline, mode=mode, config=EC(full=True),
                                     when=when, save_to=save_to, **kwargs)
        else:
            root = PipelineWrapper(root, mode='generator')
            branch = PipelineWrapper(branch, mode='execute_for', variables=variables)

            self.add_generator(f'{name}_root', generator=root, config=EC(full=True), when=when, **kwargs)
            self.add_callable(f'{name}', func=branch, config=EC(full=True), batch=O(f'{name}_root')[0], save_to=save_to,
                              when=when, **kwargs)
        if variables is not None:
            if dump is not None:
                self.dump(variables, dump)
        return self

    def add_namespace(self, namespace):
        """ Add namespace to experiment. """
        self._namespaces += [namespace]
        return self

    @property
    def _all_namespaces(self):
        common_namespaces = [sys.modules["__main__"], sys.modules["builtins"]]
        namespaces = [sys.modules[namespace] if isinstance(namespace, str) else namespace
                      for namespace in self._namespaces]

        return common_namespaces + namespaces

    def get_method(self, name):
        """ Return a method by the name """
        for namespace in self._all_namespaces:
            if hasattr(namespace, name):
                return getattr(namespace, name)
            if isinstance(namespace, dict) and name in namespace:
                return namespace[name]
        return None

    def __getattr__(self, name):
        method = self.get_method(name)
        if method is None:
            warnings.warn(f'Method {name} was not found in any namespace.')
            return None
        return _explicit_call(method, name, self)

    def save(self, src, dst, when=1, save_output_dict=False, copy=False): #pylint:disable=redefined-outer-name
        """ Save something to research results.

        Parameters
        ----------
        src : NamedExpression
            value to save.
        dst : str
            name in the results.
        when : int, str or list, optional
            iterations to execute (see `when` of `:class:ExecutableUnit`), by default 1.
        copy : bool, optional
            copy value or not, by default False
        """
        name = '__save_results' if dst is None else f'__save_results_{dst}'
        name = self.add_postfix(name)
        self.add_callable(name, _get_input, x=src, copy=copy, when=when, save_to=dst,
                          experiment=E(), save_output_dict=save_output_dict)
        return self

    def dump(self, variable=None, when='last'):
        """ Dump current results to the storage and clear it.

        Parameters
        ----------
        variable : str, optional
            names in results to dump, by default None. None means that all results will be dumped.
        when : int, str or list, optional
            iterations to execute (see `when` of `:class:ExecutableUnit`), by default 1.

        Returns
        -------
        Research
        """
        self.has_dump = True
        name = '__dump_results' if variable is None else f'__dump_results_{variable}'
        name = self.add_postfix(name)
        self.add_callable(name, _dump_results,
                          when=when,
                          variable=variable, experiment=E())
        return self

    def add_postfix(self, new_name):
        """ Add postfix for conincided unit name. """
        n_actions = sum(self._has_postfix(new_name, unit_name) for unit_name in self.actions)
        return new_name if n_actions == 0 else f"{new_name}_{n_actions}"

    def _has_postfix(self, new_name, unit_item):
        postfix = unit_item[len(new_name):]
        return (unit_item == new_name) or (len(postfix) >= 2 and postfix[0] == '_' and postfix[1:].isdigit())

    @property
    def only_callables(self):
        """ Check if experiment has only callables. """
        for unit in self.actions.values():
            if unit.callable is None:
                return False
        return True

    @property
    def results(self):
        return self.storage.results

    def copy(self):
        """ Create copy of the experiment. Is needed to create experiments for branches. """
        instance_creators = copy(self.instance_creators)
        actions = OrderedDict([(name, copy(unit)) for name, unit in self.actions.items()])
        new_experiment = Experiment(instance_creators=instance_creators, actions=actions)
        new_experiment.has_dump = self.has_dump
        return new_experiment

    def __getitem__(self, key):
        return self.actions[key]

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        _ = memo
        return self.copy()

    def init(self, index, config, executor=None):
        """ Create all instances of units to start experiment. """
        self.index = index
        self.executor = executor
        self.research = executor.research

        seed = spawn_seed_sequence(self.executor)
        self.random_seed = seed
        self.random = make_rng(seed)

        if self.research is not None:
            self.id = config.pop_config('id').config()['id']
        else:
            self.id = generate_id(config, self.random)
        self.pop_index_keys(config)
        self.config_alias = config
        self.config = config.config()

        # Get attributes from research or kwargs of executor
        params = ['loglevel', 'name', 'monitor', 'debug', 'profile',
                  'redirect_stdout', 'redirect_stderr', 'dump_results']

        for attr in params:
            value = getattr(self.executor, attr)
            setattr(self, attr, value)

        storage_class = self.executor.storage.experiment_storage_class

        self.storage = storage_class(self, loglevel=self.loglevel)
        self.logger = self.storage.logger
        self.profiler = self.storage.profiler

        self.instances = OrderedDict()

        root_experiment = executor.experiments[0] if len(executor.experiments) > 0 else None

        for name in self.actions:
            if self.actions[name].root and root_experiment is not None:
                self.actions[name] = root_experiment.actions[name]
            else:
                self.actions[name].set_unit(config=config, experiment=self)


    def pop_index_keys(self, config):
        """ Remove auxilary keys used to create prefix. """
        for key in config.keys():
            if key[0] == '#':
                config.pop_config(key)
        config.pop_config('_prefix')

    def create_stream(self, name, *streams):
        """ Create contextmanager to redirect stdout/stderr. """
        streams = [stream for stream in streams if not isinstance(stream, contextlib.nullcontext)]
        if len(streams) > 0:
            if name == 'stdout':
                return contextlib.redirect_stdout(MultiOut(*streams))
            return contextlib.redirect_stderr(MultiOut(*streams)) # 'stderr'
        return contextlib.nullcontext()

    def call(self, name, iteration, n_iters=None):
        """ Execute one iteration of the experiment. """
        context_manager_out = self.create_stream(
            'stdout', self.storage.stdout_file, self.executor.storage.stdout_file
        )
        context_manager_err = self.create_stream(
            'stderr', self.storage.stderr_file, self.executor.storage.stderr_file
        )

        with context_manager_out, context_manager_err:
            if self.is_alive or name.startswith('__'):
                if self.profiler:
                    self.profiler.enable()

                self.last = self.last or (iteration + 1 == n_iters)
                self.iteration = iteration

                exception = (StopIteration, KeyboardInterrupt) if self.debug else Exception
                try:
                    self.outputs[name], unit_time = self.actions[name](iteration, n_iters, last=self.last)
                except exception as e: #pylint:disable=broad-except
                    self.is_failed = True
                    self.last = True
                    if isinstance(e, StopIteration):
                        self.logger.info(f"Stop '{name}' [{iteration}/{n_iters}]")
                    else:
                        ex_traceback = e.__traceback__
                        msg = ''.join(traceback.format_exception(e.__class__, e, ex_traceback))
                        self.logger.error(f"Fail '{name}' [{iteration}/{n_iters}]: Exception\n{msg}")
                        if self.monitor:
                            self.monitor.fail_item_execution(name, self, msg)
                if self.is_failed and ((list(self.actions.keys())[-1] == name) or (not self.executor.finalize)):
                    self.is_alive = False

        if self.profiler:
            self.profiler.disable(iteration, name, unit_time=unit_time, experiment=self.id)

    def show_profile_info(self, **kwargs):
        return self.profiler.show_profile_info(**kwargs)

    @property
    def profile_info(self):
        return self.profiler.profile_info

    def __str__(self):
        repr = ''
        spacing = ' ' * 4

        if len(self.instance_creators) > 0:
            repr += "instances:\n"
            for name, creator in self.instance_creators.items():
                repr += spacing + f"{name}(\n"
                repr += 2 * spacing + f"root={creator.root},\n"
                repr += ''.join([spacing * 2 + f"{key}={value}\n" for key, value in creator.kwargs.items()])
                repr += spacing + ")\n"
            repr += '\n'

        if len(self.actions) > 0:
            repr += "units:\n"
            attrs = ['callable', 'generator', 'root', 'when', 'args']
            for name, action in self.actions.items():
                repr += spacing + f"{name}(\n"
                repr += ''.join([spacing * 2 + f"{key}={getattr(action, key)}\n" for key in attrs])
                kwargs = {**action.kwargs, **action.other_kwargs}
                repr += spacing * 2 + f"kwargs={kwargs}\n" + spacing + ")\n"

        return repr

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

class Executor:
    """ Executor of experiments in branches.

    Parameters
    ----------
    experiment : Experiment
        experiment scheme.
    research : Research, optional
        Research instance to get meta information (if needed), by default None.
    worker : Worker, optional
        Worker instance, by default None.
    configs : list, optional
        configs for different branches, by default None.
    executor_config : Config of dict, optional
        config for the all experiments, will be appended to each expeirment config, by default None.
    branches_configs : list, optional
        configs for each branch, will be appended to each expeirment config, by default None.
    target : str, optional
        how to parallelize branches ('threads' or 'for'), by default 'threads'.
    n_iters : int, optional
        the number of iterations to execute, by default None.
    task_name : str, optional
        name of the task, by default None
    """
    def __init__(self, experiment, research=None, worker=None, configs=None, executor_config=None,
                 branches_configs=None, target='threads', n_iters=None, task_name=None, **kwargs):
        if configs is None:
            if branches_configs is None:
                self.n_branches = 1
            else:
                self.n_branches = len(branches_configs)
        else:
            if branches_configs is not None and len(configs) != len(branches_configs):
                raise ValueError('`configs` and `branches_configs` must be of the same length.')
            self.n_branches = len(configs)

        self.configs = configs or [Config() for _ in range(self.n_branches)]
        self.executor_config = Config(executor_config or dict())
        self.branches_configs = branches_configs or [Config() for _ in range(self.n_branches)]
        self.branches_configs = [Config(config) for config in self.branches_configs]
        self.n_iters = n_iters
        self.research = research
        self.experiment_template = experiment
        self.task_name = task_name or 'Task'
        self.target = target
        self.set_params(kwargs)

        self.worker = worker
        if worker is not None:
            seed = spawn_seed_sequence(worker)
        else:
            seed = make_seed_sequence()

        self.random_seed = seed
        self.random = make_rng(seed)

        if self.research is not None:
            self.storage = self.research.storage
        else:
            storage = 'local' if self.dump_results else 'memory'
            self.storage = BaseResearchStorage(self, storage=storage)

        self.create_experiments()

        self.pid = None
        self.common_stdout = None
        self.common_stderr = None

    def set_params(self, kwargs):
        """ Set params of executor. Is used to get attributes from research or from kwargs. """
        defaults = {
            'loglevel': 'debug',
            'name': 'executor',
            'monitor': None,
            'debug': False,
            'profile': False,
            'redirect_stdout': True,
            'redirect_stderr': True,
            'dump_results': False,
            'finalize': False
        }
        for attr in defaults:
            if self.research:
                value = getattr(self.research, attr)
            else:
                value = kwargs.get(attr, defaults[attr])
            setattr(self, attr, value)

    def create_experiments(self):
        """ Initialize experiments. """
        self.experiments = []
        for index, (config, branch_config) in enumerate(zip(self.configs, self.branches_configs)):
            full_config = ConfigAlias(config) + ConfigAlias(branch_config) + ConfigAlias(self.executor_config)
            experiment = self.experiment_template.copy()
            experiment.init(index, full_config, self)
            experiment.logger.info(f"{self.task_name}[{index}] has been started with config:\n {repr(config)}")

            self.experiments.append(experiment)

    def run(self):
        """ Run experiments. """
        self.storage.create_redirection_streams()

        with self.storage.stdout_file, self.storage.stderr_file:
            self.pid = os.getpid() if self.research and self.research.parallel else None

            iterations = range(self.n_iters) if self.n_iters else itertools.count()
            if self.research:
                for experiment in self.experiments:
                    self.research.monitor.start_experiment(experiment)

            for iteration in iterations:
                for unit_name, unit in self.experiment_template.actions.items():
                    if unit.root or len(self.experiments) == 1:
                        self.call_root(iteration, unit_name)
                    else:
                        self.parallel_call(iteration, unit_name, target=self.target, debug=self.debug) #pylint:disable=unexpected-keyword-arg
                if not any([experiment.is_alive for experiment in self.experiments]):
                    break
                if self.research:
                    for experiment in self.experiments:
                        if experiment.is_alive:
                            self.research.monitor.execute_iteration(experiment)

            for index, experiment in enumerate(self.experiments):
                if self.research:
                    self.research.monitor.stop_experiment(experiment)
                experiment.logger.info(f"{self.task_name}[{index}] has been finished.")

        self.close()

    def close(self):
        """ Close storages. """
        for experiment in self.experiments:
            experiment.storage.close()

        self.storage.close_files()
        if self.research is None:
            self.storage.close()

    @parallel(init='_parallel_init_call')
    def parallel_call(self, experiment, iteration, unit_name):
        """ Parallel call of the unit 'unit_name' """
        if self.finalize or (not experiment.is_failed) or unit_name.startswith('__'):
            experiment.call(unit_name, iteration, self.n_iters)

    def _parallel_init_call(self, iteration, unit_name):
        """ Auxilary method to call before '_parallel_call'. """
        _ = iteration, unit_name
        return self.experiments

    def call_root(self, iteration, unit_name):
        """ Call root executable unit. """
        # TODO: experiment must be alive if error was in the branch after all roots
        if self.finalize or (not self.experiments[0].is_failed) or unit_name.startswith('__'):
            self.experiments[0].call(unit_name, iteration, self.n_iters)

            for experiment in self.experiments[1:]:
                if self.finalize or (not experiment.is_failed) or unit_name.startswith('__'):
                    experiment.outputs[unit_name] = self.experiments[0].outputs[unit_name]
                for attr in ['_is_alive', '_is_failed', 'iteration']:
                    setattr(experiment, attr, getattr(self.experiments[0], attr))

    @property
    def profiler(self):
        """ Executor profiler. """
        if self.experiments[0].profile:
            return ExecutorProfiler(self.experiments)
        return None

    @property
    def profile_info(self):
        """ Profile info. """
        if self.profiler:
            return self.profiler.profile_info
        return None

    def show_profile_info(self, **kwargs):
        return self.profiler.show_profile_info(**kwargs)

def _create_instance(experiments, item_name):
    if not isinstance(experiments, list):
        experiments = [experiments]
    instance = experiments[0].instance_creators[item_name](experiments[0])
    for e in experiments:
        e.instances[item_name] = instance

def _get_input(x, copy, *args, **kwargs): #pylint:disable=redefined-outer-name
    _ = args, kwargs
    return deepcopy(x) if copy else x

def _dump_results(variable, experiment):
    """ Callable to dump results. """
    experiment.storage.dump_results(variable)

def _explicit_call(method, name, experiment):
    """ Add unit into research by explicit call in research-pipeline. """
    def _method(*args, **kwargs):
        return experiment.add_executable_unit(name, src=method, args=args, **kwargs)
    return _method
