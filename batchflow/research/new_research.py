from collections import OrderedDict
from copy import copy, deepcopy
import itertools
import traceback

from .. import Config, inbatch_parallel, Pipeline
from ..named_expr import NamedExpression, eval_expr

class E(NamedExpression):
    def __init__(self, unit=None, all=False, **kwargs):
        self.unit = unit
        self.all = all

    def _get(self, **kwargs):
        experiment = kwargs['experiment']
        if self.all:
            return experiment.executor.experiments
        return [experiment]

    def get(self, **kwargs):
        experiments = self._get(**kwargs)
        results = self.transform(experiments)
        if self.all:
            return results
        return results[0]

    def transform(self, experiments):
        if self.unit is not None:
            return [exp[self.unit] for exp in experiments]
        return experiments

class EC(E):
    def __init__(self, name=None, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def transform(self, experiments):
        if self.name is None:
            return [exp.config for exp in experiments]
        else:
            return [exp.config[self.name] for exp in experiments]

class O(E):
    def __init__(self, name, **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def transform(self, experiments):
        return [exp[self.name]._output for exp in experiments]

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
    def __init__(self, namespaces=None, actions=None, research_meta=None):
        if namespaces is not None:
             self.namespaces = OrderedDict(namespaces)
        else:
            self.namespaces = OrderedDict()
        if actions is None:
            self.actions = OrderedDict() # unit_name : (namespace_name, attr_name)
        else:
            self.actions = actions

        self.state = ['_is_alive', '_exception_raised', 'iteration']

        self.experiment_meta = research_meta
        self._is_alive = True
        self._exception_raised = False
        self.last = False
        self.outputs = dict()
        self.results = OrderedDict()

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

    def add_namespace(self, name, namespace, root=False, **kwargs):
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

    def add_executable_unit(self, name, src=None, mode='func', args=None, **kwargs):
        if src is None:
            kwargs[mode] = self.parse_name(name)
        else:
            kwargs[mode] = src
        self.actions[name] = ExecutableUnit(name=name, args=args, **kwargs)
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

    def save(self, value, save_to, iterations_to_execute=None, copy=False):
        def _save_results(value, save_to, experiment, copy=copy):
            previous_values = experiment.results.get(save_to, OrderedDict())
            previous_values[experiment.iteration] = deepcopy(value) if copy else value
            experiment.results[save_to] = previous_values
        name = self.add_postfix('save_results')
        self.add_callable(name, _save_results,
                          iterations_to_execute=iterations_to_execute,
                          value=value, save_to=save_to, experiment=E())
        return self

    def add_postfix(self, name):
        return name + '_' + str(sum([item.startswith(name) for item in self.actions]))

    def dump_results(self, name):
        # TODO
        return self

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

    def generate_id(self):
        self.id = None

    def create_folder(self):
        # make dir using path from self.meta
        pass

    def create_instances(self, config, meta, executor=None):
        self.meta = meta
        self.config = config
        self.executor = executor
        self.generate_id()
        self.create_folder()
        self.instances = OrderedDict()

        root_experiment = executor.experiments[0] if len(executor.experiments) > 0 else None

        for name in self.actions:
            if self.actions[name].root and root_experiment is not None:
                self.actions[name] = root_experiment.actions[name]
            else:
                self.actions[name].set_unit(config=config, experiment=self)

    def call(self, name, iteration, n_iters=None):
        if self.is_alive:
            self.last = self.last or (iteration + 1 == n_iters)
            self.iteration = iteration
            try:
                self.outputs[name] = self.actions[name](iteration, n_iters, last=self.last)
            except Exception as e:
                self.exception_raised = True
                self.last = True
                self.process_exception(e)
            if self.exception_raised and (list(self.actions.keys())[-1] == name):
                self.is_alive = False

    def process_exception(self, exception):
        self.exception = exception
        ex_traceback = exception.__traceback__
        self.traceback = ''.join(traceback.format_exception(exception.__class__, exception, ex_traceback))

class Executor:
    def __init__(self, experiment, meta, configs=None, executor_config=None, branches_configs=None, target='threads', n_iters=None):
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
        self.meta = meta
        self.experiment_template = experiment

        self.create_experiments()

        self.parallel_call = inbatch_parallel(init=self._parallel_init_call, target=target, _use_self=False)(self._parallel_call)

    def create_experiments(self):
        self.experiments = []
        for config, branch_config in zip(self.configs, self.branches_configs):
            config = config + branch_config + self.executor_config
            experiment = self.experiment_template.copy()
            experiment.create_instances(config, self.meta, self)
            self.experiments.append(experiment)

    def run(self):
        iterations = range(self.n_iters) if self.n_iters else itertools.count()
        for iteration in iterations:
            for unit_name, unit in self.experiment_template.actions.items():
                if unit.root:
                    self.call_root(iteration, unit_name)
                else:
                    self.parallel_call(iteration, unit_name)
            if not any([experiment.is_alive for experiment in self.experiments]):
                break

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
