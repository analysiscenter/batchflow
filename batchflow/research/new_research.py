import os
from collections import OrderedDict
from copy import copy, deepcopy
import itertools
import traceback
import multiprocess as mp
import hashlib
import random
import dill

from .. import Config, inbatch_parallel, Pipeline
from .domain import Domain, ConfigAlias
from .distributor import Distributor
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
        self.dump_results = False

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
        def _save_results(value, save_to, experiment, copy): #TODO: test does copy work
            previous_values = experiment.results.get(save_to, OrderedDict())
            previous_values[experiment.iteration] = deepcopy(value) if copy else value
            experiment.results[save_to] = previous_values
        name = self.add_postfix('save_results')
        self.add_callable(name, _save_results,
                          iterations_to_execute=iterations_to_execute,
                          value=value, save_to=save_to, experiment=E(), copy=copy)
        return self

    def dump(self, variable=None, iterations_to_execute=None):
        self.dump_results = True
        def _dump_results(variable, experiment):
            values = experiment.results[variable] if variable is not None else experiment.results
            variable = variable or 'all_results'
            iteration = experiment.iteration
            variable_path = os.path.join(experiment.full_path, variable)
            if not os.path.exists(variable_path):
                os.makedirs(variable_path)
            with open(os.path.join(variable_path, str(iteration)), 'wb') as file:
                dill.dump(values, file)
        name = self.add_postfix('dump_results')
        self.add_callable(name, _dump_results,
                          iterations_to_execute=iterations_to_execute,
                          variable=variable, experiment=E())
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
        new_experiment.dump_results = self.dump_results
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
        self.id = hashlib.md5(alias.encode('utf-8')).hexdigest() + str(random.getrandbits(16))

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
        self.generate_id(config.alias(as_string=True))
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
                self.info(name, 'call')
                self.outputs[name] = self.actions[name](iteration, n_iters, last=self.last)
            except Exception as e:
                self.exception_raised = True
                self.last = True
                if isinstance(e, StopIteration):
                    self.info(name, 'was stopped by StopIteration')
                else:
                    self.process_exception(name, e)
            if self.exception_raised and (list(self.actions.keys())[-1] == name):
                self.is_alive = False

    def process_exception(self, name, exception):
        self.exception = exception
        ex_traceback = exception.__traceback__
        self.traceback = ''.join(traceback.format_exception(exception.__class__, exception, ex_traceback))
        print(f'Experiment: {self.id}, it: {self.iteration}, name: {name} :: error')
        print(self.traceback)

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

class NewResearch:
    def __init__(self, name=None, domain=None, experiment=None, n_configs=None, n_reps=1, repeat_each=100):
        self.domain = Domain(domain)
        self.n_configs = n_configs
        self.n_reps = n_reps
        self.repeat_each = repeat_each
        self.experiment = experiment or Experiment()
        self.name = name or 'research'

    def init_instance(self, *args, **kwargs):
        self.experiment.add_namespace(*args, **kwargs)
        return self

    def add_callable(self, *args, **kwargs):
        self.experiment.add_callable(*args, **kwargs)
        return self

    def add_generator(self, *args, **kwargs):
        self.experiment.add_generator(*args, **kwargs)
        return self

    def add_pipeline(self, *args, **kwargs):
        self.experiment.add_pipeline(*args, **kwargs)
        return self

    def update_domain(self, **kwargs):
        """ Add domain update functions or update parameters.

        Parameters
        ----------
        function : callable or None

        each : int or 'last'
            when update method will be called. If 'last', domain will be updated
            when iterator will be finished. If int, domain will be updated with
            that period.
        n_updates : int
            the total number of updates.
        *args, **kwargs :
            update function parameters.
        """
        self.domain.set_update(function, each, n_updates, **kwargs)
        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame or dict (see :meth:`~.Results.load`). """
        return Results(self.name, *args, **kwargs)

    def attach_env_meta(self, **kwargs):
        """ Save the information about the current state of project repository.

        Parameters
        ----------
        kwargs : dict
            dict where values are bash commands and keys are names of files to save output of the command.
        """
        commands = {
            'commit': "git log --name-status HEAD^..HEAD",
            'diff': 'git diff',
            'status': 'git status',
            **kwargs
        }

        for filename, command in commands.items():
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, _ = process.communicate()
            result = re.sub('"image/png": ".*?"', '"image/png": "..."', output.decode('utf'))
            if not os.path.exists(os.path.join(self.name, 'env')):
                os.makedirs(os.path.join(self.name, 'env'))
            with open(os.path.join(self.name, 'env', filename + '.txt'), 'w') as file:
                print(result, file=file)

    def _get_devices(self, devices):
        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        n_workers = self.workers if isinstance(self.workers, int) else len(self.workers)
        total_n_branches = n_workers * n_branches
        if devices is None:
            devices = [[[None]] * n_branches] * n_workers
        if isinstance(devices, (int, str)):
            devices = [devices]
        if isinstance(devices[0], (int, str)):
            if total_n_branches > len(devices):
                _devices = list(itertools.chain.from_iterable(
                    zip(*itertools.repeat(devices, total_n_branches // len(devices)))
                ))
                devices = _devices + devices[:total_n_branches % len(devices)]
            else:
                devices = devices + devices[:-len(devices) % (total_n_branches)]
            if total_n_branches % len(devices) == 0:
                branches_per_device = total_n_branches // len(devices)
                devices = list(itertools.chain.from_iterable(itertools.repeat(x, branches_per_device) for x in devices))
            if len(devices) % total_n_branches == 0:
                devices_per_branch = len(devices) // total_n_branches
                devices = [
                    [
                        [
                            devices[n_branches * devices_per_branch * i + devices_per_branch * j + k]
                            for k in range(devices_per_branch)
                        ] for j in range(n_branches)
                    ] for i in range(n_workers)
                ]
        if isinstance(devices[0], list):
            def _transform_item(x):
                values = [str(item) if isinstance(item, int) else item for item in x]
                return dict(device=values) if x is not None else dict()

            devices = [[_transform_item(branch_config) for branch_config in worker_config] for worker_config in devices]
        return devices

    def create_research_folder(self):
        if not os.path.exists(self.name):
            os.makedirs(self.name)
            for subfolder in ['configs', 'description', 'env', 'results']:
                config_path = os.path.join(self.name, subfolder)
                if not os.path.exists(config_path):
                    os.makedirs(config_path)
        else:
            raise ValueError(
                "Research with name '{}' already exists".format(self.name)
            )

    def run(self, workers=1, branches=1, name=None, n_iters=None, bar=False, devices=None, executor_class=None,
            dump_results=True, parallel=True, executor_target='threads'):
        """ Run research.

        Parameters
        ----------
        n_iters: int or None
            number of iterations for each configuration. If None, wait StopIteration exception for at least
            one executable unit.
        workers : int or list of dicts (Configs)
            If int - number of workers to run pipelines or workers that will run them, `worker_class` will be used.

            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.

            Each element corresponds to one worker.
        branches: int or list of dicts (Configs)
            Number of different branches with different configs with the same root. Each branch will use the same batch
            from `root`. Pipelines will be executed in different threads.

            If int - number of pipelines with different configs that will use the same prepared batch
            from `root`.

            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.
        name : str or None
            name folder to save research. By default is 'research'.
        bar : bool or callable
            Whether to show a progress bar.
            If callable, it must have the same signature as `tqdm`.
        devices : str, list or None
            all devices will be distributed between workwers and branches.
            If you want to use different devices in branches, use expression `C('device')`.

            For example, for :class:`~.TFModel` add `device=C('device')` to model config.
            If None, default gpu configuration will be used
        worker_class : type
            worker class. `PipelineWorker` by default.
        timeout : int
            each job will be killed if it doesn't answer more then that time in minutes
        trials : int
            trials to execute job

        **How does it work**

        At each iteration all pipelines and functions will be executed in the order in which were added.
        If `update_config` callable is defined, each config will be updated by that function and then
        will be passed into each `ExecutableUnit`.
        If `update_domain` callable is defined, domain will be updated with the corresponding function
        accordingly to `each` parameter of `update_domain`.
        """
        self.n_iters = n_iters
        self.workers = workers
        self.branches = branches
        self.devices = self._get_devices(devices)
        self.executor_class = Executor
        self.dump_results = dump_results
        self.parallel = parallel
        self.executor_target = executor_target

        if self.dump_results:
            self.create_research_folder()

        if isinstance(workers, int):
            self.workers = [Config() for _ in range(workers)]
        if isinstance(branches, int):
            self.branches = [Config() for _ in range(branches)]

        self.domain.set_iter(n_items=self.n_configs, n_reps=self.n_reps, repeat_each=self.repeat_each)

        if self.domain.size is None and (self.domain.update_func is None or self.domain.update_each == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)

        print("Research {} is starting...".format(self.name))

        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        tasks = DynamicQueue(self.domain, self, n_branches)
        distributor = Distributor(tasks, self)
        distributor.run(bar)

        return self

class DynamicQueue:
    """ Queue of tasks that can be changed depending on previous results. """
    def __init__(self, domain, research, n_branches):
        self.domain = domain
        self.research = research
        self.n_branches = n_branches

        self.queue = mp.JoinableQueue()
        self.withdrawn_tasks = 0

    @property
    def total(self):
        """ Total estimated size of queue before the following domain update. """
        if self.domain.size is not None:
            return self.domain.size / self.n_branches
        return None

    def update(self):
        """ Update domain. """
        new_domain = self.domain.update(self.research) #TODO: put research instead of path
        if new_domain is not None:
            self.domain = new_domain
            return True
        return False

    def next_tasks(self, n_tasks=1):
        """ Get next `n_tasks` elements of queue. """
        configs = []
        for i in range(n_tasks):
            branch_tasks = [] # TODO: rename it
            try:
                for _ in range(self.n_branches):
                    branch_tasks.append(next(self.domain))
                configs.append(branch_tasks)
            except StopIteration:
                if len(branch_tasks) > 0:
                    configs.append(branch_tasks)
                break
        for i, executor_configs in enumerate(configs):
            self.put((self.withdrawn_tasks + i, executor_configs))
        n_tasks = len(configs)
        self.withdrawn_tasks += n_tasks
        return n_tasks

    def stop_workers(self, n_workers):
        """ Stop all workers by putting `None` task into queue. """
        for _ in range(n_workers):
            self.put(None)

    def join(self):
        self.queue.join()

    def get(self):
        return self.queue.get()

    def put(self, value):
        self.queue.put(value)

    def task_done(self):
        self.queue.task_done()