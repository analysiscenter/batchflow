#pylint:disable=logging-fstring-interpolation, too-many-arguments
""" Research class for muliple parallel experiments. """

import time
import itertools
import warnings
import psutil
import multiprocess as mp
import tqdm

from .domain import Domain
from .distributor import Distributor, DynamicQueue
from .experiment import Experiment, Executor
from ..utils import to_list
from .storage import BaseResearchStorage, LocalResearchStorage, MemoryResearchStorage

from ..utils_random import make_seed_sequence

class Research:
    """ Research is an instrument to run multiple parallel experiments with different combinations of
    parameters called experiment configs. Configs are produced by :class:`domain.Domain` (some kind of
    parameters grid.)

    Parameters
    ----------
    name : str, optional
        name (relative path) of the research and corresponding folder to store results, by default 'research'.
    domain : Domain, optional
        grid of parameters (see :class:`domain.Domain`) to produce experiment configs, by default None.
    experiment : Experiment, optional
        description of the experiment (see :class:`experiment.Experiment`), by default None. Experiment can be
        defined explicitly as a parameter or constructed by Research methods (`:meth:.add_callable`,
        `:meth:.add_generator`, etc.).
    n_configs : int, optional
        the number of configs to get from domain (see `n_items` of :meth:`domain.Domain.set_iter_params`),
        by default None.
    n_reps : int, optional
        the number of repetitions for each config (see `n_reps` of :meth:`domain.Domain.set_iter_params`), by default 1.
    repeat_each : int, optional
        see `repeat_each` of :meth:`domain.Domain.set_iter_params`, by default 100.
    """
    def __init__(self, name='research', domain=None, experiment=None, n_configs=None, n_reps=1, repeat_each=None):
        self.name = name
        self.domain = Domain(domain)
        self.experiment = experiment or Experiment()
        self.n_configs = n_configs
        self.n_reps = n_reps
        self.repeat_each = repeat_each
        self.create_id_prefix = False
        self.redirect_stdout = True
        self.redirect_stderr = True
        self.storage = None

        self._env = dict() # current state of git repo and other environment information.

        self.workers = 1
        self.branches = 1
        self.n_iters = None
        self.devices = None
        self.executor_class = Executor
        self.dump_results = True
        self.parallel = True
        self.executor_target = 'threads'
        self.loglevel = 'info'
        self.bar = True
        self.detach = False
        self.tasks_queue = None
        self.distributor = None
        self.monitor = None
        self.logger = None
        self.process = None
        self.debug = False
        self.finalize = True
        self.random_seed = None
        self.profile = False
        self.memory_ratio = None
        self.n_gpu_checks = 3
        self.gpu_check_delay = 5
        self.dump_monitor = True

        self.is_loaded = False
        self._is_executed = False
        self._env_meta_to_collect = []

    def __getattr__(self, key):
        if self.monitor is not None and key in self.monitor.SHARED_VARIABLES:
            return getattr(self.monitor, key)

        def _method(*args, **kwargs):
            getattr(self.experiment, key)(*args, **kwargs)
            return self
        _method.__doc__ = getattr(self.experiment, key).__doc__
        return _method

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def update_domain(self, function, when, **kwargs):
        """ Add domain update functions or update parameters.

        Parameters
        ----------
        function : callable or None
            function to update domain, returns new domain or None (means not to update).
        when : int, str or list, optional
            iterations to update (see `when` of `:class:ExecutableUnit`), by default 1.
        kwargs : dict
            update function parameters.
        """
        self.domain.set_update(function, when, **kwargs)
        return self

    def attach_env_meta(self):
        """ Get version of packages (by "pip list" and "conda list") and python version. Results will be stored
        in research folder (if it is created) or in _env attribute.
        """
        commands = {
            'pip': 'pip list',
            'conda': 'conda list',
            'python': '#python'
        }
        self._env_meta_to_collect.append(dict(commands=commands, cwd='.', dst=''))
        return self

    def attach_git_meta(self, cwd='.'):
        """ Get git repo state (current commit, diff and status). Results will be stored
        in research folder (if it is created) or in _env attribute.

        Parameters
        ----------
        cwd : str, optional
            path to repo, by default '.'
        """
        commands = {
            'commit': "git log -1",
            'diff': 'git diff',
            'status': 'git status -uno',
        }
        replace = {'"image/png": ".*?"': '"image/png": "..."'}
        self._env_meta_to_collect.append(dict(cwd=cwd, dst=None, replace=replace, commands=commands))
        return self

    def get_devices(self, devices):
        """ Return list if lists. Each sublist consists of devices for each branch.

        Parameters
        ----------
        devices : int, str, None or list of them
            devices to split between workers and branches. (see Example below)
        Returns
        -------
        list of lists of lists
            The first nesting level corresponds to workers.
            The second to branches.
            The third is a list of devices for current branch.
            For example, worker with index 2 and its branch with index 3 will get list of devices `devices[2][3]`.

        Examples
        --------
        For 3 workers and 2 branches::

            None -> [[[None], [None]], [[None], [None]], [[None], [None]]]
            1 -> [[['1'], ['1']], [['1'], ['1']], [['1'], ['1']]]
            [1, 2] -> [[['1'], ['1']], [['1'], ['2']], [['2'], ['2']]]
            [1, 2, 3, 4, 5] -> [[['1'], ['2']], [['3'], ['4']], [['5'], ['1']]]
            [0, 1, ..., 12] -> [[['0', '1'], ['2', '3']],
                                [['4', '5'], ['6', '7']],
                                [['8', '9'], ['10', '11']]]

        """
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
                x = to_list(x)
                values = [str(item) if isinstance(item, int) else item for item in x]
                return values if x is not None else []

            devices = [[_transform_item(branch_config) for branch_config in worker_config] for worker_config in devices]
        return devices


    def run(self, name=None, workers=1, branches=1, n_iters=None, devices=None, executor_class=Executor,
            dump_results=True, parallel=True, executor_target='threads', loglevel=None, bar=True, detach=False,
            debug=False, finalize=True, git_meta=False, env_meta=False, seed=None, profile=False,
            memory_ratio=None, n_gpu_checks=3, gpu_check_delay=5, create_id_prefix=False,
            redirect_stdout=True, redirect_stderr=True):
        """ Run research.

        Parameters
        ----------
        name : str, optional
            redefine name of the research (if needed), by default None.
        workers : int or list of Config instances, optional
            number of parallel workers, by default 1. If int, number of parallel workers to execute experiments.
            If list of Configs, list of configs for each worker which will be appended to configs from domain. Each
            element corresponds to one worker.
        branches : int or list of Config instances, optional
            number of different branches with different configs with the same root, by default 1.
            If list of Configs, list of configs for each branch which will be appended to configs from domain. Each
            element corresponds to one branch.
        n_iters : int, optional
            number of experiment iterations, by default None, None means that experiment will be executed until
            StopIteration exception.
        devices : str or list, optional
            devices to split between workers and branches, by default None.
        executor_class : Executor-inherited class, optional
            executor for experiments, by default None (means that Executor will be used).
        dump_results : bool, optional
            dump results or not, by default True.
        parallel : bool, optional
            execute experiments in parallel in separate processes or not, by default True.
        executor_target : 'for' or 'threads', optional
            how to execute branches, by default 'threads'.
        loglevel : str, optional
            logging level, by default 'debug'.
        bar : bool or class
            use or not progress bar.
        detach : bool, optional
            run research in separate process or not, by default False.
        debug : bool, optional
            If False, continue research after exceptions. If True, raise Exception. Can be used only with
            `parallel=False` and `executor_target='for'`, by default False.
        finalize : bool, optional
            continue experiment iteration after exception in some unit or not, by default True.
        git_meta : bool, optional
            attach get repo state or not (see :meth:`.Research.attach_git_meta`).
        env_meta : bool, optional
            attach env meta or not (see :meth:`.Research.attach_env_meta`).
        seed : bool or int or object with a seed sequence attribute
            see :meth:`~batchflow.utils_random.make_seed_sequence`.
        profile : bool, optional
            perform Research profiling or not, be default False.
        memory_ratio : float or None, optional
            the ratio of free memory for all devices in worker to start experiment. If None, check will be skipped.
        n_gpu_checks : int, optional
            the number of such checks
        gpu_check_delay : float, optional
            time in seconds between checks.
        create_id_prefix : bool or int, optional
            add prefix to experiment id to allow to sort them by the order of parameters in domain. If int,
            the number of digits for the parameter code formatting.
        redirect_stdout, redirect_stderr : int or bool, optional
            how to redirect stdout/stderr to files:
                0 or False - no redirection,
                True - redirect to common research file "stdout.txt"/"stderr.txt" when `dump_results=True`
                       or to separate items in `research.storage.experiments_stdout` when `dump_results=False`
                1 - redirect to common research file "stdout.txt"/"stderr.txt" (only when dump_results=True)
                2 - redirect output streams of experiments into separate file in experiments folders
                3 - redirect to common file and to separate experiments files (only when dump_results=True)

        Returns
        -------
        Research instance

        **How does it work**

        At each iteration all units of the experiment will be executed in the order in which were added.
        If `update_domain` callable is defined, domain will be updated with the corresponding function
        accordingly to `when` parameter of :meth:`~.Research.update_domain`.
        """
        self.name = name or self.name

        self.workers = workers
        self.branches = branches
        self.devices = self.get_devices(devices)
        self.executor_class = executor_class
        self.dump_results = dump_results
        self.parallel = parallel
        self.executor_target = executor_target
        self.loglevel = loglevel
        self.bar = bar
        self.detach = detach
        self.profile = profile

        self.memory_ratio = memory_ratio
        self.n_gpu_checks = n_gpu_checks
        self.gpu_check_delay = gpu_check_delay
        self.create_id_prefix = create_id_prefix

        if redirect_stdout is True and not dump_results:
            redirect_stdout = 2

        if redirect_stderr is True and not dump_results:
            redirect_stderr = 2

        self.redirect_stdout = redirect_stdout
        self.redirect_stderr = redirect_stderr

        if debug and (parallel or executor_target not in ['f', 'for']):
            raise ValueError("`debug` can be True only with `parallel=False` and `executor_target='for'`")

        if not dump_results and redirect_stdout in (1, 3):
            raise ValueError("`redirect_stdout` can be 0 or 2 only when `dump_results` is False")

        if not dump_results and redirect_stderr in (1, 3):
            raise ValueError("`redirect_stderr` can be 0 or 2 only when `dump_results` is False")

        self.debug = debug
        self.finalize = finalize
        self.random_seed = make_seed_sequence(seed)

        if n_iters is None and self.experiment.only_callables:
            self.n_iters = 1
        else:
            self.n_iters = n_iters

        self.domain.set_iter_params(n_items=self.n_configs, n_reps=self.n_reps, repeat_each=self.repeat_each,
                                    create_id_prefix=self.create_id_prefix, seed=self.random_seed)

        if self.domain.size is None and (self.domain.update_func is None or self.domain.update_each == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)

        self.storage = self.dump_results
        if isinstance(self.storage, bool):
            self.storage = 'local' if self.storage else 'memory'
        if isinstance(self.storage, str):
            self.storage = BaseResearchStorage(self, self.loglevel, storage=self.storage)

        if not isinstance(self.storage, MemoryResearchStorage):
            self.experiment = self.experiment.dump() # add final dump of experiment results

        self.logger = self.storage.logger

        if git_meta:
            self.attach_git_meta()
        if env_meta:
            self.attach_env_meta()

        self.storage.collect_env_state(self._env_meta_to_collect)

        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        self.tasks_queue = DynamicQueue(self.domain, self, n_branches)
        self.distributor = Distributor(self.tasks_queue, self)

        self.monitor = ResearchMonitor(self, bar=self.bar) # process execution signals

        def _start_distributor():
            self.distributor.run()
            self.monitor.stop()

        self.logger.info("Research is starting")

        self.monitor.start()
        if self.parallel:
            try:
                self.process = mp.Process(target=_start_distributor)
                self.process.start()
                self.logger.info(f"Create separate research process [pid:{self.process.pid}]")
                self.monitor.detach(self.process)
                if not detach:
                    self.process.join()
                    self.terminate()
            except KeyboardInterrupt as e:
                self.logger.info("Research has been stopped by KeyboardInterrupt.")
                self.terminate(force=True, wait=False)
                raise e
        else:
            if detach:
                warnings.warn("detach can't be enabled when parallel=False")
            _start_distributor()
            self.terminate()
        return self

    @property
    def results(self):
        return self.storage.results

    @property
    def profiler(self):
        return self.storage.profiler

    def terminate(self, kill_processes=False, force=False, wait=True):
        """ Kill all research processes. """
        if not self.is_loaded:
            if not force and self.monitor and self.monitor.in_progress:
                answer = input(f'{self.name} is in progress. Are you sure? [y/n]').lower()
                answer = len(answer) > 0 and 'yes'.startswith(answer)
            else:
                answer = True

            if force or answer:
                if self.logger:
                    self.logger.info("Stop research.")
                if self.monitor:
                    self.monitor.stop(wait=wait)
                    self.monitor.close()
                if self.storage:
                    self.storage.close()

                if self.detach:
                    kill_processes = True

                if kill_processes and self.monitor:
                    if self.logger:
                        self.logger.info("Terminate research processes")

                    order = {'EXECUTOR': 1, 'WORKER': 2, 'DETACHED_PROCESS': 3, 'MONITOR': 4}
                    processes_to_kill = sorted(self.monitor.processes.items(), key=lambda x: order[x[1]])
                    for pid, process_type in processes_to_kill:
                        if pid is not None and psutil.pid_exists(pid):
                            process = psutil.Process(pid)
                            process.terminate()
                            if self.logger:
                                self.logger.info(f"Terminate {process_type} [pid:{pid}]")

    @property
    def is_finished(self):
        """ Whether all tasks are completed or not. """
        return self.monitor.in_queue + self.monitor.remained_experiments == 0

    def __str__(self):
        spacing = ' ' * 4
        repr = ''

        params = ['name', 'workers', 'branches', 'n_iters', 'devices', 'dump_results',
                  'parallel', 'loglevel', 'executor_target', 'executor_class']
        params_repr = []
        for param in params:
            params_repr += [f"{param}: {getattr(self, param, None)}"]
        params_repr = '\n'.join(params_repr)

        items = {'params': params_repr, 'experiment': str(self.experiment), 'domain': str(self.domain)}
        for name in items:
            repr += f"{name}:\n"
            repr += '\n'.join([spacing + item for item in str(items[name]).split('\n')])
            repr += 2 * '\n'

        return repr

    @classmethod
    def load(cls, name):
        """ Load research. """
        storage = LocalResearchStorage(name, loglevel='info', mode='r')
        return storage.research

    @classmethod
    def remove(cls, name, ask=True, force=False):
        LocalResearchStorage.remove(name, ask, force)

    def __del__(self):
        self.terminate(force=True)

class ResearchMonitor:
    #pylint:disable=attribute-defined-outside-init
    """ Class to get signals from experiment and other objects and store all states.

    Parameters
    ----------
    research : Research
        Research object
    path : str, optional
        path to save signals, by default None
    bar : bool or class
        use progress bar or not.
    """
    COLUMNS = ['time', 'task_idx', 'id', 'it', 'name', 'status', 'exception', 'worker', 'pid', 'worker_pid',
               'process_pid', 'finished', 'withdrawn', 'remains']
    SHARED_VARIABLES = ['finished_experiments', 'finished_iterations', 'remained_experiments',
                        'generated_experiments', 'stopped']

    def __init__(self, research, bar=True):
        self.queue = mp.JoinableQueue()
        self.stop_signal = mp.JoinableQueue()
        self._manager = mp.Manager()
        self.exceptions = self._manager.list()
        self.shared_values = self._manager.dict()
        self.current_iterations = self._manager.dict()
        self.processes = self._manager.dict({self._manager._process.pid: "MANAGER"})

        self.research = research
        self.bar = tqdm.tqdm(disable=(not bar), position=0, leave=True) if isinstance(bar, bool) else bar

        for key in self.SHARED_VARIABLES:
            self.shared_values[key] = 0

        self.n_iters = self.research.n_iters

        self.dump = False
        self.process = None
        self.stopped = True

    def __getattr__(self, key):
        if key in self.SHARED_VARIABLES:
            return self.shared_values[key]
        raise AttributeError(f'Unknown attribute: {key}')

    def __setattr__(self, key, value):
        if key in self.SHARED_VARIABLES:
            self.shared_values[key] = value
        else:
            super().__setattr__(key, value)

    @property
    def total(self):
        """ Total number of iterations or experiments in the current moment. It changes after domain updates. """
        if self.n_iters:
            return self.finished_iterations + self.n_iters * (self.in_queue + self.remained_experiments)
        return self.finished_experiments + self.in_queue + self.remained_experiments

    @property
    def n(self):
        """ Current iteration. """
        if self.n_iters:
            return self.finished_iterations + sum(self.current_iterations.values())
        return self.finished_experiments + len(self.current_iterations)

    @property
    def in_progress(self):
        """ The number of experiments in progress. """
        return len(self.current_iterations)

    @property
    def in_queue(self):
        """ The number of experiments in queue of tasks. """
        return self.generated_experiments - self.finished_experiments

    def detach(self, process):
        self.processes[process.pid] = 'DETACHED_PROCESS'

    def start_worker(self, worker):
        self.processes[worker.pid] = 'WORKER'

    def start_experiment(self, experiment):
        """" Signal when experiment starts. """
        self.processes[experiment.executor.pid] = 'EXECUTOR'
        self.current_iterations[experiment.id] = 0

    def tasks_info(self, generated, remains):
        self.generated_experiments = generated
        self.remained_experiments = remains

    def stop_experiment(self, experiment):
        """" Signal when experiment stops. """
        self.current_iterations.pop(experiment.id)
        self.finished_iterations += experiment.iteration + 1
        self.finished_experiments += 1

    def execute_iteration(self, experiment):
        """" Signal for iteration execution. """
        self.current_iterations[experiment.id] = experiment.iteration

    def fail_item_execution(self, name, experiment, msg):
        """" Signal for iteration execution fail. """
        self.exceptions.append({
            'id': experiment.id,
            'pid': experiment.executor.pid,
            'name': name,
            'it': experiment.iteration,
            'exception': msg
        })

    def fail_worker_execution(self, worker, msg):
        self.exceptions.append({
            'index': worker.index,
            'pid': worker.pid,
            'exception': msg
        })

    def handler(self):
        """ Signals handler. """
        with self.bar as progress:
            last_update = False
            exceptions = 0
            while True:
                if (progress.n != self.n) or (progress.total != self.total) or (len(self.exceptions) != exceptions):
                    if len(self.exceptions) != exceptions:
                        exceptions = len(self.exceptions)
                        progress.set_description_str(f"Exceptions: {exceptions}")
                    progress.n = self.n
                    progress.total = self.total
                    progress.refresh()
                if last_update:
                    break
                time.sleep(0.01)
                if not self.queue.empty():
                    last_update = True
        self.stop_signal.put(None)

    def start(self):
        """ Start handler. """
        if self.stopped:
            self.process = mp.Process(target=self.handler)
            self.process.start()
            self.processes[self.process.pid] = 'MONITOR'
            self.stopped = False

    def stop(self, wait=True):
        """ Stop handler. """
        if not self.stopped:
            self.queue.put(None)
            if wait:
                self.stop_signal.get()
            self.stopped = True
        tqdm.tqdm._instances.clear() #pylint:disable=protected-access

    def close(self):
        """ Close manager. """
        self.exceptions = list(self.exceptions)
        self.shared_values = dict(self.shared_values)
        self.current_iterations = dict(self.current_iterations)
        self.processes = dict(self.processes)
        self._manager.shutdown()
