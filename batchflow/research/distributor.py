""" Classes for multiprocess job running. """

import os
import time
import itertools
import traceback
import multiprocess as mp

import numpy as np

from .. import Config
from .domain import Domain
from .utils import generate_id
from ..utils_random import make_rng, spawn_seed_sequence
from ..utils_notebook import get_gpu_free_memory

class DynamicQueue:
    """ Queue of tasks that can be changed depending on previous results. """
    def __init__(self, domain, research, n_branches):
        self._domain = domain
        self.research = research
        self.n_branches = n_branches

        seed = spawn_seed_sequence(research)
        self.random_seed = seed
        self.random = make_rng(seed)

        self.queue = mp.JoinableQueue()
        self.done_flag = mp.JoinableQueue()

        self.configs_generated = 0
        self.configs_remains = self.domain.size

        self.finished_tasks = 0
        self.tasks_in_queue = 0

    @property
    def domain(self):
        """ Get (or create if needed) domain. """
        if self._domain.size == 0:
            domain = Domain({'repetition': [None]}) # the value of repetition will be rewritten
            domain.set_iter_params(n_reps=self._domain.n_reps, produced=self._domain.n_produced,
                                   create_id_prefix=self._domain.create_id_prefix, seed=self.random_seed)
            self._domain = domain
        return self._domain

    def update_domain(self):
        """ Update domain. """
        new_domain = self.domain.update(self.configs_generated, self.research)
        if new_domain is not None:
            self._domain = new_domain
            self.configs_remains = self._domain.size
        return new_domain is not None

    def next_tasks(self, n_tasks=1):
        """ Get next `n_tasks` elements of queue. """
        configs = []
        for i in range(n_tasks):
            branches_tasks = [] # TODO: rename it
            try:
                for _ in range(self.n_branches):
                    config = next(self.domain)
                    config['id'] = generate_id(config, self.random, self.research.create_id_prefix)
                    branches_tasks.append(config)
                configs.append(branches_tasks)
            except StopIteration:
                if len(branches_tasks) > 0:
                    configs.append(branches_tasks)
                break
        for i, executor_configs in enumerate(configs):
            self.put((self.configs_generated + i, executor_configs))

        n_configs = sum([len(item) for item in configs])

        self.configs_generated += n_configs
        self.configs_remains -= n_configs
        self.tasks_in_queue += len(configs)
        self.research.logger.info(f'Get {n_tasks} tasks with {n_configs} configs, remains {self.configs_remains}')

        return n_configs

    def stop_workers(self, n_workers):
        """ Stop all workers by putting `None` task into queue. """
        for _ in range(n_workers):
            self.put(None)

    def task_done(self):
        self.queue.task_done()
        self.done_flag.put(None)

    def worker_failed(self):
        self.queue.task_done()
        self.done_flag.put('error')

    def wait_for_finished_task(self):
        flag = self.done_flag.get()
        return 0 if flag is None else 1

    def __getattr__(self, key):
        return getattr(self.queue, key)

class Worker:
    """ Worker to get tasks from queue and run executors.

    Parameters
    ----------
    index : int
        numerical index of the worker (from 0)
    worker_config : Config
        additional config for all experiments executed in Worker
    tasks : DynamicQueue
        tasks queue
    research : Research
        research
    """
    def __init__(self, index, worker_config, tasks, research):
        self.index = index
        self.worker_config = worker_config
        self.tasks = tasks
        self.research = research

        self.pid = None

        seed = spawn_seed_sequence(research)
        self.random_seed = seed
        self.random = make_rng(seed)

    def __call__(self):
        exception = KeyboardInterrupt if self.research.debug else Exception
        try:
            self.pid = os.getpid() if self.research.parallel else None
            self.research.logger.info(f"Worker {self.index}[pid:{self.pid}] has started.")
            _return = True

            executor_class = self.research.executor_class
            n_iters = self.research.n_iters

            if isinstance(self.research.branches, int):
                branches_configs = [Config() for _ in range(self.research.branches)]
            else:
                branches_configs = self.research.branches
            n_branches = len(branches_configs)
            devices = self.research.devices[self.index]

            all_devices = np.unique([device for i in range(n_branches) for device in devices[i] if device is not None])

            if len(all_devices) > 0:
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(all_devices)

            device_reindexation = {device: str(i) for i, device in enumerate(all_devices)}
            devices = [[device_reindexation.get(i) for i in item] for item in devices]
            devices = [{'device': item[0] if len(item) == 1 else item} for item in devices]

            while True:
                bad_devices, bad_memory = self._check_memory(all_devices)
                if len(bad_devices) > 0:
                    msg = f"Worker {self.index}[pid:{self.pid}]: devices {bad_devices} " \
                        f"don't have enough memory: {bad_memory}"

                    self.research.logger.info(msg)
                    _return = False
                    break
                task = self.tasks.get()
                if task is None:
                    self.tasks.task_done()
                    break
                task_idx, configs = task

                name = f"Task {task_idx}"

                experiment = self.research.experiment
                target = self.research.executor_target # 'for' or 'thread' for branches execution

                branches_configs = [config + Config(_devices) for config, _devices in zip(branches_configs, devices)]
                branches_configs = branches_configs[:len(configs)]

                executor = executor_class(experiment, research=self.research, worker=self, target=target,
                                          configs=configs, branches_configs=branches_configs,
                                          executor_config=self.worker_config, n_iters=n_iters, task_name=name)
                if self.research.parallel:
                    process = mp.Process(target=executor.run)
                    process.start()
                    self.research.logger.info(
                        f"Worker {self.index} [pid:{self.pid}] has started task {task_idx} [pid:{process.pid}]."
                    )
                    process.join()
                else:
                    executor.run()
                self.tasks.task_done()
        except exception as e: #pylint: disable=broad-except
            ex_traceback = e.__traceback__
            msg = ''.join(traceback.format_exception(e.__class__, e, ex_traceback))
            self.research.logger.error(f"Fail worker {self.index}[pid:{self.pid}]: Exception\n{msg}")
            self.tasks.worker_failed()
            self.research.monitor.fail_worker_execution(self, msg)
        self.research.logger.info(f"Worker {self.index} [pid:{self.pid}] has stopped.")

        return _return

    def _devices_memory(self, devices):
        return np.array([get_gpu_free_memory(int(device)) for device in devices])

    def _check_memory(self, devices):
        memory_ratio = self.research.memory_ratio
        n_times = self.research.n_gpu_checks
        delay = self.research.gpu_check_delay

        if memory_ratio is None:
            return [], []

        times = 0
        while times < n_times:
            memory = self._devices_memory(devices)
            if (memory >= memory_ratio).all():
                return [], []

            msg = f"Worker {self.index}[pid:{self.pid}]: memory check failed (times: {times+1}/{n_times})"
            self.research.logger.info(msg)
            times += 1
            time.sleep(delay)

        bad_devices = devices[memory < memory_ratio]
        bad_memory = memory[memory < memory_ratio]
        return bad_devices, bad_memory

class Distributor:
    """ Distributor of jobs between workers.

    Parameters
    ----------
    tasks : DynamicQueue
        tasks queue
    research : Research
        research
    """
    def __init__(self, tasks, research):
        self.tasks = tasks
        self.research = research

    def run(self):
        """ Run disributor and all workers. """
        workers = []
        if isinstance(self.research.workers, int):
            worker_configs = [Config() for _ in range(self.research.workers)]
        else:
            worker_configs = self.research.workers
        for i, worker_config in enumerate(worker_configs):
            workers.append(Worker(i, worker_config, self.tasks, self.research))

        self.tasks.next_tasks(len(workers))
        self.research.logger.info(f'Start workers (parallel={self.research.parallel})')

        if self.research.parallel:
            processes = []
            for worker in workers:
                process = mp.Process(target=worker)
                process.start()
                processes.append(process)

            self.send_state()
            while self.tasks.tasks_in_queue > 0 and self.tasks.wait_for_finished_task() == 0:
                self.tasks.finished_tasks += 1
                self.tasks.tasks_in_queue -= 1

                self.tasks.update_domain()
                self.tasks.next_tasks(1)
                self.send_state()

            self.tasks.stop_workers(len(workers))
            for process in processes:
                process.join()

        else:
            self.send_state()
            _workers = itertools.cycle(workers)
            while self.tasks.tasks_in_queue > 0:
                self.tasks.stop_workers(1) # worker can't be in separate process so we restart it after each task
                worker = next(_workers)
                if worker():
                    self.tasks.finished_tasks += 1
                    self.tasks.tasks_in_queue -= 1

                    self.tasks.update_domain()
                    self.tasks.next_tasks(1)
                    self.send_state()

        self.research.logger.info('All workers have finished the work')

    def send_state(self):
        self.research.monitor.tasks_info(generated=self.tasks.configs_generated, remains=self.tasks.configs_remains)
