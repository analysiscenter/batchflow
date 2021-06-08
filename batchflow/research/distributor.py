""" Classes for multiprocess job running. """

import os
import multiprocess as mp

from .. import Config
from .domain import Domain

class DynamicQueue:
    """ Queue of tasks that can be changed depending on previous results. """
    def __init__(self, domain, research, n_branches):
        self._domain = domain
        self.research = research
        self.n_branches = n_branches

        self.queue = mp.JoinableQueue()

        self.configs_generated = 0
        self.configs_remains = self.domain.size

        self.finished_tasks = 0
        self.tasks_in_queue = 0

    @property
    def domain(self):
        """ Get (or create if needed) domain. """
        if self._domain.size == 0:
            domain = Domain({'repetition': [None]}) # the value of repetition will be rewritten
            domain.set_iter_params(n_reps=self._domain.n_reps, produced=self._domain.n_produced)
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

    def join(self):
        self.queue.join()

    def get(self):
        return self.queue.get()

    def put(self, *args, **kwargs):
        return self.queue.put(*args, **kwargs)

    def task_done(self):
        return self.queue.task_done()

    def empty(self):
        return self.queue.empty()

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
    responses : multiprocess.JoinableQueue
        queue for responses aboute executed tasks
    research : Research
        research
    """
    def __init__(self, index, worker_config, tasks, responses, research):
        self.index = index
        self.worker_config = worker_config
        self.tasks = tasks
        self.responses = responses
        self.research = research

        self.pid = None

    def __call__(self):
        self.pid = os.getpid()
        self.research.logger.info(f"Worker {self.index}[pid:{self.pid}] has started.")

        executor_class = self.research.executor_class
        n_iters = self.research.n_iters
        self.research.monitor.send(status='START_WORKER', worker=self)
        task = self.tasks.get()

        if isinstance(self.research.branches, int):
            branches_configs = [Config() for _ in range(self.research.branches)]
        else:
            branches_configs = self.research.branches
        n_branches = len(branches_configs)
        devices = self.research.devices[self.index]

        all_devices = set(device for i in range(n_branches) for device in devices[i] if device is not None)

        if len(all_devices) > 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(all_devices)

        device_reindexation = {device: i for i, device in enumerate(all_devices)}
        devices = [[device_reindexation.get(i) for i in item] for item in devices]
        devices = [{'device': item[0] if len(item) == 1 else item} for item in devices]

        while task is not None:
            task_idx, configs = task

            self.research.monitor.send(worker=self, status='GET_TASK', task_idx=task_idx)
            self.research.logger.info(f"Worker {self.index}[pid:{self.pid}] have got task {task_idx}.")
            name = f"Task {task_idx}"

            experiment = self.research.experiment
            target = self.research.executor_target # 'for' or 'thread' for branches execution

            branches_configs = [config + Config(_devices) for config, _devices in zip(branches_configs, devices)]
            branches_configs = branches_configs[:len(configs)]

            executor = executor_class(experiment, research=self.research, target=target, configs=configs,
                                      branches_configs=branches_configs, executor_config=self.worker_config,
                                      n_iters=n_iters, task_name=name)
            if self.research.parallel:
                process = mp.Process(target=executor.run, args=(self, ))
                process.start()
                process.join()
            else:
                executor.run(self)
            self.research.monitor.send(worker=self, status='FINISH_TASK', task_idx=task_idx)
            self.tasks.task_done()
            self.responses.put((self.index, task_idx))
            task = self.tasks.get()
        self.tasks.task_done()
        self.research.monitor.send(worker=self, status='STOP_WORKER')

        self.research.logger.info(f"Worker {self.index}[pid:{self.pid}] has stopped.")

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
        self.responses = mp.JoinableQueue()

    def run(self):
        """ Run disributor and all workers. """
        workers = []
        if isinstance(self.research.workers, int):
            worker_configs = [Config() for _ in range(self.research.workers)]
        else:
            worker_configs = self.research.workers
        for i, worker_config in enumerate(worker_configs):
            workers += [Worker(i, worker_config, self.tasks, self.responses, self.research)]

        if not self.research.parallel:
            workers = workers[:1]

        self.tasks.next_tasks(len(workers))
        self.research.logger.info(f'Start workers (parallel={self.research.parallel})')

        if self.research.parallel:
            for worker in workers:
                mp.Process(target=worker).start()

            self.send_state()
            while self.tasks.tasks_in_queue > 0:
                self.responses.get()
                self.tasks.finished_tasks += 1
                self.tasks.tasks_in_queue -= 1

                self.tasks.update_domain()
                self.tasks.next_tasks(1)
                self.send_state()

            self.tasks.stop_workers(len(workers))
            for _ in workers:
                self.tasks.join()
        else:
            self.send_state()
            while self.tasks.tasks_in_queue > 0:
                self.tasks.stop_workers(1) # worker can't be in separate process so we restart it after each task
                workers[0]()

                self.tasks.finished_tasks += 1
                self.tasks.tasks_in_queue -= 1

                self.tasks.update_domain()
                self.tasks.next_tasks(1)
                self.send_state()

        self.research.logger.info('All workers have finished the work')

    def send_state(self):
        self.research.monitor.send('TASKS', generated=self.tasks.configs_generated, remains=self.tasks.configs_remains)
