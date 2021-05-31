""" Classes for multiprocess job running. """

import os
import multiprocess as mp

from .. import Config


class Worker:
    def __init__(self, index, worker_config, tasks, responses, research):
        self.index = index
        self.worker_config = worker_config
        self.tasks = tasks
        self.research = research
        self.responses = responses

    def __call__(self):
        self.pid = os.getpid()
        self.research.logger.info(f"Worker {self.index}[{self.pid}] has started.")

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
            self.research.logger.info(f"Worker {self.index}[{self.pid}] have got task {task_idx}.")
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

        self.research.logger.info(f"Worker {self.index}[{self.pid}] has stopped.")

class Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, tasks, research):
        """
        Parameters
        ----------
        workers : int or list of Worker configs

        worker_class : Worker subclass or None
        """
        self.tasks = tasks
        self.research = research
        self.responses = mp.JoinableQueue()

    def run(self):
        """ Run disributor and workers.

        Parameters
        ----------
        jobs_queue : DynamicQueue of tasks

        n_iters : int or None

        bar : bool or callable
        """
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