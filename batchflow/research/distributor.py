""" Classes for multiprocess job running. """

import os
import multiprocess as mp

from .. import Config


class Worker:
    def __init__(self, index, worker_config, tasks, research):
        self.index = index
        self.worker_config = worker_config
        self.tasks = tasks
        self.research = research

    def __call__(self):
        self.pid = os.getpid()
        self.research.logger.info(f"Worker {self.index}[{self.pid}] has started.")

        executor_class = self.research.executor_class
        n_iters = self.research.n_iters
        self.research.monitor.send(worker=self, status='start')
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

            self.research.monitor.send(worker=self, status='get task', task_idx=task_idx)
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
            self.research.monitor.send(worker=self, status='finish task', task_idx=task_idx)
            self.tasks.task_done()
            task = self.tasks.get()
        self.tasks.task_done()
        self.research.monitor.send(worker=self, status='stop')

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
            worker_configs = workers
        for i, worker_config in enumerate(worker_configs):
            workers += [Worker(i, worker_config, self.tasks, self.research)]

        self.tasks.next_tasks(len(workers)+1)
        if self.research.parallel:
            self.research.logger.info('Start workers (parallel)')
            for worker in workers:
                mp.Process(target=worker).start()

            self.send()
            while self.tasks.in_progress():
                self.tasks.join()
                self.tasks.update_domain()
                self.send()
                self.tasks.next_tasks(1)

            self.send()
            self.tasks.stop_workers(len(workers))
            for _ in workers:
                self.tasks.join(inc=False)
        else:
            self.research.logger.info('Start workers (no parallel)')
            while self.tasks.in_progress():
                self.tasks.stop_workers(1) # worker can't be in separate process so we restart it after each task
                workers[0]()
                self.tasks.finished_tasks += 1 # instead of join
                self.tasks.update_domain()
                self.send()
                self.tasks.next_tasks(1)
        self.research.logger.info('All workers have finished the work')

    def send(self):
        self.research.monitor.send(
            finished=self.tasks.finished_tasks,
            withdrawn=self.tasks.withdrawn_tasks,
            remains=self.tasks.remains
        )