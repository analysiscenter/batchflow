""" Classes for multiprocess job running. """

import os
import csv
import logging
import multiprocess as mp
from tqdm import tqdm
from collections import OrderedDict
import datetime


class _DummyBar:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self.n = 0
        self.total = None

    def set_description(self, *args, **kwargs):
        pass

    def refresh(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

class _Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, n_iters, workers, devices, worker_class=None, timeout=None, trials=2, logger=None):
        """
        Parameters
        ----------
        workers : int or list of Worker configs

        worker_class : Worker subclass or None
        """
        self.n_iters = n_iters
        self.workers = workers
        self.devices = devices
        self.worker_class = worker_class
        self.timeout = timeout
        self.trials = trials
        self.logger = logger

        self.logfile = None
        self.errorfile = None
        self.results = None
        self.finished_jobs = None
        self.answers = None
        self.jobs_queue = None

    def run(self, jobs_queue, bar=False):
        """ Run disributor and workers.

        Parameters
        ----------
        jobs_queue : DynamicQueue of tasks

        n_iters : int or None

        bar : bool or callable
        """
        self.jobs_queue = jobs_queue

        if isinstance(bar, bool):
            bar = tqdm if bar else _DummyBar()

        self.logger.info('Distributor [id:{}] is preparing workers'.format(os.getpid()))

        if isinstance(self.workers, int):
            workers = [self.worker_class(
                devices=self.devices[i],
                worker_name=i,
                timeout=self.timeout,
                trials=self.trials,
                logger=self.logger
                )
                       for i in range(self.workers)]
        else:
            workers = [
                self.worker_class(
                    devices=self.devices[i],
                    worker_name=i,
                    timeout=self.timeout,
                    trials=self.trials,
                    logger=self.logger,
                    worker_config=worker_config
                    )
                for i, worker_config in enumerate(self.workers)
            ]
        try:
            self.logger.info('Create queue of jobs')
            self.results = mp.JoinableQueue()
        except Exception as exception: #pylint:disable=broad-except
            self.logger.error(exception)
        else:
            msg = 'Run {} workers' if len(workers) > 1 else 'Run {} worker'
            self.logger.info(msg.format(len(workers)))
            for worker in workers:
                try:
                    mp.Process(target=worker, args=(self.jobs_queue, self.results)).start()
                except Exception as exception: #pylint:disable=broad-except
                    self.logger.error(exception)
            previous_domain_jobs = 0
            n_updates = 0
            finished_iterations = dict()
            with tqdm(total=None, disable=(not bar)) as progress:
                while True:
                    n_jobs = self.jobs_queue.next_jobs(len(workers)+1)
                    jobs_in_queue = n_jobs
                    finished_jobs = 0
                    rest_of_generator = 0
                    while finished_jobs != jobs_in_queue:
                        progress.set_description('Domain updated: ' + str(n_updates))

                        estimated_size = self.jobs_queue.total
                        if estimated_size is not None:
                            total = rest_of_generator + previous_domain_jobs + estimated_size
                            if self.n_iters is not None:
                                total *= self.n_iters
                            progress.total = total
                        signal = self.results.get()
                        if self.n_iters is not None:
                            finished_iterations[signal.job] = signal.iteration
                        if signal.done:
                            finished_jobs += 1
                            finished_iterations[signal.job] = self.n_iters
                            each = self.jobs_queue.domain.update_each
                            if isinstance(each, int) and finished_jobs % each == 0:
                                was_updated = self.jobs_queue.update()
                                if was_updated:
                                    rest_of_generator = jobs_in_queue
                                n_updates += was_updated
                            if n_jobs > 0:
                                n_jobs = self.jobs_queue.next_jobs(1)
                                jobs_in_queue += n_jobs
                        if self.n_iters is not None:
                            progress.n = sum(finished_iterations.values())
                        else:
                            progress.n += signal.done
                        progress.refresh()
                    if self.jobs_queue.domain.update_each == 'last':
                        was_updated = self.jobs_queue.update()
                        n_updates += 1
                        if not was_updated:
                            break
                    else:
                        self.jobs_queue.stop_workers(len(workers))
                        self.jobs_queue.join()
                        break
                    previous_domain_jobs += finished_jobs
        self.logger.info('All workers have finished the work')
        logging.shutdown()

class Signal:
    """ Class for feedback from jobs and workers """
    def __init__(self, worker, job, iteration, n_iters, trial, done, exception, exec_actions=None, dump_actions=None):
        self.worker = worker
        self.job = job
        self.iteration = iteration
        self.n_iters = n_iters
        self.trial = trial
        self.done = done
        self.exception = exception
        self.exec_actions = exec_actions
        self.dump_actions = dump_actions

    def __repr__(self):
        return str(self.__dict__)

class Worker:
    def __init__(self, index, worker_config, tasks, research):
        self.index = index
        self.worker_config = worker_config
        self.tasks = tasks
        self.research = research

    def __call__(self):
        self.pid = os.getpid()
        executor_class = self.research.executor_class
        n_iters = self.research.n_iters
        self.research.monitor.send(worker=self, status='start')
        task = self.tasks.get()
        while task is not None:
            task_idx, executor_configs = task
            self.research.monitor.send(worker=self, status='get task', task_idx=task_idx)
            name = f"Task {task_idx}"
            executor = executor_class(self.research.experiment, research=self.research, target=self.research.executor_target,
                                      configs=executor_configs, n_iters=n_iters, task_name=name)
            if self.research.parallel:
                process = mp.Process(target=executor.run, args=(self, ))
                process.start()
                process.join()
            else:
                executor.run(self)
            self.research.monitor.send(worker=self, status='finish task', task_idx=task_idx)
            self.tasks.task_done()
            print('Worker finish task')
            task = self.tasks.get()
        self.tasks.task_done()
        self.research.monitor.send(worker=self, status='stop')

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

    def run(self, bar=False):
        """ Run disributor and workers.

        Parameters
        ----------
        jobs_queue : DynamicQueue of tasks

        n_iters : int or None

        bar : bool or callable
        """
        workers = []
        for i, worker_config in enumerate(self.research.workers):
            workers += [Worker(i, worker_config, self.tasks, self.research)]

        n_tasks = self.tasks.next_tasks(len(workers)+1)
        if self.research.parallel:
            for worker in workers:
                mp.Process(target=worker).start()
            while n_tasks > 0:
                self.tasks.join()
                self.research.results.get()
                n_tasks = self.tasks.next_tasks(1)
            self.tasks.stop_workers(len(workers))
            self.tasks.join()
        else:
            while n_tasks > 0:
                workers[0]()
                self.research.results.get()
                n_tasks = self.tasks.next_tasks(1)
            workers[0]()

class ResearchMonitor:
    COLUMNS = ['time', 'task_idx', 'id', 'it', 'name', 'status', 'exception', 'worker', 'pid', 'worker_pid']
    def __init__(self, path):
        self.queue = mp.JoinableQueue()
        self.path = path

    def send(self, experiment=None, worker=None, **kwargs):
        signal = {
            'time': str(datetime.datetime.now()),
            **kwargs
        }
        if experiment is not None:
            signal = {**signal, **{
                'id': experiment.id,
                'pid': experiment.executor.pid,
            }}
        if worker is not None:
            signal = {**signal, **{
                'worker': worker.index,
                'worker_pid': worker.pid,
            }}
        self.queue.put(signal)

    # def start_execution(self, name, experiment):
    #     self.send(experiment, name=name, it=experiment.iteration, status='start')

    def finish_execution(self, name, experiment):
        self.send(experiment, experiment.executor.worker, name=name, it=experiment.iteration, status='success')

    def fail_execution(self, name, experiment):
        self.send(experiment, experiment.executor.worker, name=name, it=experiment.iteration, status='error', exception=experiment.exception.__class__)

    def stop_iteration(self, name, experiment):
        self.send(experiment, experiment.executor.worker, name=name, it=experiment.iteration, status='stop_iteration')

    def listener(self): #TODO: rename
        filename = os.path.join(self.path, 'monitor.csv')
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

        signal = self.queue.get()
        while signal is not None:
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([str(signal.get(column, '')) for column in self.COLUMNS])
            signal = self.queue.get()

    def start(self):
        mp.Process(target=self.listener).start()

    def stop(self):
        self.queue.put(None)

class ResearchResults:
    def __init__(self):
        self.queue = mp.JoinableQueue()
        self.results = OrderedDict()

    def put(self, id, results):
        self.queue.put((id, results))

    def get(self):
        id, results = self.queue.get()
        self.results[id] = results
