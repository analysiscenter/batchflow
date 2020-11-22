""" Classes for multiprocess job running. """

import os
import logging
import multiprocess as mp
from tqdm import tqdm

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

class Distributor:
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

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        bar : bool or callable

        args, kwargs
            will be used in worker
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
            if len(workers) > 1:
                msg = 'Run {} workers'
            else:
                msg = 'Run {} worker'
            self.logger.info(msg.format(len(workers)))
            for worker in workers:
                try:
                    mp.Process(target=worker, args=(self.jobs_queue, self.results)).start()
                except Exception as exception: #pylint:disable=broad-except
                    self.logger.error(exception)
            previous_domain_jobs = 0
            n_updates = 0
            finished_iterations = dict()
            with bar(total=None) as progress:
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
