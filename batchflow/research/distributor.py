""" Classes for multiprocess job running. """

import os
import logging
import multiprocess as mp
from tqdm import tqdm

class Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, workers, devices, worker_class=None, timeout=5, trials=3):
        """
        Parameters
        ----------
        workers : int or list of Worker configs

        worker_class : Worker subclass or None
        """
        self.workers = workers
        self.devices = devices
        self.worker_class = worker_class
        self.timeout = timeout
        self.trials = trials

        self.logfile = None
        self.errorfile = None
        self.results = None
        self.finished_jobs = None
        self.answers = None
        self.queue = None

    @classmethod
    def log_info(cls, message, filename):
        """ Write message into log. """
        logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
        logging.info(message)

    @classmethod
    def log_error(cls, obj, filename):
        """ Write error message into log. """
        logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
        logging.error(obj, exc_info=True)

    def run(self, queue, dirname, n_jobs, n_iters, logfile=None, errorfile=None, bar=False, *args, **kwargs):
        """ Run disributor and workers.

        Parameters
        ----------
        queue : queue of tasks

        dirname : str

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        bar : bool or callable

        args, kwargs
            will be used in worker
        """
        self.queue = queue

        if isinstance(bar, bool):
            bar = tqdm if bar else None

        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

        self.logfile = os.path.join(dirname, self.logfile)
        self.errorfile = os.path.join(dirname, self.errorfile)

        kwargs['logfile'] = self.logfile
        kwargs['errorfile'] = self.errorfile

        self.log_info('Distributor [id:{}] is preparing workers'.format(os.getpid()), filename=self.logfile)

        if isinstance(self.workers, int):
            workers = [self.worker_class(
                devices=self.devices[i],
                worker_name=i,
                timeout=self.timeout,
                trials=self.trials,
                *args, **kwargs
                )
                       for i in range(self.workers)]
        else:
            workers = [
                self.worker_class(
                    devices=self.devices[i],
                    worker_name=i,
                    timeout=self.timeout,
                    trials=self.trials,
                    worker_config=worker_config,
                    *args, **kwargs)
                for i, worker_config in enumerate(self.workers)
            ]
        try:
            self.log_info('Create queue of jobs', filename=self.logfile)
            self.results = mp.JoinableQueue()
        except Exception as exception: #pylint:disable=broad-except
            logging.error(exception, exc_info=True)
        else:
            if len(workers) > 1:
                msg = 'Run {} workers'
            else:
                msg = 'Run {} worker'
            self.log_info(msg.format(len(workers)), filename=self.logfile)
            for worker in workers:
                self.queue.put(None)
                worker.log_info = self.log_info
                worker.log_error = self.log_error
                try:
                    mp.Process(target=worker, args=(self.queue, self.results)).start()
                except Exception as exception: #pylint:disable=broad-except
                    logging.error(exception, exc_info=True)

            self.answers = [0 for _ in range(n_jobs)]
            self.finished_jobs = []

            if bar is not None:
                if n_iters is not None:
                    print("Distributor has {} jobs with {} iterations. Totally: {}"
                          .format(n_jobs, n_iters, n_jobs*n_iters), flush=True)
                    with bar(total=n_jobs*n_iters) as progress:
                        while True:
                            signal = self.results.get()
                            position = self._get_position(signal)
                            if signal.done:
                                self.finished_jobs.append(signal.job)
                            progress.n = position
                            progress.refresh()
                            if len(self.finished_jobs) == n_jobs:
                                break
                else:
                    print("Distributor has {} jobs"
                          .format(n_jobs), flush=True)
                    with bar(total=n_jobs) as progress:
                        while True:
                            answer = self.results.get()
                            if answer.done:
                                self.finished_jobs.append(answer.job)
                            position = len(self.finished_jobs)
                            progress.n = position
                            progress.refresh()
                            if len(self.finished_jobs) == n_jobs:
                                break
            else:
                self.queue.join()
        self.log_info('All workers have finished the work', filename=self.logfile)
        logging.shutdown()

    def _get_position(self, signal, fixed_iterations=True):
        if fixed_iterations:
            if signal.done:
                self.answers[signal.job] = signal.n_iters
            else:
                self.answers[signal.job] = signal.iteration+1
        else:
            self.answers[signal.job] += 1
        return sum(self.answers)

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
