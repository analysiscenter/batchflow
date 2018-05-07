""" Classes for multiprocess job running. """

import os
import logging
import multiprocess as mp
from tqdm import tqdm

from .job import Job

class Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, n_workers, worker_class=None):
        """
        Parameters
        ----------
        n_workers : int or list of Worker instances

        worker_class : Worker subclass or None
        """
        self.n_workers = n_workers
        self.worker_class = worker_class

    def _jobs_to_queue(self, jobs):
        queue = mp.JoinableQueue()
        for idx, job in enumerate(jobs):
            queue.put((idx, Job(job)))
        for _ in range(self.n_workers):
            queue.put(None)
        return queue

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

    def run(self, jobs, dirname, n_jobs, logfile=None, errorfile=None, progress_bar=False, *args, **kwargs):
        """ Run disributor and workers.

        Parameters
        ----------
        jobs : iterable

        dirname : str

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        progress_bar : bool

        args, kwargs
            will be used in worker
        """
        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

        self.logfile = os.path.join(dirname, self.logfile)
        self.errorfile = os.path.join(dirname, self.errorfile)

        _tqdm = tqdm if progress_bar else lambda x: x

        kwargs['logfile'] = self.logfile
        kwargs['errorfile'] = self.errorfile

        self.log_info('Distributor [id:{}] is preparing workers'.format(os.getpid()), filename=self.logfile)

        if isinstance(self.n_workers, int):
            workers = [self.worker_class(worker_name=i, *args, **kwargs) for i in range(self.n_workers)]
        elif issubclass(type(self.n_workers[0]), Worker):
            for worker in self.n_workers:
                worker.set_args_kwargs(args, kwargs)
            workers = self.n_workers
            self.n_workers = len(self.n_workers)
        else:
            workers = [
                self.worker_class(worker_name=i, config=config, *args, **kwargs)
                for i, config in enumerate(self.n_workers)
            ]
            self.n_workers = len(self.n_workers)
        try:
            self.log_info('Create queue of jobs', filename=self.logfile)
            queue = self._jobs_to_queue(jobs)
            results = mp.JoinableQueue()
        except Exception as exception:
            logging.error(exception, exc_info=True)
        else:
            if len(workers) > 1:
                msg = 'Run {} workers.'
            else:
                msg = 'Run {} worker.'
            self.log_info(msg.format(len(workers)), filename=self.logfile)
            for worker in workers:
                worker.log_info = self.log_info
                worker.log_error = self.log_error

                try:
                    mp.Process(target=worker, args=(queue, results)).start()
                except Exception as exception:
                    logging.error(exception, exc_info=True)
            for _ in _tqdm(range(n_jobs)):
                results.get()
            # queue.join()
        self.log_info('All workers have finished the work.', filename=self.logfile)
        logging.shutdown()

class Worker:
    """ Worker that creates subprocess to execute job.
    Worker get queue of jobs, pop one job and execute it in subprocess. That subprocess
    call init, run_job and post class methods.
    """
    def __init__(self, worker_name=None, logfile=None, errorfile=None, config=None, *args, **kwargs):
        """
        Parameters
        ----------
        worker_name : str or int

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        config : dict or str
            additional config for pipelines in worker
        args, kwargs
            will be used in init, post and run_job
        """
        self.job = None
        self.worker_config = config or dict()
        self.args = args
        self.kwargs = kwargs
        if isinstance(worker_name, int):
            self.name = "Worker " + str(worker_name)
        elif worker_name is None:
            self.name = 'Worker'
        else:
            self.name = worker_name
        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

    def set_args_kwargs(self, args, kwargs):
        """
        Parameters
        ----------
        args, kwargs
            will be used in init, post and run_job
        """
        if 'logfile' in kwargs:
            self.logfile = kwargs['logfile']
        if 'errorfile' in kwargs:
            self.errorfile = kwargs['errorfile']
        self.logfile = self.logfile or 'research.log'
        self.errorfile = self.errorfile or 'errors.log'

        self.args = args
        self.kwargs = kwargs

    def init(self):
        """ Run before run_job. """
        pass

    def post(self):
        """ Run after run_job. """
        pass

    def run_job(self):
        """ Main part of the worker. """
        pass


    def __call__(self, queue, results):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of jobs for worker
        results : multiprocessing.Queue
            queue for feedback
        """
        self.log_info('Start {} [id:{}]'.format(self.name, os.getpid()), filename=self.logfile)

        try:
            job = queue.get()
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)
        else:
            while job is not None:
                sub_queue = mp.JoinableQueue()
                sub_queue.put(job)
                try:
                    self.log_info(self.name + ' is creating process for job ' + str(job[0]), filename=self.logfile)
                    worker = mp.Process(target=self._run_job, args=(sub_queue, ))
                    worker.start()
                    sub_queue.join()
                except Exception as exception:
                    self.log_error(exception, filename=self.errorfile)
                queue.task_done()
                results.put('done')
                job = queue.get()
        queue.task_done()
        results.put('done')

    def _run_job(self, queue):
        try:
            self.job = queue.get()
            self.log_info(
                'Job {} was started in subprocess [id:{}] by {}'.format(self.job[0], os.getpid(), self.name),
                filename=self.logfile
            )
            self.init()
            self.run_job()
            self.post()
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)
        self.log_info('Job {} was finished by {}'.format(self.job[0], self.name), filename=self.logfile)
        queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        """ Write message into log. """
        pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        """ Write error message into log. """
        pass
