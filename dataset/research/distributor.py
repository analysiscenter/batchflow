""" Classes for multiprocess job running. """

#pylint:disable=broad-except
#pylint:disable=attribute-defined-outside-init

import os
import logging
import multiprocess as mp
from queue import Empty
from tqdm import tqdm
import psutil

from .job import Job

TRAILS = 3

class Distributor:
    """ Distributor of jobs between workers. """
    def __init__(self, workers, gpu, worker_class=None, timeout=5):
        """
        Parameters
        ----------
        workers : int or list of Worker instances

        worker_class : Worker subclass or None
        """
        self.workers = workers
        self.worker_class = worker_class
        self.gpu = gpu
        self.timeout = timeout

    def _jobs_to_queue(self, jobs):
        queue = mp.JoinableQueue()
        for idx, job in enumerate(jobs):
            queue.put((idx, job))
        for _ in range(self.workers):
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

    def _get_worker_gpu(self, n_workers, index):
        if len(self.gpu) == 1:
            return [0]
        else:
            length = len(self.gpu) // n_workers
            start = index * length
            end = start + length
        return self.gpu[start:end]

    def run(self, jobs, dirname, n_jobs, n_iters, logfile=None, errorfile=None, progress_bar=False, *args, **kwargs):
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

        if isinstance(self.workers, int):
            workers = [self.worker_class(
                gpu=self._get_worker_gpu(self.workers, i),
                worker_name=i,
                timeout=self.timeout,
                *args, **kwargs
                ) 
                for i in range(self.workers)]
        else:
            workers = [
                self.worker_class(gpu=self._get_worker_gpu(len(self.workers), i), worker_name=i, 
                                  config=config, timeout=self.timeout, *args, **kwargs)
                for i, config in enumerate(self.workers)
            ]
        try:
            self.log_info('Create queue of jobs', filename=self.logfile)
            self.queue = self._jobs_to_queue(jobs)
            self.results = mp.JoinableQueue()
        except Exception as exception:
            logging.error(exception, exc_info=True)
        else:
            if len(workers) > 1:
                msg = 'Run {} workers'
            else:
                msg = 'Run {} worker'
            self.log_info(msg.format(len(workers)), filename=self.logfile)
            for worker in workers:
                worker.log_info = self.log_info
                worker.log_error = self.log_error
                try:
                    mp.Process(target=worker, args=(self.queue, self.results)).start()
                except Exception as exception:
                    logging.error(exception, exc_info=True)
            self.answers = [0 for _ in range(n_jobs)]
            with tqdm(total=n_jobs*n_iters) as progress:
                while True:
                    update = self._get_answer(n_jobs, n_iters)
                    progress.update(update)
                    if sum(self.answers) == n_jobs * n_iters:
                        break
        self.log_info('All workers have finished the work.', filename=self.logfile)
        logging.shutdown()

    def _get_answer(self, n_jobs, n_iters):
        _, job, state = self.results.get()
        if isinstance(state, int):
            state += 1
            update = state - self.answers[job]
            self.answers[job] = state
        else:
            update = n_iters - self.answers[job]
            self.answers[job] = n_iters
        return update

class Worker:
    """ Worker that creates subprocess to execute job.
    Worker get queue of jobs, pop one job and execute it in subprocess. That subprocess
    call init, run_job and post class methods.
    """
    def __init__(self, gpu, worker_name=None, logfile=None, errorfile=None, config=None, timeout=5, *args, **kwargs):
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
        self.gpu = gpu
        self.timeout = timeout

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
        self.log_info('Start {} [id:{}] (gpu: {})'.format(self.name, os.getpid(), self.gpu), filename=self.logfile)

        if len(self.gpu) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in self.gpu])

        try:
            job = queue.get()
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)
        else:
            while job is not None:
                try:
                    finished = False
                    self.log_info(self.name + ' is creating process for Job ' + str(job[0]), filename=self.logfile)
                    for trail in range(TRAILS):
                        sub_queue = mp.JoinableQueue()
                        sub_queue.put(job)
                        feedback_queue = mp.JoinableQueue()

                        worker = mp.Process(target=self._run_job, args=(sub_queue, feedback_queue))
                        worker.start()
                        #sub_queue.join()
                        pid = feedback_queue.get()
                        silence = 0
                        while True:
                            try:
                                answer = feedback_queue.get(timeout=1)
                            except Empty:
                                answer = None
                                silence += 1
                            if answer == 'done':
                                finished = True
                                break
                            elif answer is None:
                                if silence / 60 > self.timeout:
                                    break
                            else:
                                results.put((self.name, job[0], answer))
                                silence = 0
                        if finished:
                            break
                        p = psutil.Process(pid)
                        p.terminate()
                        self.log_info('Job {} [{}] failed in {}'.format(job[0], pid, self.name) , filename=self.logfile)
                except Exception as exception:
                    self.log_error(exception, filename=self.errorfile)
                queue.task_done()
                results.put((self.name, job[0], 'done'))
                job = queue.get()
        queue.task_done()

    def _run_job(self, queue, feedback_queue):
        try:
            self.feedback_queue = feedback_queue
            feedback_queue.put(os.getpid())
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
        self.log_info('Job {} [{}] was finished by {}'.format(self.job[0], os.getpid(), self.name),
                      filename=self.logfile)
        feedback_queue.put('done')
        # queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        """ Write message into log. """
        pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        """ Write error message into log. """
        pass
