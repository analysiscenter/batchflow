#pylint:disable=too-few-public-methods
#pylint:disable=broad-except
#pylint:disable=too-many-function-args
#pylint:disable=attribute-defined-outside-init
#pylint:disable=import-error

""" Classes for multiprocess task running. """

import os
import logging
import multiprocess as mp
from tqdm import tqdm

class Tasks:
    """ Tasks to workers. """
    def __init__(self, tasks):
        self.tasks = tasks

    def __iter__(self):
        return self.tasks

class Worker:
    """ Worker that creates subprocess to execute task.
    Worker get queue of tasks, pop one task and execute it in subprocess. That subprocess
    call init, run_task and post class methods.
    """
    def __init__(self, worker_name=None, logfile=None, errorfile=None, *args, **kwargs):
        """
        Parameters
        ----------
        worker_name : str or int

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        args, kwargs
            will be used in init, post and task
        """
        self.task = None
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
            will be used in init, post and task
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
        """ Run before task. """
        pass

    def post(self):
        """ Run after task. """
        pass

    def run_task(self):
        """ Main part of the worker. """
        pass


    def __call__(self, queue, results):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of tasks for worker
        """
        self.log_info('Start {} [id:{}]'.format(self.name, os.getpid()), filename=self.logfile)

        try:
            item = queue.get()
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)
        else:
            while item is not None:
                sub_queue = mp.JoinableQueue()
                sub_queue.put(item)
                try:
                    self.log_info(self.name + ' is creating process', filename=self.logfile)
                    worker = mp.Process(target=self._run, args=(sub_queue, ))
                    worker.start()
                    sub_queue.join()
                except Exception as exception:
                    self.log_error(exception, filename=self.errorfile)
                queue.task_done()
                results.put('done')
                item = queue.get()
        queue.task_done()
        results.put('done')

    def _run(self, queue):
        try:
            self.task = queue.get()
            self.log_info(
                'Task {} was started in subprocess [id:{}] by {}'.format(self.task[0], os.getpid(), self.name),
                filename=self.logfile
            )
            self.init()
            self.run_task()
            self.post()
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)
        self.log_info('Task {} was finished by {}'.format(self.task[0], self.name), filename=self.logfile)
        queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        """ Write message into log. """
        pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        """ Write error message into log. """
        pass

class Distributor:
    """ Distributor of tasks between workers. """
    def __init__(self, n_workers, worker_class=None):
        """
        Parameters
        ----------
        n_workers : int or list of Worker instances

        worker_class : Worker subclass or None
        """
        # if isinstance(n_workers, int) and worker_class is None:
        #     raise ValueError('If worker_class is None, n_workers must be list of Worker instances.')
        self.n_workers = n_workers
        self.worker_class = worker_class

    def _tasks_to_queue(self, tasks):
        queue = mp.JoinableQueue()
        for idx, task in enumerate(tasks):
            queue.put((idx, task))
        for _ in range(self.n_workers):
            queue.put(None)
        return queue

    def _put_tasks(self, queue, tasks, size):
        created_chunks = 0
        for idx, task in enumerate(tasks):
            if idx < (created_chunks + 1) * size:
                queue.put((idx, task))
            else:
                created_chunks += 1
                yield queue

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

    def run(self, tasks, dirname, n_tasks, logfile=None, errorfile=None, *args, **kwargs):
        """ Run disributor and workers.

        Parameters
        ----------
        tasks : iterable

        dirname : str

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        args, kwargs
            will be used in worker
        """
        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'

        self.logfile = os.path.join(dirname, self.logfile)
        self.errorfile = os.path.join(dirname, self.errorfile)

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
            self.log_info('Create tasks queue', filename=self.logfile)
            queue = self._tasks_to_queue(tasks)
            results = mp.JoinableQueue()
        except Exception as exception:
            logging.error(exception, exc_info=True)
        else:
            self.log_info('Run {} workers'.format(len(workers)), filename=self.logfile)
            for worker in workers:
                worker.log_info = self.log_info
                worker.log_error = self.log_error

                try:
                    mp.Process(target=worker, args=(queue, results)).start()
                except Exception as exception:
                    logging.error(exception, exc_info=True)
            for _ in tqdm(range(n_tasks)):
                results.get()
            # queue.join()
        self.log_info('All workers have finished the work.', filename=self.logfile)
        logging.shutdown()
    