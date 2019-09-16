""" Workers for research. """

import os
from queue import Empty as EmptyException
import multiprocess as mp
import psutil

from .distributor import Signal

class Worker:
    """ Worker that creates subprocess to execute job.
    Worker get queue of jobs, pop one job and execute it in subprocess. That subprocess
    call init, main and post class methods.
    """
    def __init__(self, devices, worker_name=None, logfile=None, errorfile=None,
                 worker_config=None, timeout=5, trials=2, *args, **kwargs):
        """
        Parameters
        ----------
        devices : list of lists
            devices for each branch in current worker
        worker_name : str or int

        logfile : str (default: 'research.log')

        errorfile : str (default: 'errors.log')

        worker_config : dict or str
            additional config for pipelines in worker
        args, kwargs
            will be used in init, post and main
        """
        self.devices = devices

        if isinstance(worker_name, int):
            self.worker_name = "Worker " + str(worker_name)
        elif worker_name is None:
            self.worker_name = 'Worker'
        else:
            self.worker_name = worker_name

        self.logfile = logfile or 'research.log'
        self.errorfile = errorfile or 'errors.log'
        self.worker_config = worker_config or dict()
        self.timeout = timeout
        self.trials = trials
        self.args = args
        self.kwargs = kwargs

        self.job = None
        self.finished_iterations = None
        self.queue = None
        self.feedback_queue = None
        self.trial = 3
        self.worker = None
        self.device_configs = None

    def set_args_kwargs(self, args, kwargs):
        """
        Parameters
        ----------
        args, kwargs
            will be used in init, post and main
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
        """ Run before main. """
        pass #pylint:disable=unnecessary-pass

    def post(self):
        """ Run after main. """
        pass #pylint:disable=unnecessary-pass

    def main(self):
        """ Main part of the worker. """
        pass #pylint:disable=unnecessary-pass


    def __call__(self, queue, results):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of jobs for worker
        results : multiprocessing.Queue
            queue for feedback
        """
        self.log_info('Start {} [id:{}] (devices: {})'.format(self.worker_name, os.getpid(), self.devices),
                      filename=self.logfile)

        try:
            job = queue.get()
        except Exception as exception: #pylint:disable=broad-except
            self.log_error(exception, filename=self.errorfile)
        else:
            while job is not None:
                try:
                    finished = False
                    self.log_info(self.worker_name + ' is creating process for Job ' + str(job[0]),
                                  filename=self.logfile)
                    for trial in range(self.trials):
                        one_job_queue = mp.JoinableQueue()
                        one_job_queue.put(job)
                        feedback_queue = mp.JoinableQueue()

                        task = mp.Process(target=self._run_task, args=(one_job_queue, feedback_queue, trial))
                        task.start()
                        pid = feedback_queue.get()
                        silence = 0
                        default_signal = Signal(worker=self.worker_name, job=job[0], iteration=0,
                                                n_iters=job[1].n_iters, trial=trial, done=False,
                                                exception=None)

                        while True:
                            try:
                                signal = feedback_queue.get(timeout=1)
                            except EmptyException:
                                signal = None
                                silence += 1
                            if signal is None and silence / 60 > self.timeout:
                                p = psutil.Process(pid)
                                p.terminate()
                                message = 'Job {} [{}] failed in {}'.format(job[0], pid, self.worker_name)
                                self.log_info(message, filename=self.logfile)
                                default_signal.exception = TimeoutError(message)
                                results.put(default_signal)
                                break
                            elif signal is not None and signal.done:
                                finished = True
                                default_signal = signal
                                break
                            elif signal is not None:
                                default_signal = signal
                                results.put(default_signal)
                                silence = 0
                        if finished:
                            break
                except Exception as exception: #pylint:disable=broad-except
                    self.log_error(exception, filename=self.errorfile)
                    default_signal.exception = exception
                    results.put(default_signal)
                if default_signal.done:
                    results.put(default_signal)
                else:
                    default_signal.exception = RuntimeError('Job {} [{}] failed {} times in {}'
                                                            .format(job[0], pid, self.trials, self.worker_name))
                    default_signal.done = True
                    results.put(default_signal)
                queue.task_done()
                job = queue.get()
            queue.task_done()


    def _run_task(self, queue, feedback_queue, trial):
        exception = None
        try:
            self.feedback_queue = feedback_queue
            self.trial = trial

            feedback_queue.put(os.getpid())
            self.job = queue.get()

            self.log_info(
                'Job {} was started in subprocess [id:{}] by {}'.format(self.job[0], os.getpid(), self.worker_name),
                filename=self.logfile
            )
            self.init()
            self.main()
            self.post()
        except Exception as e: #pylint:disable=broad-except
            exception = e
            self.log_error(exception, filename=self.errorfile)
        self.log_info('Job {} [{}] was finished by {}'.format(self.job[0], os.getpid(), self.worker_name),
                      filename=self.logfile)
        signal = Signal(worker=self.worker_name, job=self.job[0], iteration=self.finished_iterations,
                        n_iters=self.job[1].n_iters, trial=self.trial, done=True,
                        exception=[exception]*len(self.job[1].experiments))
        self.feedback_queue.put(signal)
        queue.task_done()

    @classmethod
    def log_info(cls, *args, **kwargs):
        """ Write message into log """
        pass #pylint:disable=unnecessary-pass

    @classmethod
    def log_error(cls, *args, **kwargs):
        """ Write error message into log """
        pass #pylint:disable=unnecessary-pass


class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before job execution. """
        i, job = self.job
        n_branches = len(job.configs)
        self.device_configs = [dict(device=str(self.devices[i])) for i in range(n_branches)]

        job.init(self.worker_config, self.device_configs)
        description = job.get_description()
        self.log_info('Job {} has the following configs:\n{}'.format(i, description), filename=self.logfile)

    def post(self):
        """ Run after job execution. """
        pass #pylint:disable=unnecessary-pass

    def _execute_on_root(self, base_unit, iteration):
        _, job = self.job
        return base_unit.action_iteration(iteration, job.n_iters) or ('last' in base_unit.execute) and job.all_stopped()

    def main(self):
        """ Job execution. """
        idx_job, job = self.job

        iteration = 0
        self.finished_iterations = iteration
        while (job.n_iters is None or iteration < job.n_iters) and job.alive_experiments() > 0:
            job.clear_stopped_list() # list with flags for each experiment
            for unit_name, base_unit in job.executable_units.items():
                exec_actions = job.get_actions(iteration, unit_name) # for each experiment is None if experiment mustn't
                                                                     # be exuted for that iteration and dict else
                # execute units
                messages = []
                exceptions = [None] * len(job.experiments)
                if base_unit.root_pipeline is not None:
                    if sum([item is not None for item in exec_actions]) > 0:
                        for i, action in enumerate(exec_actions):
                            if action is not None:
                                messages.append("J {} [{}] I {}: execute '{}' [{}]"
                                                .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        exceptions = job.parallel_execute_for(iteration, unit_name, exec_actions)
                elif base_unit.on_root and self._execute_on_root(base_unit, iteration):
                    try:
                        for i, action in enumerate(exec_actions):
                            if action is not None:
                                messages.append("J {} [{}] I {}: on root '{}' [{}]"
                                                .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        base_unit(iteration, job.experiments, *base_unit.args, **base_unit.kwargs)
                    except Exception as e: #pylint:disable=broad-except
                        exceptions = [e] * len(job.experiments)
                else:
                    for i, action in enumerate(exec_actions):
                        if action is not None:
                            messages.append("J {} [{}] I {}: execute '{}' [{}]"
                                            .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                    exceptions = job.parallel_call(iteration, unit_name, exec_actions)

                # select units that raise exceptions on that iteration
                for i, exception in enumerate(exceptions):
                    if exception is not None:
                        message = ("J {} [{}] I {}: '{}' [{}]: exception {}"
                                   .format(idx_job, os.getpid(), iteration+1, unit_name, i, repr(exception)))
                        self.log_info(message, filename=self.logfile)
                        job.stopped[i] = True

                # dump results
                dump_actions = job.get_actions(iteration, unit_name, action='dump')
                for i, experiment in enumerate(job.experiments):
                    if dump_actions[i] is not None:
                        messages.append("J {} [{}] I {}: dump '{}' [{}]"
                                        .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        experiment[unit_name].dump_result(iteration+1, unit_name)

                if base_unit.logging:
                    for message in messages:
                        self.log_info(message, filename=self.logfile)
                job.update_exceptions(exceptions)
                signal = Signal(worker=self.worker, job=idx_job, iteration=iteration, n_iters=job.n_iters,
                                trial=self.trial, done=False, exception=job.exceptions, exec_actions=exec_actions,
                                dump_actions=dump_actions)
                self.feedback_queue.put(signal)
            iteration += 1
            self.finished_iterations = iteration
