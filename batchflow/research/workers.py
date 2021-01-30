""" Workers for research. """

import os
from copy import copy
import time
from queue import Empty as EmptyException
import multiprocess as mp
import psutil

from .distributor import Signal
from .executable import PipelineStopIteration

class Worker:
    """ Worker that creates subprocess to execute job.
    Worker get queue of jobs, pop one job and execute it in subprocess. That subprocess
    call init, main and post class methods.
    """
    def __init__(self, devices, worker_name=None, worker_config=None, timeout=None, trials=2, logger=None):
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

        self.worker_config = worker_config or dict()
        self.timeout = timeout
        self.trials = trials
        self.logger = logger

        self.job = None
        self.finished_iterations = None
        self.queue = None
        self.feedback_queue = None
        self.trial = 3
        self.worker = None
        self.device_configs = None

        self.last_update_time = None

    def init(self):
        """ Run before main. """

    def post(self):
        """ Run after main. """

    def main(self):
        """ Main part of the worker. """


    def __call__(self, queue, results):
        """ Run worker.

        Parameters
        ----------
        queue : multiprocessing.Queue
            queue of jobs for worker
        results : multiprocessing.Queue
            queue for feedback
        """
        _devices = [item['device'] for item in self.devices]
        self.logger.info('Start {} [id:{}] (devices: {})'.format(self.worker_name, os.getpid(), _devices))

        try:
            job = queue.get()
        except Exception as exception: #pylint:disable=broad-except
            self.logger.error(exception)
        else:
            while job is not None:
                try:
                    finished = False
                    self.logger.info(self.worker_name + ' is creating process for Job ' + str(job[0]))
                    for trial in range(self.trials):
                        one_job_queue = mp.JoinableQueue()
                        one_job_queue.put(job)
                        feedback_queue = mp.JoinableQueue()
                        last_update_time = mp.Value('d', time.time())

                        task = mp.Process(target=self._run_task, args=(one_job_queue, feedback_queue,
                                                                       trial, last_update_time))
                        task.start()
                        pid = feedback_queue.get()
                        final_signal = Signal(worker=self.worker_name, job=job[0], iteration=0,
                                              n_iters=job[1].n_iters, trial=trial, done=False,
                                              exception=None)

                        while True:
                            try:
                                signal = feedback_queue.get(timeout=1)
                            except EmptyException:
                                signal = None
                            if signal is None:
                                execution_time = (time.time() - last_update_time.value) / 60
                                if self.timeout is not None and execution_time > self.timeout:
                                    p = psutil.Process(pid)
                                    p.terminate()
                                    message = f'Job {job[0]} [{pid}] failed in {self.worker_name} because of timeout'
                                    self.logger.info(message)
                                    final_signal.exception = TimeoutError(message)
                                    results.put(copy(final_signal))
                                    break
                            if signal is not None and signal.done:
                                finished = True
                                final_signal = signal
                                break
                            if signal is not None:
                                final_signal = signal
                                results.put(copy(final_signal))
                        if finished:
                            break
                except Exception as exception: #pylint:disable=broad-except
                    self.logger.error(exception)
                    final_signal.exception = exception
                    results.put(copy(final_signal))
                if final_signal.done:
                    results.put(copy(final_signal))
                else:
                    final_signal.exception = RuntimeError('Job {} [{}] failed {} times in {}'
                                                          .format(job[0], pid, self.trials, self.worker_name))
                    final_signal.done = True
                    results.put(copy(final_signal))
                queue.task_done()
                job = queue.get()
            queue.task_done()


    def _run_task(self, queue, feedback_queue, trial, last_update_time):
        exception = None
        try:
            self.feedback_queue = feedback_queue
            self.trial = trial
            self.last_update_time = last_update_time

            feedback_queue.put(os.getpid())
            self.job = queue.get()

            self.logger.info(
                'Job {} was started in subprocess [id:{}] by {}'.format(self.job[0], os.getpid(), self.worker_name)
            )
            self.init()
            self.main()
            self.post()
        except Exception as e: #pylint:disable=broad-except
            exception = e
            self.logger.error(exception)
        self.logger.info('Job {} [{}] was finished by {}'.format(self.job[0], os.getpid(), self.worker_name))
        signal = Signal(worker=self.worker_name, job=self.job[0], iteration=self.finished_iterations,
                        n_iters=self.job[1].n_iters, trial=self.trial, done=True,
                        exception=[exception]*len(self.job[1].experiments))
        self.feedback_queue.put(signal)
        queue.task_done()


class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before job execution. """
        i, job = self.job
        n_branches = len(job.configs)
        all_devices = set(device for i in range(n_branches)
                          for device in self.devices[i]['device'] if device is not None)

        if len(all_devices) > 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(all_devices)

        mapping = {
            item: 'gpu:' + str(i) for i, item in enumerate(all_devices)
        }

        self.device_configs = [
            {'device': [mapping[device] for device in self.devices[i]['device'] if device is not None]}
            for i in range(n_branches)
        ]
        job.init(self.worker_config, self.device_configs, self.last_update_time)
        description = job.get_description()
        self.logger.info('Job {} has the following configs:\n{}'.format(i, description))

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
                                                                     # be executed for that iteration and dict else
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
                    if sum([item is not None for item in exec_actions]) > 0:
                        for i, action in enumerate(exec_actions):
                            if action is not None:
                                messages.append("J {} [{}] I {}: on root '{}' [{}]"
                                                .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        exceptions = job.call_on_root(iteration, unit_name)
                else:
                    for i, action in enumerate(exec_actions):
                        if action is not None:
                            messages.append("J {} [{}] I {}: execute '{}' [{}]"
                                            .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                    exceptions = job.parallel_call(iteration, unit_name, exec_actions)

                # select units that raise exceptions on that iteration
                for i, exception in enumerate(exceptions):
                    if exception is not None:
                        if not isinstance(exception, PipelineStopIteration):
                            message = ("J {} [{}] I {}: '{}' [{}]: exception {}"
                                       .format(idx_job, os.getpid(), iteration+1, unit_name, i, repr(exception)))
                            self.logger.info(message)
                            self.logger.error(exception)
                        else:
                            message = ("J {} [{}] I {}: '{}' [{}]: Pipeline was stopped by StopIteration"
                                       .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                            self.logger.info(message)
                        job.stopped[i] = True

                # dump results
                dump_actions = job.get_actions(iteration, unit_name, action='dump')
                for i, experiment in enumerate(job.experiments):
                    if dump_actions[i] is not None:
                        messages.append("J {} [{}] I {}: dump '{}' [{}]"
                                        .format(idx_job, os.getpid(), iteration+1, unit_name, i))
                        experiment[unit_name].dump_result(job.ids[i], iteration+1, unit_name)

                if base_unit.logging:
                    for message in messages:
                        self.logger.info(message)
                job.update_exceptions(exceptions)
                signal = Signal(worker=self.worker, job=idx_job, iteration=iteration, n_iters=job.n_iters,
                                trial=self.trial, done=False, exception=job.exceptions, exec_actions=exec_actions,
                                dump_actions=dump_actions)
                self.feedback_queue.put(signal)
            iteration += 1
            self.finished_iterations = iteration
