#pylint:disable=no-value-for-parameter
#pylint:disable=attribute-defined-outside-init
#pylint:disable=broad-except
#pylint:disable=too-many-nested-blocks

""" Workers for research. """

import os

from .distributor import Worker, Signal

class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before job execution. """
        i, job = self.job
        n_branches = len(job.configs)

        if len(self.gpu) <= 1:
            self.gpu_configs = [dict(device='/device:GPU:0') for i in range(n_branches)]
        else:
            self.gpu_configs = [dict(device='/device:GPU:{}'.format(i)) for i in range(n_branches)]

        job.init(self.worker_config, self.gpu_configs)

        description = job.get_description()
        self.log_info('Job {} has the following configs:\n{}'.format(i, description), filename=self.logfile)

    def post(self):
        """ Run after job execution. """
        pass

    def _exception_processing(self, j, i, job, exceptions):
        n_executed = len([item for item in job.exceptions if item is None])
        # for idx, exception in enumerate(exceptions):
        #     if exception is not None:
        #         self.log_info(
        #             'J {} [{}], I {}, Config {}: {}'
        #             .format(i, os.getpid(), j+1, idx, repr(exception)), filename=self.logfile
        #         )
        return n_executed == 0

    def run_job(self):
        """ Job execution. """
        idx_job, job = self.job

        iteration = 0
        self.finished_iterations = iteration

        while (job.n_iters is None or iteration < job.n_iters) and job.alive_experiments() > 0:
            job.stopped = [False for _ in range(len(job.experiments))]
            for unit_name, base_unit in job.executable_units.items():
                if base_unit.root_pipeline is not None:
                    exceptions = job.parallel_execute_for(iteration, unit_name, run=base_unit.to_run)
                elif base_unit.on_root and (base_unit.action_iteration(iteration, n_iters) or base_unit.exec_for == -1 and job.all_stopped()):
                    exceptions = [None] * len(job.executable_units)
                    try:
                        base_unit(iteration, job.experiments, *base_unit.args, **base_unit.kwargs)
                    except Exception as e:
                        exceptions = [e] * len(job.experiments)
                        raise e
                else:
                    exceptions = job.parallel_call(iteration, unit_name)
                for i, exception in enumerate(exceptions):
                    if isinstance(exception, StopIteration):
                        job.stopped[i] = True
                if base_unit.action_iteration(iteration, job.n_iters, action='dump'):
                    for i, experiment in enumerate(job.experiments):
                        exception = exceptions[i]
                        if exception is None:
                            experiment[unit_name].dump_result(iteration+1, unit_name)
                            self.log_info('J {} [{}] I {}: dump {} [{}]'
                                 .format(idx_job, os.getpid(), iteration+1, unit_name, i), filename=self.logfile)
                if base_unit.dump_for == -1:
                    for i, experiment in enumerate(job.experiments):
                        if job.stopped[i]:
                            experiment[unit_name].dump_result(iteration+1, unit_name)
                            self.log_info('J {} [{}] I {}: dump {} [{}]'
                                 .format(idx_job, os.getpid(), iteration+1, unit_name, i), filename=self.logfile)
                job.update_exceptions(exceptions)             
            signal = Signal(self.worker, idx_job, iteration, job.n_iters, self.trial, False, job.exceptions)
            self.feedback_queue.put(signal)
            iteration += 1
            self.finished_iterations = iteration
