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

    def alive_experiments(self):
        return len([item for item in job.exceptions if item is None])

    def run_job(self):
        """ Job execution. """
        idx_job, job = self.job

        iteration = 0

        while (job.n_iters is None or iteration < job.n_iters) and self.alive_experiments() > 0:
            self.finished_iterations = iteration
            job.stoped = [False for i in len(job.experiments)]
            for unit_name, base_unit in job.executable_units.items():
                if base_unit.root_pipeline is not None:
                    exceptions = job.parallel_execute_for(iteration, unit_name, run=base_unit.to_run)
                elif base_unit.on_root and base_unit.action_iteration(iteration, n_iters) or base_unit.dump_for == -1 and job.all_stoped():
                    exceptions = [None] * len(job.executable_units)
                    try:
                        base_unit(j, job.experiments, *base_unit.args, **base_unit.kwargs)
                    except Exception as e:
                        exceptions = [e] * len(job.experiments)
                else:
                    exceptions = job.parallel_call(j, unit_name)
                if base_unit.action_iteration(iteration, action='dump') or base_unit.dump_for == -1:
                    for experiment, exception in zip(job.experiments, job.exceptions):
                        if (exception is not None and base_unit.dump_for == -1) or (exception is None):
                            experiment[unit_name].dump_result(j+1, unit_name)                            
                for i, exception in enumerate(job.exceptions):
                    job.stoped[i] = job.stoped[i] or isinstance(exception, StopIteration)
            signal = Signal(self.worker, i, j, job.n_iters, self.trial, False, exceptions)
            self.feedback_queue.put(signal)
            j += 1
        j -= 1
        for unit_name, base_unit in job.executable_units.items():
            self.log_info('J {} [{}], final dump {}'
                    .format(i, os.getpid(), unit_name), filename=self.logfile)
            if not self._to_exec(j, base_unit.dump_for):
                for experiment in job.experiments:
                        experiment[unit_name].dump_result(j+1, unit_name)