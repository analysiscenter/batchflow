#pylint:disable=no-value-for-parameter
#pylint:disable=attribute-defined-outside-init
#pylint:disable=broad-except
#pylint:disable=too-many-nested-blocks

""" Workers for research. """

import os

from .distributor import Worker

class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before job execution. """
        i, job = self.job
        job.init(self.worker_config)

        description = job.get_description()
        self.log_info('Job {} has the following configs:\n{}'.format(i, description), filename=self.logfile)

    def post(self):
        """ Run after job execution. """
        i, _ = self.job
        self.log_info('Job {}: saving final results...'.format(i), filename=self.logfile)

    def run_job(self):
        """ Job execution. """
        i, job = self.job

        for j in range(job.n_iters):
            try:
                for unit_name, base_unit in job.executable_units.items():
                    if j in base_unit.exec_for:
                        if base_unit.to_run:
                            self.log_info(
                                'Job {} [{}], iteration {}: run pipeline {}'
                                .format(i, os.getpid(), j+1, unit_name), filename=self.logfile
                            )
                        if base_unit.root_pipeline is not None:
                            job.parallel_execute_for(j, unit_name, run=base_unit.to_run)
                        elif base_unit.on_root:
                            self.log_info(
                                        'Job {} [{}], iteration {}: execute function {} on root'
                                        .format(i, os.getpid(), j+1, unit_name), filename=self.logfile
                                    )
                            base_unit(j, job.experiments, *base_unit.args, **base_unit.kwargs)
                        else:
                            for experiment in job.experiments:
                                if base_unit.function is not None:
                                    self.log_info(
                                        'Job {} [{}], iteration {}: execute function {}'
                                        .format(i, os.getpid(), j+1, unit_name), filename=self.logfile
                                    )
                                experiment[unit_name](j, experiment, *experiment[unit_name].args, **experiment[unit_name].kwargs)

                    if j in base_unit.dump_for:
                        self.log_info('Job {} [{}], iteration {}: dump results for {}...'
                                      .format(i, os.getpid(), j+1, unit_name), filename=self.logfile)
                        for experiment in job.experiments:
                            experiment[unit_name].dump_result(unit_name)

            except StopIteration:
                self.log_info('Job {} [{}] was stopped after {} iterations'.format(i, os.getpid(), j+1),
                              filename=self.logfile)
                break
