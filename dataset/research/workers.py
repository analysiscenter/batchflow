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
        description = job.get_description()
        self.log_info('Job {} has the following configs:\n{}'.format(i, description), filename=self.logfile)
        job.init(self.worker_config)

    def post(self):
        """ Run after job execution. """
        i, job = self.job
        self.log_info('Job {}: saving final results...'.format(i), filename=self.logfile)
        job.dump_all()

    def run_job(self):
        """ Job execution. """
        i, job = self.job

        for j in range(job.config['n_iters']):
            try:
                for name, pipeline in job.config['pipelines'].items():
                    if j in pipeline['execute_for']:
                        if pipeline['root'] is not None:
                            job.parallel_execute_for(name)
                        else:
                            for experiment in job.experiments:
                                if pipeline['run']:
                                    self.log_info(
                                        'Job {} [{}], iteration {}: run pipeline {}'
                                        .format(i, os.getpid(), j+1, name), filename=self.logfile
                                    )
                                    experiment.run(name)
                                else:
                                    experiment.next_batch(name)
                        job.put_pipeline_result(j, name)

                    if j in pipeline['dump_for']:
                        self.log_info('Job {} [{}], iteration {}: dump results for {}...'
                                      .format(i, os.getpid(), j+1, name), filename=self.logfile)
                        for item in job.experiments:
                            item.dump_pipeline_result(name, name)

                for name, function in job.config['functions'].items():
                    if j in function['execute_for']:
                        self.log_info('Job {} [{}], iteration {}: call function {}...'
                                      .format(i, os.getpid(), j+1, name), filename=self.logfile)
                        for item in job.experiments:
                            item.call_function(j, name)
                    if j in function['dump_for']:
                        self.log_info('Job {} [{}], iteration {}: dump results for function {}...'
                                      .format(i, os.getpid(), j+1, name), filename=self.logfile)
                        for item in job.experiments:
                            item.dump_function_result(name, name)
            except StopIteration:
                self.log_info('Job {} [{}] was stopped after {} iterations'.format(i, os.getpid(), j+1),
                              filename=self.logfile)
                break
