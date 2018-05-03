#pylint:disable=no-value-for-parameter
#pylint:disable=attribute-defined-outside-init
#pylint:disable=broad-except
#pylint:disable=too-many-nested-blocks

""" Workers for research. """

import os

from .. import Config, Pipeline, inbatch_parallel
from .distributor import Worker

class PipelineWorker(Worker):
    """ Worker that run pipelines. """

    def init(self):
        """ Run before task execution. """
        i, job = self.job
        description = job.get_description()
        self.log_info('Task {} has the following configs:\n{}'.format(i, description), filename=self.logfile)
        job.init(self.worker_config)

    def post(self):
        """ Run after task execution. """
        i, _ = self.job
        self.log_info('Task {}: saving final results...'.format(i), filename=self.logfile)
        self.dump_all()

    def dump_all(self):
        """ Dump final results. """
        _, job = self.job
        for name, _ in job.config['pipelines'].items():
            for item, config, repetition in zip(
                    self.experiments,
                    job.config['configs'],
                    job.config['repetition']
                ):
                path = os.path.join(
                    job.config['name'],
                    'results',
                    config.alias(as_string=True),
                    str(repetition),
                    name + '_final'
                )
                item.dump_result(name, path)

    def run_job(self):
        """ Task execution. """
        i, job = self.job

        for j in range(job.config['n_iters']):
            try:
                for name, pipeline in job.config['pipelines'].items():
                    if j in pipeline['execute_for']:
                        if pipeline['preproc'] is not None:
                            job.parallel_execute_for(name)
                        else:
                            for experiment, config, repetition in zip(
                                    job.experiments,
                                    job.config['configs'],
                                    job.config['repetition']
                            ):
                                if pipeline['run']:
                                    self.log_info(
                                        'Task {}, iteration {}: run pipeline {}'
                                        .format(i, j, name), filename=self.logfile
                                    )
                                    experiment.run(name)
                                    experiment.put_result(j, name)
                                    experiment.post_run(name)
                                else:
                                    experiment.next_batch(name)
                                    experiment.put_result(j, name)

                    if j in pipeline['dump_for']:
                        self.log_info('Task {}, iteration {}: dump results for {}...'
                                      .format(i, j, name), filename=self.logfile)
                        for item, config, repetition in zip(
                                job.experiments,
                                task['configs'],
                                task['repetition']
                            ):
                            path = os.path.join(
                                job['name'],
                                'results',
                                config.alias(as_string=True),
                                str(repetition),
                                name + '_dump'
                            )
                            item.dump_result(name, path)
            except StopIteration:
                self.log_info('Task {} was stopped after {} iterations'.format(i, j+1), filename=self.logfile)
                break
