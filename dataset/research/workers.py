#pylint:disable=no-value-for-parameter
#pylint:disable=attribute-defined-outside-init
#pylint:disable=broad-except
#pylint:disable=too-many-nested-blocks

""" Workers for research. """

import os

from .. import Config, Pipeline, inbatch_parallel
from .distributor import Worker
from .singlerun import SingleRunning

class PipelineWorker(Worker):
    """ Worker that run pipelines. """
    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, item, single_runnings, batch, name):
        _ = single_runnings
        try:
            item.run_on_batch(batch, name)
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)

    def _parallel_init(self, single_runnings, batch, name):
        _ = batch, name
        return single_runnings

    def init(self):
        """ Run before task execution. """
        i, task = self.task
        if isinstance(task['n_branches'], list):
            description = '\n'.join([str({**config.alias(), **_config})
                                     for config, _config in zip(task['configs'], task['n_branches'])])
        else:
            description = '\n'.join([str(config.alias()) for config in task['configs']])

        self.log_info('Task {} has the following configs:\n{}'.format(i, description), filename=self.logfile)

        self.single_runnings = []
        for idx, config in enumerate(task['configs']):
            single_running = SingleRunning()
            for name, pipeline in task['pipelines'].items():
                pipeline_copy = pipeline['ppl'] + Pipeline()

                pipeline['execute_for'] = SingleRunning.get_iterations(pipeline['execute_for'], task['n_iters'])

                single_running.add_pipeline(pipeline_copy, pipeline['var'],
                                            name=name, execute_for=pipeline['execute_for'], **pipeline['kwargs'])
            if isinstance(task['n_branches'], list):
                n_branches = task['n_branches'][idx]
            else:
                n_branches = Config()

            worker_config = self.kwargs.get('config', Config())
            self.log_info(worker_config, filename=self.logfile)

            single_running.add_common_config(config.config()+n_branches+worker_config)
            single_running.init()
            self.single_runnings.append(single_running)

    def post(self):
        """ Run after task execution. """
        _, task = self.task
        self.log_info('Saving results...', filename=self.logfile)
        for item, config, repetition in zip(self.single_runnings, task['configs'], task['repetition']):
            item.save_results(os.path.join(task['name'], 'results',
                                           config.alias(as_string=True), str(repetition), 'final'))

    def run_task(self):
        """ Task execution. """
        i, task = self.task

        for j in range(task['n_iters']):
            try:
                for name, pipeline in task['pipelines'].items():
                    if j in pipeline['execute_for']:
                        if pipeline['preproc'] is not None:
                            batch = pipeline['preproc'].next_batch()
                            self._parallel_run(self.single_runnings, batch, name)
                        else:
                            for item, config, repetition in zip(
                                    self.single_runnings,
                                    task['configs'],
                                    task['repetition']
                            ):
                                if pipeline['run']:
                                    self.log_info('Run pipeline {}'.format(name), filename=self.logfile)
                                    item.run(name)
                                    filename = os.path.join(
                                        task['name'],
                                        'results',
                                        config.alias(as_string=True),
                                        str(repetition),
                                        name + '_' + str(j)
                                    )
                                    item.save_results(filename, names=name)
                                else:
                                    item.next_batch(name)
            except StopIteration:
                self.log_info('Task {} was stopped after {} iterations'.format(i, j+1), filename=self.logfile)
                break

class SavingWorker(PipelineWorker):
    """ Worker that run pipelines and save first model. """
    def post(self):
        """ Run after task execution. """
        super().post()
        _, task = self.task
        for item, config, repetition in zip(self.single_runnings, task['configs'], task['repetition']):
            if repetition == 0:
                filename = os.path.join(task['name'],
                                        'results',
                                        config.alias(as_string=True),
                                        str(task['repetition']) + '_model')
                item.get_pipeline('train')['ppl'].get_model_by_name('model').save(filename)
