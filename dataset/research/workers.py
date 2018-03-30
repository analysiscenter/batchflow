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
    def _parallel_run(self, item, single_runnings, batch, iteration, name):
        _ = single_runnings
        try:
            item.run_on_batch(batch, name)
            item.put_result(iteration, name)
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)

    def _parallel_init(self, single_runnings, batch, iteration, name):
        _ = batch, name, iteration
        return single_runnings

    def get_iterations(self, execute_for, n_iters=None):
        """ Get indices of iterations from execute_for. """
        if n_iters is not None:
            if isinstance(execute_for, int):
                if execute_for == -1:
                    execute_for = [n_iters - 1]
                else:
                    execute_for = list(range(-1, n_iters, execute_for))
            elif execute_for is None:
                execute_for = list(range(n_iters))
        return execute_for

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

                pipeline['execute_for'] = self.get_iterations(pipeline['execute_for'], task['n_iters'])
                if pipeline['dump_for'] is not None:
                    pipeline['dump_for'] = self.get_iterations(pipeline['dump_for'], task['n_iters'])
                else:
                    pipeline['dump_for'] = []
                single_running.add_pipeline(pipeline_copy, pipeline['var'],
                                            name=name, post_run=pipeline['post_run'], **pipeline['kwargs'])
            if isinstance(task['n_branches'], list):
                n_branches = task['n_branches'][idx]
            else:
                n_branches = Config()

            worker_config = self.kwargs.get('config', Config())

            single_running.add_common_config(config.config()+n_branches+worker_config)
            single_running.init()
            self.single_runnings.append(single_running)

    def post(self):
        """ Run after task execution. """
        self.log_info('Saving final results...', filename=self.logfile)
        self.dump_all()

    def dump_all(self):
        """ Dump final results. """
        _, task = self.task
        for name, _ in task['pipelines'].items():
            for item, config, repetition in zip(
                    self.single_runnings,
                    task['configs'],
                    task['repetition']
                ):
                path = os.path.join(
                    task['name'],
                    'results',
                    config.alias(as_string=True),
                    str(repetition),
                    name + '_final'
                )
                item.dump_result(name, path)

    def run_task(self):
        """ Task execution. """
        i, task = self.task

        for j in range(task['n_iters']):
            try:
                for name, pipeline in task['pipelines'].items():
                    if j in pipeline['execute_for']:
                        if pipeline['preproc'] is not None:
                            batch = pipeline['preproc'].next_batch()
                            self._parallel_run(self.single_runnings, batch, j, name)
                        else:
                            for item, config, repetition in zip(
                                    self.single_runnings,
                                    task['configs'],
                                    task['repetition']
                            ):
                                if pipeline['run']:
                                    self.log_info('Run pipeline {}'.format(name), filename=self.logfile)
                                    item.run(name)
                                    item.put_result(j, name)
                                    item.post_run(name)
                                else:
                                    item.next_batch(name)
                                    item.put_result(j, name)

                    if j in pipeline['dump_for']:
                        self.log_info('Iteration {}: dump results for {}...'.format(j, name), filename=self.logfile)
                        for item, config, repetition in zip(
                            self.single_runnings,
                            task['configs'],
                            task['repetition']
                            ):
                            path = os.path.join(
                                task['name'],
                                'results',
                                config.alias(as_string=True),
                                str(repetition),
                                name + '_dump'
                            )
                            item.dump_result(name, path)
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
