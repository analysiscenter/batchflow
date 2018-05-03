""" Class for one task. """

from collections import OrderedDict

from .. import Config, inbatch_parallel

class Job:
    def __init__(self, config):
        self.experiments = []
        self.config = config

    def init(self, worker_config):
        for idx, config in enumerate(self.config['configs']):

            experiment = Experiment({**self.config, 'worker_config': worker_config})

            for name, pipeline in self.config['pipelines'].items():
                experiment.add_pipeline(pipeline, config, name)

            self.experiments.append(experiment)

    def parallel_execute_for(self, name):
        batch = self.config['pipelines'][name]['preproc'].next_batch()
        self._parallel_run(batch, name)


    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, item, batch, name):
        try:
            item.execute_for(batch, name)
        except Exception as exception:
            self.log_error(exception, filename=self.errorfile)

    def _parallel_init(self, batch, name):
        _ = batch, name
        return self.experiments

    def get_description(self):
        if isinstance(self.config['n_branches'], list):
            description = '\n'.join([str({**config.alias(), **_config})
                                     for config, _config in zip(self.config['configs'], self.config['n_branches'])])
        else:
            description = '\n'.join([str(config.alias()) for config in self.config['configs']])
        return description

class Experiment:
    def __init__(self, config):
        self.config = config
        self.pipelines = OrderedDict()
        self.results = dict()

    def add_pipeline(self, pipeline, config, name):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None pipeline must have run action with lazy=True.
        variables : str or list of str or None
            names of pipeline variables to remember at each repetition.
        name : str (default None)
            name of pipeline. If name is None pipeline will have name 'ppl_{index}'
        execute_for : int, list or None
            If -1, pipeline will be executed just at last iteration.
            If other int, pipeline will be excuted for iterations with that step
            If list, pipeline will be excuted for that iterations
            If None, pipeline will executed on each iteration.
        kwargs :
            parameters in pipeline config that depends on the name of the other config. For example,
            if test pipeline imports model from the other pipeline with name 'train' in SingleRunning,
            corresponding parameter in import_model must be C('import_from') and add_pipeline
            must be called with parameter import_from='train'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        pipeline['var'] = pipeline['var'] or []
        if not isinstance(pipeline['var'], list):
            pipeline['var'] = [pipeline['var']]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        import_config = {key: self.pipelines[value]['ppl'] for key, value in pipeline['kwargs'].items()}

        pipeline_config = Config(config.config() + self.config['worker_config'] + import_config)

        pipeline['execute_for'] = self.get_iterations(pipeline['execute_for'], self.config['n_iters'])
        if pipeline['dump_for'] is not None:
            pipeline['dump_for'] = self.get_iterations(pipeline['dump_for'], self.config['n_iters'])
        else:
            pipeline['dump_for'] = []

        pipeline['ppl'].set_config(pipeline_config)

        self.pipelines[name] = pipeline

        self.results[name] = {var: [] for var in pipeline['var']}
        self.results[name]['iterations'] = []

    def execute_for(self, batch, name):
        self.pipelines[name]['ppl'].execute_for(batch)

    def next_batch(self, name):
        """
        Get next batch from pipleine.

        Parameters
        ----------
        name : str
            pipeline name
        """
        self.pipelines[name]['ppl'].next_batch()

    def run(self, name, reset=True):
        """ Run pipelines till the end. """
        if reset:
            self.pipelines[name]['ppl'].reset_iter()
        self.pipelines[name]['ppl'].run()

    def post_run(self, name):
        """ Run function after run. """
        res = self.pipelines[name]['post_run'](self.pipelines[name]['ppl'])
        for key, value in res.items():
            if key in self.results[name]:
                self.results[name][key].append(value)
            else:
                self.results[name][key] = [value]

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

    def _variable_len(self, name, variable):
        if name in self.pipelines:
            return len(self.pipelines[name]['ppl'].get_variable(variable))
        else:
            return None

    def get_pipeline(self, name):
        """
        Parameters
        ----------
        name : str
        """
        return self.pipelines[name]

    def run_on_batch(self, batch, name):
        """
        Run pipeline on prepared batch.

        Parameters
        ----------
        batch : dataset.Batch

        name : str
            pipeline name
        """
        self.pipelines[name]['ppl'].execute_for(batch)

    def put_result(self, iteration, name):
        """ Put pipeline variable into results. """
        for var in self.pipelines[name]['var']:
            self.results[name][var].append(self.pipelines[name]['ppl'].get_variable(var))
        self.results[name]['iterations'].append(iteration)

    def dump_result(self, name, path):
        """ Dump results. """
        foldername, _ = os.path.split(path)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(path, 'wb') as file:
            pickle.dump(self.results[name], file)
