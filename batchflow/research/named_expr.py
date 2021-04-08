""" Contains named expression classes for Research """

import os

from .results import Results
from ..named_expr import NamedExpression, eval_expr

class ResearchNamedExpression(NamedExpression):
    """ NamedExpression base class for Research objects """
    def _get(self, **kwargs):
        name = self._get_name(**kwargs)
        return name, kwargs

class REU(ResearchNamedExpression): # ResearchExecutableUnit
    """ NamedExpression for ExecutableUnit """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        experiment = kwargs['experiment']
        return experiment

    def get(self, **kwargs):
        experiment = self._get(**kwargs)
        if isinstance(experiment, (list, tuple)):
            _experiment = experiment
        else:
            _experiment = [experiment]
        name = self.name if self.name is not None else list(_experiment[0].keys())[0]
        res = [item[name] for item in _experiment]
        if len(_experiment) == 1:
            return res[0]
        return res

class RP(REU): # ResearchPipeline
    """ NamedExpression for Pipeline in Research """
    def __init__(self, name=None, root=False):
        super().__init__(name)
        self.root = root

    def get(self, **kwargs):
        if self.name is None:
            raise ValueError('`name` must be defined for RP expressions')
        res = super().get(**kwargs)
        attr = 'root_pipeline' if self.root else 'pipeline'

        if isinstance(res, list):
            return [getattr(item, attr) for item in res]
        return getattr(res, attr)

class RI(ResearchNamedExpression): # ResearchIteration
    """ NamedExpression for iteration of Research """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['iteration']

    def get(self, **kwargs):
        iteration = self._get(**kwargs)
        return iteration

class RC(REU): # ResearchConfig
    """ NamedExpression for Config of the ExecutableUnit """
    def __init__(self, name=None):
        super().__init__(name=name)

    def get(self, **kwargs):
        res = super().get(**kwargs)

        if isinstance(res, list):
            return [getattr(item, 'config') for item in res]
        return getattr(res, 'config')

class RD(ResearchNamedExpression): # ResearchDir
    """ NamedExpression for folder with the Research """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['path']

    def get(self, **kwargs):
        path = self._get(**kwargs)
        return path

class RID(ResearchNamedExpression): # ResearchExperimentID
    """ NamedExpression for id (experiment_id) for the current experiment """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['job'], kwargs['experiment'], kwargs.get('index')

    def get(self, **kwargs):
        job, experiment, index = self._get(**kwargs)
        if index is None:
            unit = list(experiment.values())[0]
            return job.ids[unit.index]
        else:
            return job.ids[index]

class REP(ResearchNamedExpression): # ResearchExperimentPath
    """ NamedExpression for path to folder corresponding to the current config """
    def __init__(self, name=None, relative=False):
        """ NamedExpression for path inside to experiment folder.

        Parameters
        ----------
        name : str or None
            NamedExpression name
        relative : bool
            if False, absolute path including name of the root research folder,
            if True, the path inside of the research folder.
        """
        super().__init__(name)
        self.relative = relative

    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['job'], kwargs['experiment'], kwargs['path'], kwargs.get('experiment_path')

    def get(self, **kwargs):
        _, experiment, research_path, experiment_path = self._get(**kwargs)
        # unit to get attributes (each unit will have the same so we take the first)
        if experiment_path is None:
            unit = next(iter(experiment.values()))
            experiment_path = unit.experiment_path # path to folder with current experiment
        if self.relative:
            return experiment_path
        return os.path.join(research_path, experiment_path)

class RR(ResearchNamedExpression): # ResearchResults
    """ NamedExpression for Results of the Research """
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs

    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        if kwargs['path'] is None:
            path = kwargs['job'].research_path
        else:
            path = kwargs['path']
        return path

    def get(self, **kwargs):
        path = self._get(**kwargs)
        return Results(path, *eval_expr(self.args, **kwargs), **eval_expr(self.kwargs, **kwargs))
