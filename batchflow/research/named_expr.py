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
        if self.name is not None:
            res = [item[self.name] for item in _experiment]
            if len(_experiment) == 1:
                return res[0]
            return res
        return experiment

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
    def __init__(self, name):
        super().__init__(name=name)

    def get(self, **kwargs):
        res = super().get(**kwargs)

        if isinstance(res, list):
            return [getattr(item, 'config') for item in res]
        return getattr(res, 'config')

class RD(ResearchNamedExpression): # ResearchDir
    """ NamedExpression for fodler with the Research """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['path']

    def get(self, **kwargs):
        path = self._get(**kwargs)
        return path

class RID(ResearchNamedExpression): # ResearchExperimentID
    """ NamedExpression for id (sample_index) for the current experiment """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['job'], kwargs['experiment']

    def get(self, **kwargs):
        job, experiment = self._get(**kwargs)
        unit = list(experiment.values())[0]
        return job.ids[unit.index]

class REP(ResearchNamedExpression): # ResearchExperimentPath
    """ NamedExpression for path to the current experiment """
    def _get(self, **kwargs):
        _, kwargs = super()._get(**kwargs)
        return kwargs['job'], kwargs['experiment']

    def get(self, **kwargs):
        job, experiment = self._get(**kwargs)
        unit = list(experiment.values())[0]
        return os.path.join(unit.path, job.ids[unit.index])

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
