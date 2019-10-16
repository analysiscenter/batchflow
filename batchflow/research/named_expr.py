from collections import OrderedDict

from .. import NamedExpression
from ..named_expr import add_ops, AN_EXPR, UNARY_OPS, OPERATIONS

class ResearchNamedExpression(NamedExpression):
    param_names = 'job', 'iteration', 'experiment'

    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.op_class = ResearchNamedExpression

    @classmethod
    def _get_params(cls, job=None, iteration=None, experiment=None):
        return OrderedDict(job=job, iteration=iteration, experiment=experiment)

class EU(ResearchNamedExpression): # ExecutableUnit
    def get(self, job=None, iteration=None, experiment=None):
        _ = job, iteration
        if isinstance(experiment, (list, tuple)):
            _experiment = experiment
        else:
            _experiment = [experiment]
        if self.name is not None:
            res = [item[self.name] for item in _experiment]
            if len(_experiment) == 1:
                return res[0]
            else:
                return res
        else:
            return experiment

class RP(EU): # ResearchPipeline
    def __init__(self, name=None, root=False):
        super().__init__(name)
        self.root = root

    def get(self, job=None, iteration=None, experiment=None):
        _ = job, iteration
        if self.name is None:
            raise ValueError('`name` must be defined for RP expressions')
        res = super().get(job, iteration, experiment)
        attr = 'root_pipeline' if self.root else 'pipeline'

        if isinstance(res, list):
            return [getattr(item, attr) for item in res]
        else:
            return getattr(res, attr)

class RI(ResearchNamedExpression): # research iteration
    def get(self, job=None, iteration=None, experiment=None):
        _ = job, iteration, experiment
        return iteration

# class RR(ResearchNamedExpression): # research results
#     def __init__(self, *args, **kwargs):
#         self.args = args
#         self.kwargs = kwargs

#     def get(self, job, iteration, experiment):
#         path = job.research_path
#         return Results(path=path).load(*self.args, **self.kwargs)
