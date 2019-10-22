from .results import Results
from .. import NamedExpression
from ..named_expr import NamedExpression, add_ops, AN_EXPR, UNARY_OPS, OPERATIONS

class ResearchNamedExpression(NamedExpression):
    param_names = ('job', 'iteration', 'experiment', 'path', 'batch', 'pipeline', 'model')

    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.op_class = ResearchNamedExpression

    @classmethod
    def _get_params(cls, job=None, iteration=None, experiment=None, path=None, **kwargs):
        return dict(job=job, iteration=iteration, experiment=experiment, path=path, **kwargs)

class ResearchExecutableUnit(ResearchNamedExpression): # ExecutableUnit
    def get(self, **kwargs):
        if isinstance(kwargs['experiment'], (list, tuple)):
            _experiment = kwargs['experiment']
        else:
            _experiment = [kwargs['experiment']]
        if self.name is not None:
            res = [item[self.name] for item in _experiment]
            if len(_experiment) == 1:
                return res[0]
            else:
                return res
        else:
            return kwargs['experiment']

class ResearchPipeline(ResearchExecutableUnit): # ResearchPipeline
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
        else:
            return getattr(res, attr)

class ResearchIteration(ResearchNamedExpression): # research iteration
    def get(self, **kwargs):
        return kwargs['iteration']

class ResearchConfig(ResearchExecutableUnit):
    def get(self, **kwargs):
        if self.name is None:
            raise ValueError('`name` must be defined for RC expressions')
        res = super().get(**kwargs)

        if isinstance(res, list):
            return [getattr(item, 'config') for item in res]
        else:
            return getattr(res, 'config')

class ResearchResults(ResearchNamedExpression): # research results
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs

    def get(self, **kwargs):
        if kwargs['path'] is None:
            path = kwargs['job'].research_path
        else:
            path = kwargs['path']
        return Results(path=path).load(*self.args, **self.kwargs)
