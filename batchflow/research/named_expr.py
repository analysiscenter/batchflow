from .. import NamedExpression, W
from ..named_expr import add_ops, AN_EXPR, UNARY_OPS, OPERATIONS

class ResearchNamedExpression(NamedExpression):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.op_class = ResearchNamedExpression

    def set_params(self, job=None, iteration=None, experiment=None):
        self.params = job, iteration, experiment

    def str(self):
        """ Convert a named expression value to a string """
        return ResearchNamedExpression(AN_EXPR, op='#str', a=self)

    def get(self, job=None, iteration=None, experiment=None):
        """ Return a value of a named expression

        Parameters
        ----------
        batch
            a batch which should be used to calculate a value
        pipeline
            a pipeline which should be used to calculate a value
            (might be omitted if batch is passed)
        model
            a model which should be used to calculate a value
            (usually omitted, but might be useful for F- and L-expressions)
        """
        if self.params:
            job, iteration, experiment = self.params
        name = self._get_name(job, iteration, experiment)
        if name == AN_EXPR:
            return self._get_value(job, iteration, experiment)
        raise ValueError("Undefined value")

    def _get_name(self, job=None, iteration=None, experiment=None):
        if isinstance(self.name, NamedExpression):
            if self.params:
                job, iteration, experiment = self.params
            return self.name.get(job, iteration, experiment)
        return self.name

    def _get_value(self, job=None, iteration=None, experiment=None):
        if self.name == AN_EXPR:
            a = self.eval_expr(self.a, job, iteration, experiment)
            b = self.eval_expr(self.b, job, iteration, experiment)
            if self.op in UNARY_OPS:
                return OPERATIONS[self.op](a)
            return OPERATIONS[self.op](a, b)
        raise ValueError("Undefined value")

    @classmethod
    def eval_expr(cls, expr, job=None, iteration=None, experiment=None):
        """ Evaluate a named expression recursively """
        args = dict(job=job, iteration=iteration, experiment=experiment)

        if isinstance(expr, NamedExpression):
            job_, iteration_, experiment_ = expr.params or (None, None, None)
            args = {**dict(job=job_, iteration=iteration_, experiment=experiment_),
                    **dict(job=job, iteration=iteration, experiment=experiment)}

            _expr = expr.get(**args)
            if isinstance(expr, W):
                expr = _expr
            elif isinstance(_expr, NamedExpression):
                expr = cls.eval_expr(_expr, **args)
            else:
                expr = _expr
        elif isinstance(expr, (list, tuple)):
            _expr = []
            for val in expr:
                _expr.append(cls.eval_expr(val, **args))
            expr = type(expr)(_expr)
        elif isinstance(expr, dict):
            _expr = type(expr)()
            for key, val in expr.items():
                key = cls.eval_expr(key, **args)
                val = cls.eval_expr(val, **args)
                _expr.update({key: val})
            expr = _expr
        return expr

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
