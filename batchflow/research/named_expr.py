class ResearchNamedExpression:
    def __init__(self, name=None):
        self.name = name

    def get(self, job, iteration, experiment):
        raise NotImplementedError('The method `get` should be implemented in child-classes!')


class EU(ResearchNamedExpression): # ExecutableUnit
    def get(self, job, iteration, experiment):
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
            return experiment

class RP(EU): # ResearchPipeline
    def __init__(self, name=None, root=False):
        super().__init__(name)
        self.root = root

    def get(self, job, iteration, experiment):
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
    def get(self, job, iteration, experiment):
        _ = job, iteration, experiment
        return iteration

class RR(ResearchNamedExpression): # research results
    def get(self, job, iteration, experiment):
        pass
