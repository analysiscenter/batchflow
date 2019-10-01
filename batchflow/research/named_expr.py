class ResearchNamedExpression:
    def __init__(self, name=None):
        self.name = name

    def get(self, job, iteration, experiment, *args, **kwargs):
        raise NotImplementedError('The method `get` should be implemented in child-classes!')


class EU(ResearchNamedExpression): # ExecutableUnit
    def __init__(self, name=None):
        self.name = name

    def get(self, job, iteration, experiment):
        _ = job, iteration
        if self.name is None:
            return experiment[self.name]
        else:
            return experiment

class RI(ResearchNamedExpression): # research iteration
    def get(self, job, iteration, experiment):
        _ = job, iteration, experiment
        return iteration

class RR(ResearchNamedExpression): # research results
    pass