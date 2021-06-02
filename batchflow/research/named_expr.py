""" Contains named expression classes for Research """

import os

from ..named_expr import NamedExpression, eval_expr

class E(NamedExpression):
    def __init__(self, unit=None, all=False, **kwargs):
        self.name = None
        self.unit = unit
        self.all = all

    def _get(self, **kwargs):
        experiment = kwargs['experiment']
        if self.all:
            return experiment.executor.experiments
        return [experiment]

    def get(self, **kwargs):
        experiments = self._get(**kwargs)
        results = self.transform(experiments)
        if self.all:
            return results
        return results[0]

    def transform(self, experiments):
        if self.unit is not None:
            return [exp[self.unit] for exp in experiments]
        return experiments

class EC(E):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def transform(self, experiments):
        if self.name is None:
            return [exp.config for exp in experiments]
        else:
            return [exp.config[self.name] for exp in experiments]

class O(E):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def transform(self, experiments):
        return [exp[self.name].output for exp in experiments]
