""" Contains named expression classes for Research """

from ..named_expr import NamedExpression

class E(NamedExpression):
    """ NamedExpression for Experiment or its unit in Research.

    Parameters
    ----------
    unit : str, optional
        name of unit to, by default None. If None, experiment will be returned.
    all : bool, optional
        if True, return all experiments in executor, otherwise just one, by default False.
    """
    def __init__(self, unit=None, all=False, **kwargs):
        _ = kwargs
        self.name = None
        self.unit = unit
        self.all = all

    def _get(self, **kwargs):
        experiment = kwargs['experiment']
        if self.all:
            return experiment.executor.experiments
        return [experiment]

    def get(self, **kwargs):
        """ Return a value. """
        experiments = self._get(**kwargs)
        results = self._transform(experiments)
        if self.all:
            return results
        return results[0]

    def _transform(self, experiments):
        if self.unit is not None:
            return [exp[self.unit] for exp in experiments]
        return experiments

    def __copy__(self):
        return self

class EC(E):
    """ NamedExpression for Experiment config.

    Parameters
    ----------
    name : str, optional
        key of config to get, by default None. If None, return entire config.
    full : bool, optional
        return aixilary key ('device', 'repetition', etc.) or not, by default False.
    """
    def __init__(self, name=None, full=False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.full = full

    def _transform(self, experiments):
        if self.name is None:
            return [self.remove_keys(exp.config) for exp in experiments]
        return [exp.config[self.name] for exp in experiments]

    def remove_keys(self, config):
        if self.full:
            return config
        else:
            return {key: config[key] for key in config if key not in ['device', 'repetition', 'updates']}

class O(E):
    """ NamedExpression for ExecutableUnit output.

    Parameters
    ----------
    name : str
        name of the unit to get output.
    """
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def _transform(self, experiments):
        return [exp[self.name].output for exp in experiments]
