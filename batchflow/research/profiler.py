#pylint: disable=super-init-not-called
""" Research profilers. """

from cProfile import Profile
import warnings
import multiprocess as mp

try:
    import pandas as pd
except ImportError:
    from . import _fake as pd

from ..profiler import Profiler

class ExperimentProfiler(Profiler):
    """ Profiler for Research experiments. """
    def show_profile_info(self, per_iter=False, per_experiment=False, detailed=False,
                          groupby=None, columns=None, sortby=None, limit=10):
        """ Show stored profiling information with varying levels of details.

        Parameters
        ----------
        per_iter : bool
            Whether to make an aggregation over iters or not.
        detailed : bool
            Whether to use information from :class:`cProfiler` or not.
        groupby : str or sequence of str
            Used only when `per_iter` is True, directly passed to pandas.
        columns : sequence of str
            Columns to show in resultining dataframe.
        sortby : str or tuple of str
            Column id to sort on. Note that if data is aggregated over iters (`per_iter` is False),
            then it must be a full identificator of a column.
        limit : int
            Limits the length of resulting dataframe.
        parse : bool
            Allows to re-create underlying dataframe from scratches.
        """
        if self.profile_info is None:
            warnings.warn("Profiling has not been enabled.")
            return None

        detailed = False if not self.detailed else detailed

        if per_iter is False and detailed is False:
            groupby = ['experiment', 'action', 'iter'] if per_experiment else ['action', 'iter']
            columns = columns or ['total_time', 'pipeline_time']
            sortby = sortby or ('total_time', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.groupby(groupby)[columns].mean()
                      .groupby(groupby[:-1]).agg(aggs)
                      .sort_values(sortby, ascending=False))

            if per_experiment:
                result = result.sort_index(level=0)

        elif per_iter is False and detailed is True:
            groupby = ['experiment', 'action', 'id'] if per_experiment else ['action', 'id']
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or ('tottime', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            level = [0, 1] if per_experiment else 0
            result = (self.profile_info.reset_index().groupby(groupby).agg(aggs)
                    .sort_values([*groupby[:-1], sortby], ascending=[True] * (len(groupby)-1) + [False])
                    .groupby(level=level).apply(lambda df: df[:limit]).droplevel(level))

        elif per_iter is True and detailed is False:
            groupby = groupby or ['iter', 'action']
            groupby = ['experiment', *groupby] if per_experiment else groupby

            columns = columns or ['action', 'total_time', 'pipeline_time']
            sortby = sortby or 'total_time'
            result = (self.profile_info.reset_index().groupby(groupby)[columns].mean()
                      .sort_values([*groupby[:-1], sortby], ascending=[True] * (len(groupby)-1) + [False]))

        elif per_iter is True and detailed is True:
            groupby = groupby or ['iter', 'action', 'id']
            groupby = ['experiment', *groupby] if per_experiment else groupby

            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or 'tottime'
            result = (self.profile_info.reset_index().set_index(groupby)[columns]
                      .sort_values([*groupby[:-1], sortby], ascending=[True] * (len(groupby)-1) + [False])
                      .groupby(level=[0, 1]).apply(lambda df: df[:limit]).droplevel([0, 1]))
        return result

class ExecutorProfiler(ExperimentProfiler):
    """ Profiler for Executor experiments. """
    def __init__(self, experiments):
        self.profilers = [experiment._profiler for experiment in experiments]
        self.detailed = experiments[0]._profiler.detailed

    @property
    def profile_info(self):
        return pd.concat([profiler.profile_info for profiler in self.profilers])

class ResearchProfiler(ExperimentProfiler):
    """ Profiler for Research. """
    def __init__(self, detailed=True):
        self.detailed = detailed
        self.experiments_info = mp.Manager().dict()

    def put(self, experiment, df):
        self.experiments_info[experiment] = df

    @property
    def profile_info(self):
        return pd.concat([self.experiments_info[exp] for exp in self.experiments_info])
