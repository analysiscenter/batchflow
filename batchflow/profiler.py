""" Profiler for batchflow units """

import time
from pstats import Stats
from cProfile import Profile
import threading
import warnings

try:
    import pandas as pd
except ImportError:
    from . import _fake as pd

class Profiler:
    """ Profiler for batchflow units.

    Parameters
    ----------
    profile : bool or {0, 1, 2} or 'detailed'
        whether to use profiler
    """

    UNIT_NAME = 'action'

    def __init__(self, detailed=True):
        if detailed:
            self.detailed = True
            self._profiler = Profile()
        else:
            self.detailed = False
            self._profiler = None

        self._profile_info = []
        self._profile_info_lock = threading.Lock()
        self.start_time = None

    @property
    def profile_info(self):
        return pd.concat(self._profile_info)

    def enable(self):
        """ Enable profiling. """
        self.start_time = time.time()
        if self.detailed:
            self._profiler.enable()

    def disable(self, iteration, name, **kwargs):
        """ Disable profiling. """
        if self.detailed:
            self._profiler.disable()
        total_time = time.time() - self.start_time
        self._add_profile_info(iteration, name, start_time=self.start_time, total_time=total_time, **kwargs)

    def _add_profile_info(self, iter_no, name, total_time, **kwargs):
        if self.detailed:
            stats = Stats(self._profiler)
            self._profiler.clear()

            indices, values = [], []
            for key, value in stats.stats.items():
                for k, v in value[4].items():
                    # action name, method_name, file_name, line_no, callee
                    indices.append((name, '{}::{}::{}::{}'.format(key[2], *k)))
                    row_dict = {
                        'iter': iter_no, 'total_time': total_time, 'eval_time': stats.total_tt, # base stats
                        'ncalls': v[0], 'tottime': v[2], 'cumtime': v[3], # detailed stats
                        **kwargs
                    }
                    values.append(row_dict)
        else:
            indices = [(name, '')]
            values = [{'iter': iter_no, 'total_time': total_time, 'eval_time': total_time,
                    **kwargs}]

        multiindex = pd.MultiIndex.from_tuples(indices, names=[self.UNIT_NAME, 'id'])
        df = pd.DataFrame(values, index=multiindex)

        with self._profile_info_lock:
            self._profile_info.append(df)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_profile_info_lock')
        state['_profiler'] = None
        return state

    def __setstate__(self, state):
        self._profile_info_lock = threading.Lock()
        for k, v in state.items():
            setattr(self, k, v)

class PipelineProfiler(Profiler):
    """ Profiler for batchflow pipelines. """
    def show_profile_info(self, per_iter=False, detailed=False,
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
            columns = columns or ['total_time', 'eval_time']
            sortby = sortby or ('total_time', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.groupby(['action', 'iter'])[columns].mean().groupby('action').agg(aggs)
                      .sort_values(sortby, ascending=False))

        elif per_iter is False and detailed is True:
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or ('tottime', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.reset_index().groupby(['action', 'id']).agg(aggs)
                      .sort_values(['action', sortby], ascending=[True, False])
                      .groupby(level=0).apply(lambda df: df[:limit]).droplevel(0))

        elif per_iter is True and detailed is False:
            groupby = groupby or ['iter', 'action']
            columns = columns or ['action', 'total_time', 'eval_time', 'batch_id']
            sortby = sortby or 'total_time'
            result = (self.profile_info.reset_index().groupby(groupby)[columns].mean()
                      .sort_values(['iter', sortby], ascending=[True, False]))

        elif per_iter is True and detailed is True:
            groupby = groupby or ['iter', 'action', 'id']
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or 'tottime'
            result = (self.profile_info.reset_index().set_index(groupby)[columns]
                      .sort_values(['iter', 'action', sortby], ascending=[True, True, False])
                      .groupby(level=[0, 1]).apply(lambda df: df[:limit]).droplevel([0, 1]))
        return result
