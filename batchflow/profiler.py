""" Profiler for batchflow units """
from time import perf_counter
from pstats import Stats
from cProfile import Profile
import threading
import warnings

from .utils_import import make_delayed_import
pd = make_delayed_import('pandas')


class Profiler:
    """ Profiler for batchflow units.

    Parameters
    ----------
    profile : bool or {0, 1, 2} or 'detailed'
        whether to use profiler
    """
    UNIT_NAME = 'action'


    def __init__(self, detailed=True):
        self.start_time = None
        self.detailed = detailed
        self.profiler = Profile() if detailed else None
        self.iteration = 0

        self._profile_info = []  # list of dicts with info about each item
        self._profile_info_lock = threading.Lock()

    @property
    def profile_info(self):
        """ Prepare profile results dataframe. """
        if not self._profile_info:
            return None
        df = pd.DataFrame(self._profile_info).set_index('name').sort_values('iter')
        df.index.name = self.UNIT_NAME

        n_unique_units = df.index.unique().size
        df['iter'] = df['iter'] // n_unique_units + 1
        return df

    def enable(self):
        """ Start profiling. """
        self.start_time = perf_counter()
        if self.detailed:
            self.profiler.enable()

    def disable(self, iteration, name, **kwargs):
        """ Disable profiling. """
        total_time = perf_counter() - self.start_time

        if self.detailed:
            self.profiler.disable()
            stats = Stats(self.profiler)
            self.profiler.clear()

            values = []
            for key, value in stats.stats.items():
                for k, v in value[4].items():
                    call_id = f'{key[2]}::{k[0]}::{k[1]}::{k[2]}' # method_name, file_name, line_no, callee
                    row_dict = {
                        'name': name, 'id': call_id,
                        'iter': self.iteration, 'outer_iter': iteration,
                        'total_time': total_time, 'eval_time': stats.total_tt, # base stats
                        'ncalls': v[0], 'tottime': v[2], 'cumtime': v[3], # detailed stats
                        **kwargs
                    }
                    values.append(row_dict)
        else:
            values = [{
                'name': name,
                'iter': self.iteration, 'outer_iter': iteration,
                'total_time': total_time,
                **kwargs
            }]

        with self._profile_info_lock:
            self._profile_info.extend(values)
            self.iteration += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_profile_info_lock')
        state['profiler'] = None
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

        self.profiler = Profiler() if self.detailed else None
        self._profile_info_lock = threading.Lock()

    def __add__(self, other):
        """ Combine multiple profilers with their collected info concatenated. """
        new = type(self)(detailed=self.detailed)
        new._profile_info = self._profile_info + other._profile_info
        return new


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
        profile_info = self.profile_info
        if profile_info is None:
            warnings.warn("Profiling has not been enabled.")
            return None

        detailed = False if not self.detailed else detailed

        if per_iter is False and detailed is False:
            columns = columns or ['total_time']
            sortby = sortby or ('total_time', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (profile_info.groupby(['action', 'iter'])[columns]
                      .mean(numeric_only=True).groupby('action').agg(aggs, numeric_only=True)
                      .sort_values(sortby, ascending=False))

        elif per_iter is False and detailed is True:
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or ('tottime', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (profile_info.reset_index().groupby(['action', 'id']).agg(aggs, numeric_only=True)
                      .sort_values(['action', sortby], ascending=[True, False])
                      .groupby(level=0).apply(lambda df: df[:limit]).droplevel(0))

        elif per_iter is True and detailed is False:
            groupby = groupby or ['iter', 'action']
            columns = columns or ['action', 'total_time', 'batch_id']
            sortby = sortby or 'total_time'
            result = (profile_info.reset_index().groupby(groupby)[columns].mean(numeric_only=True)
                      .sort_values(['iter', sortby], ascending=[True, False]))

        else: # per_iter is True and detailed is True:
            groupby = groupby or ['iter', 'action', 'id']
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or 'tottime'
            result = (profile_info.reset_index().set_index(groupby)[columns]
                      .sort_values(['iter', 'action', sortby], ascending=[True, True, False])
                      .groupby(level=[0, 1]).apply(lambda df: df[:limit]).droplevel([0, 1]))
        return result
