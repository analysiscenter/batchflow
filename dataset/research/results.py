#pylint:disable=broad-except
#pylint:disable=too-few-public-methods

""" Class for research results. """

import os
import glob
import numpy as np
import dill
import seaborn as sns
import matplotlib.pyplot as plt

from ..config import Config


class Stat:
    """ Get statistics from research results. """
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            name of research to load
        """
        self.name = name

        self._get_research()
        self.results = None

    def _get_research(self):
        with open(os.path.join(self.name, 'description'), 'rb') as file:
            self.research = dill.load(file)

    def load_stat(self, alias, index=None):
        """ Load results of research. """
        results = self._empty_results()

        if index is None:
            mask = '*'
        elif isinstance(index, int):
            mask = str(index)
        else:
            mask = '[' + ','.join([str(i) for i in index]) + ']'
        path_mask = os.path.join(self.name, 'results', alias, mask)
        for name in glob.iglob(path_mask):
            with open(name, 'rb') as file:
                self._put_result(results, dill.load(file))

        return self._list_to_array(results)

    def _put_result(self, results, new_result):
        for name in new_result:
            for variable in new_result[name]:
                results[name][variable].append(new_result[name][variable])

    def _list_to_array(self, results):
        try:
            for name in results:
                for variable in results[name]:
                    results[name][variable] = np.array(results[name][variable])
                results[name]['iterations'] = results[name]['iterations'][0]
            return results
        except Exception:
            return results

    def _empty_results(self):
        results = dict()
        for name, pipeline in self.research.pipelines.items():
            results[name] = {variable: [] for variable in pipeline['var']}
            results[name]['iterations'] = []
        return Config(results)

    def _get_aliases(self, aliases):
        if isinstance(aliases, str):
            aliases = [aliases]
        return aliases

    def plot_density(self, aliases, pipeline, variable, iteration,
                     window=0, figsize=None, axes=None, xlim=None, ylim=None,
                     *args, **kwargs):
        """ Plot density of metric values. """
        aliases = self._get_aliases(aliases)
        results = [self.load_stat(alias)[pipeline] for alias in aliases]

        left = iteration - window
        right = iteration + window + 1

        for i, result in enumerate(results):
            _range = np.where(np.isin(result['iterations'], np.arange(left, right)))
            results[i] = result[variable][:, _range].reshape(-1)

        fig = plt.figure(figsize=figsize)
        axes = axes or [0.1, 0.4, 0.8, 0.5]
        plt_ax = fig.add_axes(axes)
        for result, alias in zip(results, aliases):
            sns.distplot(result, label=alias, ax=plt_ax, *args, **kwargs)
        if xlim is not None:
            plt_ax.set_xlim(xlim)
        if ylim is not None:
            plt_ax.set_ylim(ylim)
        plt_ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        plt.show()
