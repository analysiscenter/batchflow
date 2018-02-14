#pylint:disable=broad-except
#pylint:disable=too-few-public-methods

""" Class for research results. """

import os
import glob
import numpy as np
import dill

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
