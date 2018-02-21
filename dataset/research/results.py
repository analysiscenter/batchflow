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
        aliases = self._get_aliases(aliases)
        results = [self.load_stat(alias)[pipeline] for alias in aliases]

        left = iteration - window
        right = iteration + window + 1
        
        for i, result in enumerate(results):
            results[i] = result[variable][:, np.where(np.isin(result['iterations'], np.arange(left, right)))].reshape(-1)

        fig = plt.figure(figsize=figsize)
        axes = axes or [0.1, 0.4, 0.8, 0.5]
        ax = fig.add_axes(axes)
        for result, alias in zip(results, aliases):
            sns.distplot(result, label=alias, ax=ax, *args, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        plt.show()

"""
    def plot_density2(self, iteration, params_ind=None, metric='loss', window=0,
                     mode=None, xlim=None, ylim=None, axes=None, figsize=None,
                     show=True, *args, **kwargs):

        params_ind = self._select_params(params_ind)
        left = max(iteration-window, 0)
        right = min(iteration+window+1, self.n_iters)
        if figsize is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=figsize)
        if axes is None:
            axes = [0.1, 0.4, 0.8, 0.5]
        ax = fig.add_axes(axes)

        if params_ind is None:
            params_ind = list(range(len(self.results)))

        for ind in params_ind:
            stat = self.results[ind]
            x = np.array(stat[-1][mode][metric])[:, left:right]
            x = x.reshape(-1)
            label = stat[0]+'_'+self._alias_to_str(stat[2])
            sns.distplot(x, label=label, ax=ax, *args, **kwargs)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title("{} {}: iteration {}".format(mode, metric, iteration+1))
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        if show:
            plt.show()
        return ax

    def make_video(self, vizualization, name, params_ind=None, plots_per_sec=1., key_frames=None, *args, **kwargs):

        name = os.path.join(self.name, name)
        if os.path.isfile(name):
            os.remove(name)
        tmp_folder = os.path.join(self.name, '.tmp')

        try:
            call(['ffmpeg.exe'])
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg.exe was not found.")

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)

        self._clear_tmp_folder()

        for iteration in range(self.n_iters):
            if key_frames is not None:
                frame = self._get_frame(iteration, key_frames)
                kwargs = {**kwargs, **frame}
            mask = '{:0' + str(int(np.ceil(np.log10(self.n_iters)))) + 'd}.png'
            mask = os.path.join(tmp_folder, '') + mask
            self.vizualizations[vizualization](iteration, params_ind, show=False, *args, **kwargs)
            plt.savefig(mask.format(iteration))
            plt.close()

        mask = '%0{}d.png'.format(int(np.ceil(np.log10(self.n_iters))))
        mask = os.path.join(tmp_folder, mask)
        res = call(["ffmpeg.exe", "-r", str(plots_per_sec), "-i", mask, "-c:v", "libx264", "-vf",
                    "fps=25", "-pix_fmt", "yuv420p", name])
        self._clear_tmp_folder()
        if res != 0:
            raise OSError("Video can't be created")
"""