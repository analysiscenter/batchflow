""" Transformations of ranges of values: normalizations and quantizations. """
import numpy as np

class Normalizer:
    """ Class to hold parameters and methods for (de)normalization with provided stats.
    Depending on the parameters, stats for normalization will be taken from (in order of priority):
        - supplied `normalization_stats`, if provided
        - computed from `array` directly

    Parameters
    ----------
    mode : {'mean', 'std', 'meanstd', 'minmax'} or callable
        If str, then normalization description.
        If callable, then it will be called on `src` data with additional `normalization_stats` argument.
    clip_to_quantiles : bool
        Whether to clip the data to quantiles, specified by `q` parameter.
        Quantile values are taken from `normalization_stats`, provided by either of the ways.
    q : tuple of numbers
        Quantiles for clipping. Used as keys to `normalization_stats`, provided by either of the ways.
    normalization_stats : dict, optional
        If provided, then used to get statistics for normalization.
        Otherwise, compute them from a given array. 
    inplace : bool
        Whether to apply operation inplace or return a new array.
    """
    def __init__(self, mode='meanstd', clip_to_quantiles=False, q=(0.01, 0.99),
                 normalization_stats=None, inplace=False):
        self.mode = mode
        self.clip_to_quantiles = clip_to_quantiles
        self.q = q
        self.normalization_stats = normalization_stats
        self.inplace = inplace

    def normalize(self, array, normalization_stats=None, mode=None, return_stats=False):
        """ Normalize image with provided stats.

        Parameters
        ----------
        array : numpy.ndarray

        normalization_stats : dict or None, optional
            If provided, then used to get statistics for normalization, by default None.
            If None, self.normalization_stats will be used.
            If self.normalization_stats is also None, then statistics will by computed by array.
        return_stats : bool, optional
            Whether to return stats used for normalization, by default False

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, dict)
        """
        clip_to_quantiles = self.clip_to_quantiles
        mode = self.mode if mode is None else mode
        array = array if self.inplace else array.copy()

        normalization_stats = normalization_stats if normalization_stats is not None else self.normalization_stats

        if normalization_stats is None:
            normalization_stats = {}

            if clip_to_quantiles:
                normalization_stats['q'] = np.quantile(array, self.q)
                np.clip(array, *normalization_stats['q'], out=array)
                clip_to_quantiles = False

            if isinstance(mode, str):
                if 'mean' in mode:
                    normalization_stats['mean'] = np.mean(array)
                if 'std' in mode:
                    normalization_stats['std'] = np.std(array)
                if 'min' in mode:
                    normalization_stats['min'] = np.min(array)
                if 'max' in mode:
                    normalization_stats['max'] = np.max(array)
    
        if clip_to_quantiles:
            np.clip(array, *normalization_stats['q'], out=array)

        # Actual normalization
        if callable(mode):
            array[:] = mode(array, normalization_stats)
        else:
            if 'mean' in mode:
                array -= normalization_stats['mean']
            if 'std' in mode:
                array /= normalization_stats['std'] + 1e-6
            if 'min' in mode and 'max' in mode:
                if clip_to_quantiles:
                    min_, max_ = normalization_stats['q']
                else:
                    min_, max_ = normalization_stats['min'], normalization_stats['max']

                array -= min_
                if min_ != max_:
                    array /= (max_ - min_)

        if return_stats:
            return array, normalization_stats
        return array

    def __call__(self, array):
        return self.normalize(array)

    def denormalize(self, array, normalization_stats=None, mode=None):
        """ Deormalize image with provided stats.

        Parameters
        ----------
        array : numpy.ndarray

        normalization_stats : dict or None, optional
            If provided, then used to get statistics for denormalization, by default None.
            If None, self.normalization_stats will be used.
        return_stats : bool, optional
            Whether to return stats used for normalization, by default False

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, dict)
        """
        array = array if self.inplace else array.copy()
        mode = self.mode if mode is None else mode

        if self.normalization_stats is not None:
            normalization_stats = self.normalization_stats

        if callable(mode):
            array[:] = mode(array, normalization_stats)
        else:
            if 'std' in mode:
                array *= normalization_stats['std']
            if 'mean' in mode:
                array += normalization_stats['mean']
            if 'min' in mode and 'max' in mode:
                if self.clip_to_quantiles:
                    min_, max_ = normalization_stats['q']
                else:
                    min_, max_ = normalization_stats['min'], normalization_stats['max']
 
                if min_ != max_:
                    array *= max_ - min_
                array += min_

        return array

class Quantizer:
    """ Class to hold parameters and methods for (de)quantization. """
    def __init__(self, ranges, clip=True, center=False, mean=None, dtype=np.int8):
        # Parse parameters
        if center:
            ranges = tuple(item - self.v_mean for item in ranges)

        self.ranges = ranges
        self.clip, self.center = clip, center
        self.mean = mean
        self.dtype = dtype

        self.bins = np.histogram_bin_edges(None, bins=254, range=ranges).astype(np.float32)

    def quantize(self, array):
        """ Quantize data: find the index of each element in the pre-computed bins and use it as the new value.
        Converts `array` to int8 dtype. Lossy.
        """
        if self.center:
            array -= self.mean
        if self.clip:
            array = np.clip(array, *self.ranges)
        array = np.digitize(array, self.bins) - 128
        return array.astype(self.dtype)

    def dequantize(self, array):
        """ Dequantize data: use each element as the index in the array of pre-computed bins.
        Converts `array` to float32 dtype. Unable to recover full information.
        """
        array += 128
        array = self.bins[array]
        if self.center:
            array += self.mean
        return array.astype(np.float32)

    def __call__(self, array):
        return self.quantize(array)
