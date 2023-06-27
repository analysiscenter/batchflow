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

    def normalize(self, array, normalization_stats=None, return_stats=False):
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
        array = array if self.inplace else array.copy()

        if normalization_stats is None:
            if self.normalization_stats is not None:
                normalization_stats = self.normalization_stats
            else:
                if clip_to_quantiles:
                    np.clip(array, *np.quantile(array, self.q), out=array)
                    clip_to_quantiles = False

                if callable(self.mode):
                    normalization_stats = {
                        'mean': np.mean(array),
                        'std': np.std(array),
                        'min': np.min(array),
                        'max': np.max(array),
                    }
                else:
                    normalization_stats = {}
                    if 'mean' in self.mode:
                        normalization_stats['mean'] = np.mean(array)
                    if 'std' in self.mode:
                        normalization_stats['std'] = np.std(array)
                    if 'min' in self.mode:
                        normalization_stats['min'] = np.min(array)
                    if 'max' in self.mode:
                        normalization_stats['max'] = np.max(array)
        if clip_to_quantiles:
            np.clip(array, normalization_stats['q_01'], normalization_stats['q_99'], out=array)

        # Actual normalization
        if callable(self.mode):
            array[:] = mode(array, normalization_stats)
        else:
            if 'mean' in self.mode:
                array -= normalization_stats['mean']
            if 'std' in self.mode:
                array /= normalization_stats['std'] + 1e-6
            if 'min' in self.mode and 'max' in self.mode:
                if clip_to_quantiles:
                    array -= normalization_stats['q_01']
                    array /= normalization_stats['q_99'] - normalization_stats['q_01']
                elif normalization_stats['max'] != normalization_stats['min']:
                    array -= normalization_stats['min']
                    array /= normalization_stats['max'] - normalization_stats['min']
                else:
                    array -= normalization_stats['min']
        if return_stats:
            return array, normalization_stats
        return array

    def __call__(self, array):
        return self.normalize(array)

    def denormalize(self, array, normalization_stats=None):
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

        if self.normalization_stats is not None:
            normalization_stats = self.normalization_stats

        if callable(self.mode):
            array[:] = self.mode(array, normalization_stats)
        else:
            if 'std' in self.mode:
                array *= normalization_stats['std'] # TODO: eps to normalize/denormalize?
            if 'mean' in self.mode:
                array += normalization_stats['mean']
            if 'min' in self.mode and 'max' in self.mode:
                if normalization_stats['max'] != normalization_stats['min']:
                    array *= normalization_stats['max'] - normalization_stats['min']
                    array += normalization_stats['min']
                else:
                    array += normalization_stats['min']
        return array

class Quantizer:
    """ Class to hold parameters and methods for (de)quantization. """
    def __init__(self, data, ranges, clip=True, center=False, mean=None, dtype=np.int8):
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
