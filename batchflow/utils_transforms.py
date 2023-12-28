""" Transformations of ranges of values: normalizations and quantizations. """
import numpy as np

try:
    import bottleneck as bn
except ImportError:
    bn = np

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
    expect_nan : bool
        Whether to process nan values in statistics evaluation or not.
        `np.nan*`/`bn.nan*` functionality is slower than default statistics computation, but sometimes it is necessary.
    """

    def __init__(self, mode='meanstd', clip_to_quantiles=False, q=(0.01, 0.99),
                 normalization_stats=None, expect_nan=False):
        self.mode = mode
        self.clip_to_quantiles = clip_to_quantiles
        self.q = q
        self.normalization_stats = normalization_stats
        self.expect_nan = expect_nan

    def normalize(self, array, normalization_stats=None, mode=None, return_stats=False, inplace=False,
                  clip_to_quantiles=None):
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
        inplace : bool
            Whether to apply operation inplace or return a new array.

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, dict)
        """
        clip_to_quantiles = self.clip_to_quantiles if clip_to_quantiles is None else clip_to_quantiles
        mode = self.mode if mode is None else mode
        normalization_stats = self.normalization_stats if normalization_stats is None else normalization_stats

        array = array if inplace else array.copy()

        if normalization_stats is None:
            normalization_stats = {}

            if clip_to_quantiles:
                quantiles = np.quantile(array, self.q) if not self.expect_nan else np.nanquantile(array)
                normalization_stats['quantiles'] = quantiles
                np.clip(array, *normalization_stats['quantiles'], out=array)
                clip_to_quantiles = False

            if isinstance(mode, str):
                if 'mean' in mode:
                    normalization_stats['mean'] = np.mean(array) if not self.expect_nan else bn.nanmean(array)
                if 'std' in mode:
                    normalization_stats['std'] = np.std(array) if not self.expect_nan else bn.nanstd(array)
                if 'min' in mode:
                    normalization_stats['min'] = np.min(array) if not self.expect_nan else bn.nanmin(array)
                if 'max' in mode:
                    normalization_stats['max'] = np.max(array) if not self.expect_nan else bn.nanmax(array)
        else:
            if clip_to_quantiles:
                np.clip(array, *normalization_stats['quantiles'], out=array)

        # Actual normalization
        if callable(mode):
            array[:] = mode(array, normalization_stats)
        else:
            if 'mean' in mode:
                array -= normalization_stats['mean']
            if 'std' in mode:
                array /= normalization_stats['std'] + 1e-6
            if mode == 'minmax':
                if clip_to_quantiles:
                    min_, max_ = normalization_stats['quantiles']
                else:
                    min_, max_ = normalization_stats['min'], normalization_stats['max']

                array -= min_
                if min_ != max_:
                    array /= (max_ - min_)

        if return_stats:
            return array, normalization_stats
        return array

    __call__ = normalize

    def denormalize(self, array, normalization_stats=None, mode=None, inplace=False):
        """ Denormalize image with provided stats.

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
        array = array if inplace else array.copy()
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
                    min_, max_ = normalization_stats['quantiles']
                else:
                    min_, max_ = normalization_stats['min'], normalization_stats['max']

                if min_ != max_:
                    array *= max_ - min_
                array += min_

        return array

    def reset_stats(self, normalization_stats=None):
        self.normalization_stats = normalization_stats

class Quantizer:
    """ Class to hold parameters and methods for (de)quantization.

    Parameters
    ----------
    ranges : tuple
        Bounds to create bins.
    clip : bool, optional
        Whether to clip data to selected ranges, by default True
    center : bool, optional
        Whether to make data have 0-mean before quantization, by default False
    mean : _type_, optional
        Mean value for centering, by default None
    dtype : numpy.dtype, optional
        dtype for the quantized array, by default np.int8
    copy : bool, optional
        Whether to make copy of the data under the hood, by default False.
        Enabled copy will not allow to change input data but quantization will be slower.
    """
    def __init__(self, ranges, clip=True, center=False, mean=None, dtype=np.int8):
        if dtype not in [np.int8, np.int16, np.uint8, np.uint16]:
            raise TypeError(f'{dtype} is not supported, use int8, int16, uint8, uint16')
        self.ranges = ranges
        self.clip, self.center = clip, center
        self.mean = mean
        self.dtype = dtype

        n_bins = np.iinfo(dtype).max - np.iinfo(dtype).min - 1
        self.bins = np.histogram_bin_edges(None, bins=n_bins, range=ranges).astype(np.float32)

    def quantize(self, array, copy=False):
        """ Quantize data: find the index of each element in the pre-computed bins and use it as the new value.
        Converts `array` to `self.dtype`. Lossy.

        Parameters
        ----------
        array : numpy.ndarray

        copy : bool, optional
            Whether to make copy of the data under the hood, by default False.
            Enabled copy will not allow to change input data but quantization will be slower.

        Resturns
        --------
        numpy.ndarray
            quantized array
        """
        if copy:
            array = array.copy()
        if self.center:
            array -= self.mean
        if self.clip:
            array = np.clip(array, *self.ranges)

        array = np.digitize(array, self.bins, right=False) + np.iinfo(self.dtype).min
        if not self.clip:
            array[array == np.iinfo(self.dtype).max + 1] = np.iinfo(self.dtype).max # to put maximum value into bin
        return array.astype(self.dtype)

    __call__ = quantize

    def dequantize(self, array, copy=False):
        """ Dequantize data: use each element as the index in the array of pre-computed bins.
        Converts `array` to float32 dtype. Unable to recover full information.

        Parameters
        ----------
        array : numpy.ndarray

        copy : bool, optional
            Whether to make copy of the data under the hood, by default False.
            Enabled copy will not allow to change input data but dequantization will be slower.

        Resturns
        --------
        numpy.ndarray
            dequantized array
        """
        if copy:
            array = array.copy()
        array = array.astype(np.int32) - np.iinfo(self.dtype).min
        np.clip(array, 0, len(self.bins)-1, out=array)
        array = self.bins[array]
        if self.center:
            array += self.mean
        return array.astype(np.float32)

    def compute_mean_error(self, data):
        """ Estimate quantization error on data. """
        quantized_data = self.quantize(data)
        dequantized_data = self.dequantize(quantized_data)
        return np.mean(np.abs(dequantized_data - data)) / data.std()

    @property
    def estimated_absolute_error(self):
        return np.diff(self.bins).max()
