# pylint: disable=redefined-outer-name, missing-docstring
import sys
import pytest
import numpy as np

sys.path.append('..')
from batchflow import make_rng, Normalizer, Quantizer


@pytest.fixture
def array():
    return np.arange(0, 100).astype('float32')

@pytest.fixture
def normal_array():
    return make_rng(42).normal(size=1000).astype('float32')

class TestNormalizer:
    def test_mean_normalization(self, array):
        result = Normalizer(mode='mean').normalize(array)
        assert np.isclose(result, array - np.mean(array)).all()

    def test_meanstd_normalization(self, array):
        result = Normalizer(mode='meanstd').normalize(array)
        assert np.isclose(result, (array - np.mean(array)) / np.std(array)).all()

    def test_minmax_normalization(self, array):
        result = Normalizer(mode='minmax').normalize(array)
        assert np.isclose(result, (array - np.min(array)) / np.ptp(array)).all()

    def test_callable(self, array):
        def func(x, _):
            return (x - np.mean(x)) / np.std(x)
        result = Normalizer(mode=func).normalize(array)
        assert np.isclose(result, func(array, None)).all()

    def test_clipping(self, array):
        result = Normalizer(mode='meanstd', clip_to_quantiles=True).normalize(array)
        q = np.quantile(array, (0.01, 0.99))
        target = np.clip(array, *q)
        target = (target - np.mean(target)) / np.std(target)
        assert np.isclose(result, target).all()

    @pytest.mark.parametrize('mode', ['min', 'minmax', 'meanstd'])
    def test_denormalize(self, array, mode):
        normalizer = Normalizer(mode=mode)
        result, stats = normalizer.normalize(array, return_stats=True)
        result = normalizer.denormalize(result, normalization_stats=stats)
        assert np.isclose(array, result, atol=1e-5).all()

    @pytest.mark.parametrize('mode', ['min', 'minmax', 'meanstd'])
    def test_denormalize_with_clipping(self, array, mode):
        normalizer = Normalizer(mode=mode, clip_to_quantiles=True)
        result, stats = normalizer.normalize(array, return_stats=True)
        result = normalizer.denormalize(result, normalization_stats=stats)

        q = np.quantile(array, (0.01, 0.99))
        target = np.clip(array, *q)

        assert np.isclose(result, target, atol=1e-5).all()

    def test_outer_stats(self, array):
        stats = {'mean': 20, 'std': 3}

        normalizer = Normalizer(mode='meanstd', normalization_stats=stats)
        result = normalizer.normalize(array)

        assert np.isclose(result, (array - 20) / 3, atol=1e-5).all()

    def test_outer_stats_denormalize(self, array):
        stats = {'quantiles': (5, 95), 'q': (0.05, 0.95)}

        normalizer = Normalizer(mode='minmax', normalization_stats=stats, clip_to_quantiles=True)
        result = normalizer.normalize(array)
        result = normalizer.denormalize(result)

        target = np.clip(array, 5, 95)

        assert np.isclose(result, target, atol=1e-5).all()

class TestQuantizer:
    def test_quantize(self, normal_array):
        ranges = (np.min(normal_array), np.max(normal_array))
        quantizer = Quantizer(ranges=ranges)
        quantized = quantizer.quantize(normal_array)
        dequantized = ((quantized + 128) / 255) * normal_array.ptp() + normal_array.min()

        assert (np.abs(dequantized - normal_array) < quantizer.estimated_absolute_error).all()

    @pytest.mark.parametrize('dtype', [np.int8, np.int16, np.uint8, np.uint16])
    def test_dequantize(self, normal_array, dtype):
        ranges = (np.min(normal_array), np.max(normal_array))
        quantizer = Quantizer(ranges=ranges, dtype=dtype)
        quantized = quantizer.quantize(normal_array, copy=True)
        dequantized = quantizer.dequantize(quantized, copy=True)

        assert (np.abs(normal_array - dequantized) < quantizer.estimated_absolute_error).all()

    @pytest.mark.parametrize('clip', [False, True])
    @pytest.mark.parametrize('dtype', [np.int8, np.int16, np.uint8, np.uint16])
    def test_ranges(self, normal_array, clip, dtype):
        ranges = np.quantile(normal_array, (0.05, 0.95))
        quantizer = Quantizer(ranges=ranges, clip=clip, dtype=dtype)
        quantized = quantizer.quantize(normal_array, copy=True)
        dequantized = quantizer.dequantize(quantized, copy=True)

        diff = np.abs(normal_array - dequantized)
        central_mask = np.logical_and(normal_array > ranges[0], normal_array < ranges[1])

        assert (diff[central_mask] < quantizer.estimated_absolute_error).all()
        if clip:
            assert set(quantized[~central_mask]) == set([np.iinfo(dtype).min+1, np.iinfo(dtype).max])
        else:
            assert set(quantized[~central_mask]) == set([np.iinfo(dtype).min, np.iinfo(dtype).max])

    @pytest.mark.parametrize('clip', [False, True])
    @pytest.mark.parametrize('dtype', [np.int8, np.int16, np.uint8, np.uint16])
    def test_zero_transform(self, normal_array, clip, dtype):
        ranges = np.quantile(normal_array, (0.05, 0.95))
        ranges = np.abs(ranges).min()
        ranges = (-ranges, ranges)

        quantizer = Quantizer(ranges=ranges, clip=clip, dtype=dtype)

        assert quantizer.quantize([0]) == [np.iinfo(dtype).max - (np.iinfo(dtype).max - np.iinfo(dtype).min) // 2]

    @pytest.mark.parametrize('clip', [False, True])
    @pytest.mark.parametrize('dtype', [np.int8, np.int16, np.uint8, np.uint16])
    def test_outliers_transform(self, normal_array, clip, dtype):
        ranges = normal_array.min(), normal_array.max()

        quantizer = Quantizer(ranges=ranges, clip=clip, dtype=dtype)

        assert (quantizer.quantize([ranges[0]-1, ranges[1]+1]) == [np.iinfo(dtype).min+clip, np.iinfo(dtype).max]).all()
