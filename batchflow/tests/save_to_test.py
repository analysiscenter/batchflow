"""
Test save_to function
"""

import sys
import pytest

import numpy as np

sys.path.append('../..')
from batchflow import Config, Pipeline, C, save_to


def test_save_to_array():
    arr = np.zeros(3)

    save_to(what=[1, 2, 3], where=arr)

    assert isinstance(arr, np.ndarray)
    assert (arr == [1, 2, 3]).all()


def test_save_to_c():
    pipeline = Pipeline(config=Config(some=100))

    save_to(what=200, where=C('value'), pipeline=pipeline)

    assert pipeline.config['some'] == 100
    assert pipeline.config['value'] == 200


def test_save_to_list():
    arr = np.zeros(3)
    pipeline = Pipeline(config=Config(some=100))

    save_to(what=[[1, 2, 3], 200], where=[arr, C('value')], pipeline=pipeline)

    assert (arr == [1, 2, 3]).all()
    assert pipeline.config['value'] == 200
