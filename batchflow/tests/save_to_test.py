# pylint: disable=missing-docstring
"""
Test save_to function
"""

import sys

import numpy as np

sys.path.append('../..')
from batchflow import Config, Pipeline, C, save_data_to


class TestSaveTo:
    def test_save_to_array(self):

        arr = np.zeros(3)

        save_data_to(data=[1, 2, 3], dst=arr)

        assert isinstance(arr, np.ndarray)
        assert (arr == [1, 2, 3]).all()


    def test_save_to_c(self):
        pipeline = Pipeline(config=Config(some=100))

        save_data_to(data=200, dst=C('value'), pipeline=pipeline)

        assert pipeline.config['some'] == 100
        assert pipeline.config['value'] == 200


    def test_save_to_list(self):
        arr = np.zeros(3)
        pipeline = Pipeline(config=Config(some=100))

        save_data_to(data=[[1, 2, 3], 200], dst=[arr, C('value')], pipeline=pipeline)

        assert (arr == [1, 2, 3]).all()
        assert pipeline.config['value'] == 200
