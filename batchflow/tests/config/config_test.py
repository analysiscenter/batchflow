import sys
import copy
import pytest

sys.path.append('../../..')

from batchflow import Config

def test_dict_init(data_dict_init):
	processed, expected = data_dict_init
	assert expected == processed