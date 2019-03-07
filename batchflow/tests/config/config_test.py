"""
Each test function receives test data from corresponding fixture.
Then the verifiable Config method is applied to this data.
Finally, formed output is compared with expected one via assert.
"""

import sys
import copy
import pytest

sys.path.append('../../..')

from batchflow import Config

def test_dict_init(data_dict_init):
	processed = Config(data_dict_init[0])
	expected = Config(data_dict_init[1])
	assert expected.config == processed.config