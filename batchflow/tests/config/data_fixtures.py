"""
Each fixture function generates test data for corresponding test function,
returning specific data set depending of parameter.
"""

import sys
import copy
import pytest

sys.path.append('../../..')

from batchflow import Config

@pytest.fixture(params=[0, 1, 2, 3])
def data_dict_init(request):

	data = [
		dict(
			process = {'a/b' : 1},
			expect = {'a': {'b' : 1}}
			),

		dict(
			process = {'a/b' : 1, 'a/c' : 2},
			expect = {'a' : {'b' : 1, 'c' : 2}}
			),

		dict(
			process = {'a' : {'b' : 1}, 'a/c' : 2},
			expect = {'a' : {'b' : 1, 'c' : 2}}
			),

		dict(
			process = {'a' : Config({'b' : 2})},
			expect = {'a' : {'b' : 2}}
			)
		]

	return data[request.param].get('process'), data[request.param].get('expect')