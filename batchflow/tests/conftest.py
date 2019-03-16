""" Pytest configuration. """
# pylint: disable=invalid-name, unused-import
import pytest

from .config_pass_test import single_config, multi_config, model_and_config
from .config_data_fixtures import (data_dict_init, data_list_init, data_config_init,
							data_pop, data_get, data_put, data_flatten, data_add,
							data_items, data_keys, data_values, data_update)