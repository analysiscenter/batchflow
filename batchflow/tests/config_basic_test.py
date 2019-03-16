"""
Each test function receives test data from corresponding fixture.
Then the verifiable Config method is applied to this data.
Finally, formed output is compared with expected one via assert.
"""

import sys

sys.path.append('../..')

from batchflow import Config

def test_dict_init(data_dict_init):
    """
    Tests Config.__init__() using input of dictionary type
    """
    processed_config = Config(data_dict_init.get('process')).config
    expected_config = Config(data_dict_init.get('expect')).config
    assert expected_config == processed_config

def test_list_init(data_list_init):
    """
    Tests Config.__init__() using input of list type
    """
    processed_config = Config(data_list_init.get('process')).config
    expected_config = Config(data_list_init.get('expect')).config
    assert expected_config == processed_config

def test_config_init(data_config_init):
    """
    Tests Config.__init__() using input of Config type
    """
    processed_config = Config(data_config_init.get('process')).config
    expected_config = Config(data_config_init.get('expect')).config
    assert expected_config == processed_config

def test_pop(data_pop):
    """
    Tests Config.pop()
    """
    expected_config = data_pop.get('expect_config').config
    expected_return = data_pop.get('expect_value')
    processed_return = data_pop.get('process').pop(data_pop.get('key'))
    processed_config = data_pop.get('process').config
    assert (expected_return, expected_config) == (processed_return, processed_config)

def test_get(data_get):
    """
    Tests Config.get()
    """
    expected_config = data_get.get('expect_config').config
    expected_return = data_get.get('expect_value')
    processed_return = data_get.get('process').get(data_get.get('key'))
    processed_config = data_get.get('process').config
    assert (expected_return, expected_config) == (processed_return, processed_config)

def test_put(data_put):
    """
    Tests Config.put()
    """
    expected_config = data_put.get('expect').config
    data_put.get('process').put(data_put.get('key'), data_put.get('value'))
    processed_config = data_put.get('process').config
    assert expected_config == processed_config

def test_flatten(data_flatten):
    """
    Tests Config.flatten()
    """
    expected_config = data_flatten.get('expect_config').config
    expected_return = data_flatten.get('expect_return')
    processed_return = data_flatten.get('process').flatten()
    processed_config = data_flatten.get('process').config
    assert (expected_return, expected_config) == (processed_return, processed_config)

def test_add(data_add):
    """
    Tests Config.add()
    """
    expected_configs = (data_add.get('expect_left').config, data_add.get('expect_right').config)
    expected_return = data_add.get('expect_return').config
    processed_return = (data_add.get('process_left') + data_add.get('process_right')).config
    processed_configs = (data_add.get('process_left').config, data_add.get('process_right').config)
    assert (expected_return, expected_configs) == (processed_return, processed_configs)

def test_items(data_items):
    """
    Tests Config.items()
    """
    expected_config = data_items.get('expect_config').config
    expected_returns = (data_items.get('expect_full'), data_items.get('expect_flat'))
    processed_returns = (list(data_items.get('process').items(flatten=False)),
                         list(data_items.get('process').items(flatten=True)))
    processed_config = data_items.get('process').config
    assert (expected_returns, expected_config) == (processed_returns, processed_config)

def test_keys(data_keys):
    """
    Tests Config.keys()
    """
    expected_config = data_keys.get('expect_config').config
    expected_returns = (data_keys.get('expect_full'), data_keys.get('expect_flat'))
    processed_returns = (list(data_keys.get('process').keys(flatten=False)),
                         list(data_keys.get('process').keys(flatten=True)))
    processed_config = data_keys.get('process').config
    assert (expected_returns, expected_config) == (processed_returns, processed_config)

def test_values(data_values):
    """
    Tests Config.values()
    """
    expected_config = data_values.get('expect_config').config
    expected_returns = (data_values.get('expect_full'), data_values.get('expect_flat'))
    processed_returns = (list(data_values.get('process').values(flatten=False)),
                         list(data_values.get('process').values(flatten=True)))
    processed_config = data_values.get('process').config
    assert (expected_returns, expected_config) == (processed_returns, processed_config)

def test_update(data_update):
    """
    Tests Config.update()
    """
    expected_configs = (data_update.get('expect_left').config, data_update.get('expect_right').config)
    data_update.get('process_left').update(data_update.get('process_right'))
    processed_configs = (data_update.get('process_left').config, data_update.get('process_right').config)
    assert expected_configs == processed_configs
