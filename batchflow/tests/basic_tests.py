import sys
import pytest
sys.path.append('../..')
from batchflow import Config

@pytest.mark.parametrize('variation', [0,1,2])
def test_dict_init(variation):

    processed_inputs = [{'a' : 1, 'b' : 2, 'a' : 3, 'b' : 4},
                        {'a' : 1, 'b/a' : 2, 'b' : 3, 'a/b' : 4},
                        {'a' : {'b' : 1}, 'a' : 2}]
                
    expected_inputs = [{'a': 3, 'b' : 4},
                       {'a/b' : 4, 'b' : 3},
                       {'a' : 2}]
    
    processed_config = Config(processed_inputs[variation])
    expected_config = Config(expected_inputs[variation])
                             
    assert expected_config.config == processed_config.config

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_list_init(variation):

    processed_inputs = [[('a', 1),('b', 2),('a', 3),('b', 4)],
                        [('a', 1), ('b/a', 2), ('b', 3), ('a/b', 4)],
                        [('a/b', 1), ('a/b', 2), ('c', 3)]]
                
    expected_inputs = [{'a': 3, 'b' : 4},
                       {'a/b' : 4, 'b' : 3},
                       {'a/b' : 2, 'c' : 3}]
    
    processed_config = Config(processed_inputs[variation])
    expected_config = Config(expected_inputs[variation])
                             
    assert expected_config.config == processed_config.config

@pytest.mark.parametrize('variation', [0])
def test_config_init(variation):

    inputs = [Config({'a' : 0, 'b' : 1})]
    
    processed_config = Config(inputs[variation])
    expected_config = inputs[variation]
                             
    assert expected_config.config == processed_config.config

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_pop(variation):

    inputs = [Config({'a' : 0, 'b' : 1}),
              Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
              Config({'a' : {}, 'b' : 0})]
    # "expects" entry should include a triplet of expected Config, pop key and its value
    expects = [(Config({'b' : 1}), 'a', 0),
               (Config({'a' : {'b' : {'d' : 2}}, 'b' : 3}), 'a/b/c', 1),
               (Config({'b' : 0}), 'a', dict())]
    processed_return = inputs[variation].pop(expects[variation][1])
    expected_return = expects[variation][2]
    processed_config = inputs[variation]
    expected_config = expects[variation][0]
                             
    assert expected_config.config == processed_config.config
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_get(variation):

    inputs = [Config({'a' : 0, 'b' : 1}),
              Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
              Config({'a' : {}, 'b' : 0})]
    # "expects" entry should include a triplet of expected Config, get key and its value
    expects = [(Config({'a' : 0, 'b' : 1}), 'a', 0),
              (Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}), 'a', {'b' : {'c' : 1, 'd' : 2}}),
              (Config({'a' : {}, 'b' : 0}), 'a', dict())] #todo: implement copy()
    processed_return = inputs[variation].get(expects[variation][1])
    expected_return = expects[variation][2]
    processed_config = inputs[variation]
    expected_config = expects[variation][0]
                             
    assert expected_config.config == processed_config.config
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_put(variation):

    inputs = [Config({'a' : 0, 'b' : 1}),
              Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
              Config({'a' : {}, 'b' : 0})]
    # "expects" entry should include a triplet of expected Config, put key and its value
    expects = [(Config({'a' : 0, 'b' : 1, 'c' : 2}), 'c', 2),
              (Config({'a' : {'b' : {'c' : 1, 'd' : 2, 'e' : {'f' : 0}}}, 'b' : 3}), 'a/b/e/f', 0),
              (Config({'a/b/c/d/e' : 1, 'b' : 0}), 'a/b/c', {'d/e' : 1})]
    processed_return = inputs[variation].put(expects[variation][1], expects[variation][2])
    expected_return = None
    processed_config = inputs[variation]
    expected_config = expects[variation][0]
                             
    assert expected_config.config == processed_config.config
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_flatten(variation):

    inputs = [Config({'a' : {'b' : 1, 'c' : 2}, 'b' : 3}),
              Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
              Config({'a' : {}, 'b' : 0})]
    expects = [{'a/b' : 1, 'a/c' : 2, 'b' : 3},
               {'a/b/c' : 1, 'a/b/d' : 2, 'b' : 3},
               {'a' : dict(), 'b' : 0}]
    
    expected_config = Config(inputs[variation].config)
    processed_return = inputs[variation].flatten()
    expected_return = expects[variation]
    processed_config = inputs[variation]
                             
    assert expected_config.config == processed_config.config
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2]) # 2 FAILED
def test_add(variation):

    inputs = [(Config({'a' : 1, 'b' : 2}), Config({'b' : 3, 'c' : 4})),
              (Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}), Config({'a/b/c/d/e' : 4, 'a/b/c/d/f' : 5})),
              (Config({'a' : {'b' : 1, 'c' : 2}, 'b' : {'d' : dict()}}), Config({'a' : dict(), 'b' : 0}))]
    expects = [Config({'a' : 1, 'b' : 3, 'c' : 4}),
               Config({'a' : {'b' : {'c' : {'d' : {'e' : 4, 'f' : 5}}, 'd' : 2}}, 'b' : 3}),
               Config({'a' : {'b' : 1, 'c' : 2}, 'b' : 0})]
    processed_return = inputs[variation][0] + inputs[variation][1]
    expected_return = expects[variation]
    
    assert expected_return.config == processed_return.config

@pytest.mark.parametrize('variation', [0, 1, 2]) # 2 FAILED
def test_items(variation):

    inputs = [Config({'a':{'b':1, 'c':2}}),
              Config({'a' : {}, 'b' : None}),
              Config({'a' : {'b' : 1}, 'a' : {'c' : {'d' : 2}}})]
    expects = [([('a', {'b': 1, 'c': 2})], [('a/b', 1), ('a/c', 2)]),
               ([('a', {}),('b', None)], [('a', {}),('b', None)]),
               ([('a', {'c' : {'d' : 2}})], [('a/c/d', 2)])]
    processed_return = (list(inputs[variation].items(False)), list(inputs[variation].items(True)))
    expected_return = expects[variation]
    
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_keys(variation):

    inputs = [Config({'a':{'b':1, 'c':2}}),
              Config({'a' : {}, 'b' : None}),
              Config({'a' : {'b' : 1}}) + Config({'a' : {'c' : {'d' : 2}}})]
    expects = [(['a'], ['a/b', 'a/c']),
               (['a', 'b'], ['a', 'b']),
               (['a'], ['a/b', 'a/c/d'])]
    processed_return = (list(inputs[variation].keys(False)), list(inputs[variation].keys(True)))
    expected_return = expects[variation]
    
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2])
def test_values(variation):

    inputs = [Config({'a':{'b':1, 'c':2}}),
              Config({'a' : {}, 'b' : None}),
              Config({'a' : {'b' : 1}}) + Config({'a' : {'c' : {'d' : 2}}})]
    expects = [([{'b': 1, 'c': 2}], [1, 2]),
               ([{}, None], [{}, None]),
               ([{'b' : 1, 'c' : {'d' : 2}}], [1, 2])]
    processed_return = (list(inputs[variation].values(False)), list(inputs[variation].values(True)))
    expected_return = expects[variation]
    
    assert expected_return == processed_return

@pytest.mark.parametrize('variation', [0, 1, 2, 3])
def test_update(variation):

    inputs = [(Config({'a':{'b':1, 'c':2}}), Config({'a/b' : 3, 'a/c' : 4})),
              (Config({'a' : {'b' : 1}}), Config({'a' : {'c' : {'d' : 2}}})),
              (Config({'a' : {}, 'b' : None}), Config({'a' : None, 'b' : {}})),
              (Config({'a' : {'b' : 1, 'c' : 2}, 'b' : {'d' : dict()}}), Config({'a' : dict(), 'b' : 0}))]
    expects = [Config({'a/b' : 3, 'a/c' : 4}),
               Config({'a' : {'b' : 1, 'c' : {'d' : 2}}}),
               Config({'a' : None, 'b' : {}}),
               Config({'a' : {'b' : 1, 'c' : 2}, 'b' : 0})]
    inputs[variation][0].update(inputs[variation][1])
    
    assert expects[variation].config == inputs[variation][0].config
