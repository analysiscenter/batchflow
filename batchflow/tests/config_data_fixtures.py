"""
Each fixture function generates test data for corresponding test function from config_basic_test.py,
returning specific data set depending of parameter.

All simple keys and values are chosen to be single-letter characters and integers correspondingly,
since nothing in testing process depends on their data type.
"""

import sys
import pytest

sys.path.append('../../..')

from batchflow import Config

@pytest.fixture(params=[0, 1, 2, 3, 4])
def data_dict_init(request):
    """
    Generates data set for test_dict_init consisting of:
        1. «process» — nested dictionary with slashed indexing
        2. «expect» — normal nested dictionary

    After being both passed to Config class constructor contents of obtained class instances are expected to match.
    """
    data = [
        dict(
            process={'a/b' : 1},
            expect={'a': {'b' : 1}}
            ),

        dict(
            process={'a/b' : 1, 'a/c' : 2},
            expect={'a' : {'b' : 1, 'c' : 2}}
            ),

        dict(
            process={'a' : 1, 'a/b' : 2, 'a/c' : 3},
            expect={'a' : {'b' : 2, 'c' : 3}}
            ),

        dict(
            process={'a' : {'b' : 1}, 'a/c' : 2},
            expect={'a' : {'b' : 1, 'c' : 2}}
            ),

        dict(
            process={'a' : Config({'b' : 2})},
            expect={'a' : {'b' : 2}}
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1, 2, 3, 4])
def data_list_init(request):
    """
    Generates data set for test_list_init consisting of:
        1. «process» — list with pairs (key, value)
        2. «expect» — normal nested dictionary

    After being both passed to Config class constructor contents of obtained class instances are expected to match.
    """
    data = [
        dict(
            process=[('a/b', 1)],
            expect={'a' : {'b' : 1}}
            ),

        dict(
            process=[('a/b', 1), ('a/c', 2)],
            expect={'a' : {'b' : 1, 'c' : 2}}
            ),

        dict(
            process=[('a', 1), ('a/b', 2), ('a/c', 3)],
            expect={'a' : {'b' : 2, 'c' : 3}}
            ),

        dict(
            process=[('a', {'b' : 1}), ('a/c', 2)],
            expect={'a' : {'b' : 1, 'c' : 2}}
            ),

        dict(
            process=[('a', Config([('b', 2)]))],
            expect={'a' : {'b' : 2}}
            )
        ]

    return data[request.param]

# No idea 'bout good tests for this kind of initialization.
@pytest.fixture(params=[0])
def data_config_init(request):
    """
    Generates data set for test_config_init consisting of:
        1. «process» — instance of Config
        2. «expect» — normal dictionary

    After being both passed to Config class constructor contents of obtained class instances are expected to match.
    """
    data = [
        dict(
            process=Config({'a' : 0}),
            expect={'a' : 0}
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1])
def data_pop(request):
    """
    Generates data set for test_pop consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «key» — string key value present in «process» — either single-character or with nested indexing
        3. «expect_value» — either integer or dictionary value corresponding to the «key»
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1, 'c' : 2}}),
            key='a/b',
            expect_value=1,
            expect_config=Config({'a' : {'c' : 2}})
            ),

        dict(
            process=Config({'a' : {'b' : {'c' : 1, 'd' : 2}}}),
            key='a/b',
            expect_value={'c' : 1, 'd' : 2},
            expect_config=Config({'a' : {}})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1])
def data_get(request):
    """
    Generates data set for test_get consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «key» — string key value present in «process» — either single-character or with nested indexing
        3. «expect_value» — either integer or dictionary value corresponding to the «key»
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1, 'c' : 2}}),
            key='a/b',
            expect_value=1,
            expect_config=Config({'a' : {'b' : 1, 'c' : 2}})
            ),

        dict(
            process=Config({'a' : {'b' : {'c' : 1}}}),
            key='a/b',
            expect_value={'c' : 1},
            expect_config=Config({'a' : {'b' : {'c' : 1}}})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1, 2])
def data_put(request):
    """
    Generates data set for test_put consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «key» — arbitrary string key value — either single-character or with nested indexing
        3. «value» — arbitrary value corresponding to the «key» — either simply integer value or dictionary
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : 1}),
            key='b',
            value=2,
            expect=Config({'a' : 1, 'b' : 2})
            ),

        dict(
            process=Config({'a' : {'b' : 1}}),
            key='a/c',
            value=2,
            expect=Config({'a' : {'b' : 1, 'c' : 2}})
            ),

        dict(
            process=Config({'a' : {}}),
            key='a/b',
            value={'c/d' : 1},
            expect=Config({'a/b/c/d' : 1})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1, 2])
def data_flatten(request):
    """
    Generates data set for test_flatten consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «expect_return» — expected result of applying tested method to «process»
        3. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1}, 'c' : 2}),
            expect_return={'a/b' : 1, 'c' : 2},
            expect_config=Config({'a' : {'b' : 1}, 'c' : 2})
            ),

        dict(
            process=Config({'a' : {'b' : 1, 'c' : {'d' : 2}}}),
            expect_return={'a/b' : 1, 'a/c/d' : 2},
            expect_config=Config({'a' : {'b' : 1, 'c' : {'d' : 2}}})
            ),

        dict(
            process=Config({'a' : {'b' : {}, 'c' : None}}),
            expect_return={'a/b' : dict(), 'a/c' : None},
            expect_config=Config({'a' : {'b' : {}, 'c' : None}})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1, 2])
def data_add(request):
    """
    Generates data set for test_add consisting of:
        1. «process_left» — instance of Config, to which tested method is applied
        2. «process_right» — instance of Config, which acts as tested method argument
        3. «expect_return» — expected result of applying tested method to «process_left» with argument «process_right»
        4. «expect_return» — expected result of applying tested method to «process_left»
        5. «expect_right» — expected result of using «process_right» as an argument in tested method
    """
    data = [
        dict(
            process_left=Config({'a' : 1, 'b' : 2}),
            process_right=Config({'b' : 3, 'c' : 4}),
            expect_return=Config({'a' : 1, 'b' : 3, 'c' : 4}),
            expect_left=Config({'a' : 1, 'b' : 2}),
            expect_right=Config({'b' : 3, 'c' : 4})
            ),

        dict(
            process_left=Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
            process_right=Config({'a/b/c/d/e' : 4, 'a/b/c/d/f' : 5}),
            expect_return=Config({'a' : {'b' : {'c' : {'d' : {'e' : 4, 'f' : 5}}, 'd' : 2}}, 'b' : 3}),
            expect_left=Config({'a' : {'b' : {'c' : 1, 'd' : 2}}, 'b' : 3}),
            expect_right=Config({'a/b/c/d/e' : 4, 'a/b/c/d/f' : 5})
            ),

        dict(
            process_left=Config({'a' : {'b' : 1, 'c' : 2}, 'b' : {'d' : dict()}}),
            process_right=Config({'a' : dict(), 'b' : None}),
            expect_return=Config({'a' : {'b' : 1, 'c' : 2}, 'b' : None}),
            expect_left=Config({'a' : {'b' : 1, 'c' : 2}, 'b' : {'d' : dict()}}),
            expect_right=Config({'a' : dict(), 'b' : None})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1])
def data_items(request):
    """
    Generates data set for test_items consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «expect_full» — list of «process» items expected as a return of tested method with flatten=False
        3. «expect_flat» — list of «process» items expected as a return of tested method with flatten=False
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1, 'c' : 2}}),
            expect_full=[('a', {'b': 1, 'c': 2})],
            expect_flat=[('a/b', 1), ('a/c', 2)],
            expect_config=Config({'a' : {'b' : 1, 'c' : 2}})
            ),

        dict(
            process=Config({'a' : {'b' : None}}),
            expect_full=[('a', {'b' : None})],
            expect_flat=[('a/b', None)],
            expect_config=Config({'a' : {'b' : None}})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1])
def data_keys(request):
    """
    Generates data set for test_keys consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «expect_full» — list of «process» keys expected as a return of tested method with flatten=False
        3. «expect_flat» — list of «process» keys expected as a return of tested method with flatten=False
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1, 'c' : 2}}),
            expect_full=['a'],
            expect_flat=['a/b', 'a/c'],
            expect_config=Config({'a' : {'b' : 1, 'c' : 2}})
            ),

        dict(
            process=Config({'a' : {'b' : None}, 'c' : 1}),
            expect_full=['a', 'c'],
            expect_flat=['a/b', 'c'],
            expect_config=Config({'a' : {'b' : None}, 'c' : 1})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1])
def data_values(request):
    """
    Generates data set for test_values consisting of:
        1. «process» — instance of Config, to which tested method is applied
        2. «expect_full» — list of «process» values expected as a return of tested method with flatten=False
        3. «expect_flat» — list of «process» values expected as a return of tested method with flatten=False
        4. «expect_config» — instance of Config expected to match «process» by the end of the test
    """
    data = [
        dict(
            process=Config({'a' : {'b' : 1, 'c' : 2}}),
            expect_full=[{'b': 1, 'c': 2}],
            expect_flat=[1, 2],
            expect_config=Config({'a' : {'b' : 1, 'c' : 2}})
            ),

        dict(
            process=Config({'a' : {'b' : None}, 'c' : 1}),
            expect_full=[{'b' : None}, 1],
            expect_flat=[None, 1],
            expect_config=Config({'a' : {'b' : None}, 'c' : 1})
            )
        ]

    return data[request.param]

@pytest.fixture(params=[0, 1, 2, 3])
def data_update(request):
    """
    Generates data set for test_update consisting of:
        1. «process_left» — instance of Config, to which tested method is applied
        2. «process_right» — instance of Config, which acts as tested method argument
        3. «expect_return» — expected result of applying tested method to «process_left»
        4. «expect_right» — expected result of using «process_right» as an argument in tested method
    """
    data = [
        dict(
            process_left=Config({'a' : {'b' : 1, 'c' : 2}}),
            process_right=Config({'a/b' : 3, 'a/d' : 4}),
            expect_left=Config({'a' : {'b' : 3, 'c' : 2, 'd' : 4}}),
            expect_right=Config({'a/b' : 3, 'a/d' : 4})
            ),

        dict(
            process_left=Config({'a' : {'b' : 1}}),
            process_right=Config({'a' : {'c' : {'d' : 2}}}),
            expect_left=Config({'a' : {'b' : 1, 'c' : {'d' : 2}}}),
            expect_right=Config({'a' : {'c' : {'d' : 2}}})
            ),

        dict(
            process_left=Config({'a' : {}, 'b' : None}),
            process_right=Config({'a' : None, 'b' : {}}),
            expect_left=Config({'a' : None, 'b' : {}}),
            expect_right=Config({'a' : None, 'b' : {}})
            ),

        dict(
            process_left=Config({'a' : {'b' : 1, 'c' : 2}, 'b' : {'d' : dict()}}),
            process_right=Config({'a' : dict(), 'b' : 0}),
            expect_left=Config({'a' : {'b' : 1, 'c' : 2}, 'b' : 0}),
            expect_right=Config({'a' : dict(), 'b' : 0})
            )
        ]

    return data[request.param]
