""" Test for passing information through config. """
# pylint: disable=import-error, wrong-import-position, no-name-in-module
# pylint: disable=missing-docstring, redefined-outer-name
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import tensorflow as tf
from batchflow.models.tf import TFModel



LOCATIONS = set(['initial_block', 'body', 'block', 'head'])



@pytest.fixture()
def single_config():
    """ Fixture that returns the simplest config for single-input model. """

    class SingleModel(TFModel):
        test_container = {}

        @classmethod
        def default_config(cls):
            config = super().default_config()
            config['body/block'] = {}
            return config

        @classmethod
        def initial_block(cls, inputs, name='initial_block', **kwargs):
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_initial_block'] = kwargs
            return inputs

        @classmethod
        def body(cls, inputs, name='body', **kwargs):
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_body'] = kwargs

            block_args = cls.pop('block', kwargs)
            block_args = {**kwargs, **block_args}
            inputs = cls.block(inputs, name='block', **block_args)
            return inputs

        @classmethod
        def block(cls, inputs, **kwargs):
            kwargs = cls.fill_params('body/block', **kwargs)
            cls.test_container['test_block'] = kwargs
            return inputs

        @classmethod
        def head(cls, inputs, name='head', **kwargs):
            inputs = super().head(inputs, **kwargs)
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_head'] = kwargs
            return inputs

    config = {'inputs': {'images': {'shape': (10, 10, 3)},
                         'labels': {'classes': 2}},
              'initial_block/inputs': 'images',
              'head': {'layout': 'f', 'units': 2},
              'loss': 'ce'}

    return SingleModel, config


@pytest.fixture()
def multi_config():
    """ Fixture that returnst the simplest config for multi-input model. """

    class MultiModel(TFModel):
        test_container = {}

        @classmethod
        def default_config(cls):
            config = TFModel.default_config()
            config['body/block'] = {}
            config['body/branch'] = {}
            return config

        @classmethod
        def initial_block(cls, inputs, name='initial_block', **kwargs):
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_initial_block'] = kwargs
            return inputs

        @classmethod
        def body(cls, inputs, name='body', **kwargs):
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_body'] = kwargs

            block_args = cls.pop('block', kwargs)
            block_args = {**kwargs, **block_args}
            branch_args = cls.pop('branch', kwargs)
            branch_args = {**kwargs, **branch_args}
            with tf.variable_scope(name):
                input_1, input_2 = inputs
                x_1 = cls.block(input_1, **block_args)
                x_2 = cls.branch(input_2, **branch_args)
                output = tf.add(x_1, x_2)
            return output

        @classmethod
        def block(cls, input_1, **kwargs):
            kwargs = cls.fill_params('body/block', **kwargs)
            cls.test_container['test_block'] = kwargs
            return input_1

        @classmethod
        def branch(cls, input_2, **kwargs):
            kwargs = cls.fill_params('body/branch', **kwargs)
            cls.test_container['test_branch'] = kwargs
            return input_2

        @classmethod
        def head(cls, inputs, name='head', **kwargs):
            inputs = super().head(inputs, name='head', **kwargs)
            kwargs = cls.fill_params(name, **kwargs)
            cls.test_container['test_head'] = kwargs
            return inputs

    config = {'inputs': {'images_1': {'shape': (10, 10, 3)},
                         'images_2': {'shape': (10, 10, 3)},
                         'labels': {'classes': 2}},
              'initial_block/inputs': ['images_1', 'images_2'],
              'head': {'layout': 'f', 'units': 2},
              'loss': 'ce'}

    return MultiModel, config


@pytest.fixture()
def model_and_config(request):
    """ Fixture to get values of 'single_config' or 'multi_config'. """
    return request.getfixturevalue(request.param)



@pytest.mark.parametrize('model_and_config',
                         ['single_config', 'multi_config',],
                         indirect=['model_and_config'])
class Test_config_pass():
    """ Tests to show correct flow of parameters passed to 'config' to
    actual parts of the model.

    There is a following pattern in every test:
        First of all, we get model class and its 'config' via 'model_and_config' argument.
        Then 'config' is modified by adding or changing some of the keys and values. Usually,
        modifications are done only at part, specified by 'location' parameter.
        At last, we check that our modifications are actually in place by observing
        contents of 'test_container'.
        """
    @pytest.mark.parametrize('location', LOCATIONS)
    def test_common(self, location, model_and_config):
        """ Easiest way to pass a key to all of the parts of the model is to
        pass it to 'common'.
        """
        model, config = model_and_config
        config['common/common_key'] = 'common_key_modified'
        container = model(config).test_container
        assert container['test_' + location]['common_key'] == 'common_key_modified'

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc(self, location, model_and_config):
        """ It is possible to directly send parameters to desired location.
        Those keys will not appear in other places.
        """
        model, config = model_and_config
        destination = location + '/' + location + '_key'
        value = location + '_value_modified'
        config[destination] = value
        container = model(config).test_container
        assert container['test_' + location][location + '_key'] == value # check that key is delivered
        for loc in LOCATIONS - set([location, 'block']):
            assert value not in container['test_' + loc].values() # check that key is not present in other places

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc_priority(self, location, model_and_config):
        """ Parameters, passed to a certain location, take priority over 'common' keys. """
        model, config = model_and_config
        config['common'] = {location + '_key': 'wrong_value'}
        destination = location + '/' + location + '_key'
        value = location + '_value_modified'
        config[destination] = value
        container = model(config).test_container
        assert container['test_' + location][location + '_key'] == value

    def test_block(self, model_and_config):
        """ If model structure is nested (e.g. block inside body), inner
        parts inherit properties from outer (e.g. block has all of the keys
        passed to body).
        """
        model, config = model_and_config
        config['body/body_key'] = 'body_key_modified'
        config['body/block/block_key'] = 'block_value_modified'
        container = model(config).test_container
        assert container['test_block']['body_key'] == 'body_key_modified'
        assert container['test_block']['block_key'] == 'block_value_modified'

    def test_block_priority(self, model_and_config):
        """ If the same parameter is defined both in outer and inner part,
        the inner takes priority.
        """
        model, config = model_and_config
        config['body/block_key'] = 'wrong_value'
        config['body/block/block_key'] = 'block_value_modified'
        container = model(config).test_container
        assert container['test_block']['block_key'] == 'block_value_modified'
