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
        @classmethod
        def default_config(cls):
            config = super().default_config()
            config['body/block'] = {}
            return config

        def build_config(self, names=None):
            config = super().build_config(names)
            SingleModel.test_container = {}
            return config

        @classmethod
        def initial_block(cls, inputs, name='initial_block', **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_initial_block'] = kwmodel_and_configs
            return inputs

        @classmethod
        def body(cls, inputs, name='body', **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_body'] = kwmodel_and_configs

            block_model_and_configs = cls.pop('block', kwmodel_and_configs)
            block_model_and_configs = {**kwmodel_and_configs, **block_model_and_configs}
            inputs = cls.block(inputs, name='block', **block_model_and_configs)
            return inputs

        @classmethod
        def block(cls, inputs, **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params('body/block', **kwmodel_and_configs)
            cls.test_container['test_block'] = kwmodel_and_configs
            return inputs

        @classmethod
        def head(cls, inputs, name='head', **kwmodel_and_configs):
            inputs = super().head(inputs, **kwmodel_and_configs)
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_head'] = kwmodel_and_configs
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
        @classmethod
        def default_config(cls):
            config = TFModel.default_config()
            config['body/block'] = {}
            config['body/branch'] = {}
            return config

        def build_config(self, names=None):
            config = super().build_config(names)
            MultiModel.test_container = {}
            return config

        @classmethod
        def initial_block(cls, inputs, name='initial_block', **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_initial_block'] = kwmodel_and_configs
            return inputs

        @classmethod
        def body(cls, inputs, name='body', **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_body'] = kwmodel_and_configs

            block_model_and_configs = cls.pop('block', kwmodel_and_configs)
            block_model_and_configs = {**kwmodel_and_configs, **block_model_and_configs}
            branch_model_and_configs = cls.pop('branch', kwmodel_and_configs)
            branch_model_and_configs = {**kwmodel_and_configs, **branch_model_and_configs}
            with tf.variable_scope(name):
                input_1, input_2 = inputs
                x_1 = cls.block(input_1, **block_model_and_configs)
                x_2 = cls.branch(input_2, **branch_model_and_configs)
                output = tf.add(x_1, x_2)
            return output

        @classmethod
        def block(cls, input_1, **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params('body/block', **kwmodel_and_configs)
            cls.test_container['test_block'] = kwmodel_and_configs
            return input_1

        @classmethod
        def branch(cls, input_2, **kwmodel_and_configs):
            kwmodel_and_configs = cls.fill_params('body/branch', **kwmodel_and_configs)
            cls.test_container['test_branch'] = kwmodel_and_configs
            return input_2

        @classmethod
        def head(cls, inputs, name='head', **kwmodel_and_configs):
            inputs = super().head(inputs, name='head', **kwmodel_and_configs)
            kwmodel_and_configs = cls.fill_params(name, **kwmodel_and_configs)
            cls.test_container['test_head'] = kwmodel_and_configs
            return inputs

    config = {'inputs': {'images_1': {'shape': (10, 10, 3)},
                         'images_2': {'shape': (10, 10, 3)},
                         'labels': {'classes': 2}},
              'initial_block/inputs': ['images_1', 'images_2'],
              'head': {'layout': 'f', 'units': 2},
              'loss': 'ce'}

    return MultiModel, config


@pytest.fixture
def model_and_config(request):
    """ Fixture to get values of 'single_config' or 'multi_config'. """
    return request.getfixturevalue(request.param)



@pytest.mark.parametrize('model_and_config',
                         ['single_config', 'multi_config',],
                         indirect=['model_and_config'])
class Test_config_pass():
    @pytest.mark.parametrize('location', LOCATIONS)
    def test_common(self, location, model_and_config):
        """ Easiest way to pass a key to all of the parts of the model is to
        pass it to 'common'.
        """
        model, config = model_and_config
        if not config.get('common'):
            config['common'] = {'common_key': 'common_key_modified'}
        else:
            config['common'].update({'common_key': 'common_key_modified'})
        container = model(config).test_container
        assert container['test_' + location]['common_key'] == 'common_key_modified'

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc(self, location, model_and_config):
        """ It is possible to directly send parameters to desired location.
        Those keys will not appear in other places.
        """
        model, config = model_and_config
        value = location + '_value_modified'
        if not config.get(location):
            config[location] = {location + '_key': value}
        else:
            config[location].update({location + '_key': value})
        container = model(config).test_container
        assert container['test_' + location][location + '_key'] == value # check that key is delivered
        for loc in LOCATIONS - set([location, 'block']):
            assert value not in container['test_' + loc].values() # check that key is not present in other places

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc_priority(self, location, model_and_config):
        """ Parameters, passed to a certain location, take priority over 'common' keys. """
        model, config = model_and_config
        config['common'] = {location + '_key': 'wrong_value'}
        value = location + '_value_modified'
        if not config.get(location):
            config[location] = {location + '_key': value}
        else:
            config[location].update({location + '_key': value})
        container = model(config).test_container
        assert container['test_' + location][location + '_key'] == value

    def test_block(self, model_and_config):
        """ If model structure is nested (e.g. block inside body), inner
        parts inherit properties from outer (e.g. block has all of the keys
        passed to body).
        """
        model, config = model_and_config
        config['body'] = {'body_key': 'body_key_modified'}
        config['body/block'] = {'block_key': 'block_value_modified'}
        container = model(config).test_container
        assert container['test_block']['body_key'] == 'body_key_modified'
        assert container['test_block']['block_key'] == 'block_value_modified'

    def test_block_priority(self, model_and_config):
        """ If the same parameter is defined both in outer and inner part,
        the inner takes priority.
        """
        model, config = model_and_config
        config['body'] = {'block_key': 'wrong_value'}
        config['body/block'] = {'block_key': 'block_value_modified'}
        container = model(config).test_container
        assert container['test_block']['block_key'] == 'block_value_modified'
