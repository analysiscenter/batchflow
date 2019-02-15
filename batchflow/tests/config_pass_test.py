""" Test for passing information through config. """
# pylint: disable=import-error, wrong-import-position
# pylint: disable=missing-docstring, redefined-outer-name
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import tensorflow as tf
from ...models.tf.base import TFModel



LOCATIONS = set(['initial_block', 'body', 'block', 'head'])



@pytest.fixture()
def single_config():
    """ Fixture that returns the simplest config for single-input model. """
    config = {'inputs': {'images': {'shape': (10, 10, 3)},
                         'labels': {'classes': 2}},
              'initial_block/inputs': 'images',
              'head': {'layout': 'f', 'units': 2},
              'loss': 'ce'}
    return config


@pytest.fixture()
def multi_config():
    """ Fixture that returnst the simplest config for multi-input model. """
    config = {'inputs': {'images_1': {'shape': (10, 10, 3)},
                         'images_2': {'shape': (10, 10, 3)},
                         'labels': {'classes': 2}},
              'initial_block/inputs': ['images_1', 'images_2'],
              'head': {'layout': 'f', 'units': 2},
              'loss': 'ce'}
    return config


@pytest.fixture()
def model_and_config(single_config, multi_config):
    """ Fixture to choose between different types of models.

    It is necessary to define model classes inside this fixture for them to
    be created anew for every test (and for every combination of parameters).

    Parameters
    ----------

    mode : {'single', 'multi'}
        type of returned model and config.

    Returns
    -------
    tuple
        First element is subclass of TFModel.
        Second element is dict with model configuration.
    """

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


    def _model_and_config(mode):
        if mode == 'single':
            return SingleModel, single_config
        if mode == 'multi':
            return MultiModel, multi_config
        raise ValueError('mode must be either single or multi')

    return _model_and_config



@pytest.mark.parametrize('model_type', ['single', 'multi'])
class Test_plural():
    @pytest.mark.parametrize('location', LOCATIONS)
    def test_common(self, model_type, location, model_and_config):
        """ Easiest way to pass a key to all of the parts of the model is to
        pass it to 'common'.
        """
        model, config = model_and_config(model_type)
        if not config.get('common'):
            config['common'] = {'common_key': 'common_key_modified'}
        else:
            config['common'].update({'common_key': 'common_key_modified'})
        container = model(config).test_container
        assert container['test_' + location]['common_key'] == 'common_key_modified'


    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc(self, model_type, location, model_and_config):
        """ It is possible to directly send parameters to desired location.
        Those keys will not appear in other places.
        """
        model, config = model_and_config(model_type)
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
    def test_loc_priority(self, model_type, location, model_and_config):
        """ Parameters, passed to a certain location, take priority over 'common' keys. """
        model, config = model_and_config(model_type)
        config['common'] = {location + '_key': 'wrong_value'}
        value = location + '_value_modified'
        if not config.get(location):
            config[location] = {location + '_key': value}
        else:
            config[location].update({location + '_key': value})
        container = model(config).test_container
        assert container['test_' + location][location + '_key'] == value

    def test_block(self, model_type, model_and_config):
        """ If model structure is nested (e.g. block inside body), inner
        parts inherit properties from outer (e.g. block has all of the keys
        passed to body).
        """
        model, config = model_and_config(model_type)
        config['body'] = {'body_key': 'body_key_modified'}
        config['body/block'] = {'block_key': 'block_value_modified'}
        container = model(config).test_container
        assert container['test_block']['body_key'] == 'body_key_modified'
        assert container['test_block']['block_key'] == 'block_value_modified'

    def test_block_priority(self, model_type, model_and_config):
        """ If the same parameter is defined both in outer and inner part,
        the inner takes priority.
        """
        model, config = model_and_config(model_type)
        config['body'] = {'block_key': 'wrong_value'}
        config['body/block'] = {'block_key': 'block_value_modified'}
        container = model(config).test_container
        assert container['test_block']['block_key'] == 'block_value_modified'
