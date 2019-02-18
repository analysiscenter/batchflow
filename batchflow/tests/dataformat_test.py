""" Test for passing information through config. """
# pylint: disable=import-error, wrong-import-position, no-name-in-module
# pylint: disable=missing-docstring, redefined-outer-name
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import numpy as np
import tensorflow as tf

from batchflow import Pipeline, ImagesBatch, Dataset
from batchflow import B, V
from batchflow.models.tf import VGG7, ResNet18, Inception_v1


AVAILABLE_MODELS = [VGG7, ResNet18, Inception_v1]
LOCATIONS = set(['initial_block', 'body', 'block', 'head'])


@pytest.fixture()
def model_setup():
    """ Pytest fixture to generate fake dataset and model config with desired format.

    Parameters
    ----------
    d_f: {'channels_last', 'channels_first'}
        desired format of returns.

    Returns
    -------
    tuple
        First element is instance of Dataset.
        Second element is dict with model description.
    """
    def _model_setup(d_f):
        if d_f == 'channels_last':
            shape_in = (100, 100, 2)
        elif d_f == 'channels_first':
            shape_in = (2, 100, 100)

        size = 50
        batch_shape = (size,) + shape_in
        images_array = np.random.random(batch_shape)
        labels_array = np.random.choice(10, size=size)
        data = images_array, labels_array
        fake_dataset = Dataset(index=size,
                               batch_class=ImagesBatch,
                               preloaded=data)

        model_config = {'inputs': {'images': {'shape': shape_in},
                                   'labels': {'classes': 10}},
                        'initial_block/inputs': 'images'}
        return fake_dataset, model_config

    return _model_setup

@pytest.fixture()
def get_model_pipeline():
    """ Creates instance of Pipeline that is configured to use given model
    with passed parameters.

    Parameters
    ----------

    model_class : subclass of TFModel
        Architecture of model. List of available models is defined at 'AVAILABLE_MODELS'.

    current_config : dict
        Dictionary with parameters of model.

    Returns
    -------
    Pipeline
        Test pipeline that consists of initialization of model and
        preparing for training with given config.
    """
    def _get_model_pipeline(model_class, current_config):
        test_pipeline = (Pipeline()
                         .init_variable('current_loss')
                         .init_model('dynamic', model_class,
                                     'TestModel', current_config)
                         .to_array()
                         .train_model('TestModel',
                                      fetches='loss',
                                      images=B('images'),
                                      labels=B('labels'),
                                      save_to=V('current_loss'))
                         )
        return test_pipeline
    return _get_model_pipeline



@pytest.mark.parametrize('model_and_config',
                         ['single_config', 'multi_config',],
                         indirect=['model_and_config'])
class Test_dataformat():
    """ This class holds tests that are checking ability to pass
    'data_format' to different places in model.

    There is a following pattern in every test:
        First of all, we get class and 'config' of our model via 'model_and_config'
        fixture.
        Then we optionally modify 'config'. In most of them only 'location' is changed.
        Finally, we assert that our modification was actually communicated to desired place.
    """
    @pytest.mark.xfail(run=True)
    @pytest.mark.parametrize('location', LOCATIONS)
    def test_default_dataformat(self, location, model_and_config):
        """ Default value for 'data_format' is 'channels_last'. """
        model, config = model_and_config
        container = model(config).test_container
        assert container['test_' + location]['data_format'] == 'channels_last'

    @pytest.mark.parametrize('location', LOCATIONS)
    def test_common_dataformat(self, location, model_and_config):
        """ Easiest way to change 'data_format' for every part of network is to
        pass it to 'common'.
        """
        model, config = model_and_config
        if not config.get('common'):
            config['common'] = {'data_format': 'channels_first'}
        else:
            config['common'].update({'data_format': 'channels_first'})
        container = model(config).test_container
        assert container['test_' + location]['data_format'] == 'channels_first'

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc_dataformat(self, location, model_and_config):
        """ 'data_format' can be passed directly to desired location. """
        model, config = model_and_config
        if not config.get(location):
            config[location] = {'data_format': 'channels_first'}
        else:
            config[location].update({'data_format': 'channels_first'})
        container = model(config).test_container
        assert container['test_' + location]['data_format'] == 'channels_first'
        for loc in LOCATIONS - set([location, 'block']):
            assert 'channels_first' not in container['test_' + loc].values()

    def test_block_dataformat(self, model_and_config):
        """ Parameters, passed to inner parts take priority over outers. """
        model, config = model_and_config
        config['body'] = {'data_format': 'channels_last'}
        config['body/block'] = {'data_format': 'channels_first'}
        container = model(config).test_container
        assert container['test_block']['data_format'] == 'channels_first'



@pytest.mark.parametrize('model', AVAILABLE_MODELS)
class Test_models:
    """ Tests in this class show that we can train model with given 'data_format.

    There is a following pattern in every test:
        First of all, we get 'fake_data' and 'config' via 'model_setup' fixture.
        Then we optionally modify 'config'. In most cases it is done only at 'location'.
        Finally, we assert that our modification was actually applied to model by attempting
        to train it on a small batch.

    """
    def test_last_default(self, model, model_setup, get_model_pipeline):
        """ Default value for 'data_format' is 'channels_last'. """
        fake_dataset, config = model_setup(d_f='channels_last')

        test_pipeline = get_model_pipeline(model, config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_common(self, model, model_setup, get_model_pipeline):
        """ We can explicitly pass 'data_format', it has no effect in this case. """
        fake_dataset, config = model_setup(d_f='channels_last')
        config['common'] = {'data_format': 'channels_last'}
        test_pipeline = get_model_pipeline(model, config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_first_common(self, model, model_setup, get_model_pipeline):
        """ That is intended way to communicate 'data_format' with model. """
        if int(tf.__version__.split('.')[1]) < 12:
            pytest.skip('too old to work')
        fake_dataset, config = model_setup(d_f='channels_first')
        config['common'] = {'data_format': 'channels_first'}
        test_pipeline = get_model_pipeline(model, config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7
