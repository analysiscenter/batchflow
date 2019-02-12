""" Tests for data_d_f functionality for models with multiple inputs. """
# pylint:  disable=import-error, wrong-import-position
# pylint: disable=missing-docstring, redefined-outer-name
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest
import numpy as np
import tensorflow as tf

from ..models.tf.base import TFModel
from batchflow import Pipeline, ImagesBatch, Dataset
from batchflow import B, V
from batchflow.models.tf.layers import conv_block
from ..models.tf.vgg import VGG7

class SingleModel(TFModel):
    pass

@pytest.fixture()
def single_setup():
    """ Pytest fixture that is used to generate fake dataset and model
    config with desired format.

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
    def _single_setup(d_f):
        if d_f == 'channels_last':
            shape_in = (32, 32, 4)
            shape_out = (2, 2, 64)
        elif d_f == 'channels_first':
            shape_in = (4, 32, 32)
            shape_out = (64, 2, 2)

        size = 50
        batch_shape = (size,) + shape_in
        output = (size,) + shape_out
        images_array = np.random.random(batch_shape)
        labels_array = np.random.choice(5, size=output)
        data = images_array, labels_array
        fake_dataset = Dataset(index=size,
                               batch_class=ImagesBatch,
                               preloaded=data)

        model_config = {'inputs': {'images': {'shape': shape_in},
                                   'labels': {'classes': 5, 'shape': shape_out}},
                        'initial_block/inputs': 'images',
                        'head': {'layout': 'a'},
                        'body': {'layout': 'cna cna',
                                 'filters': [6, 64],
                                 'kernel_size': [11, 7],
                                 'strides': [3, 1],
                                 'padding': ['valid']*2},
                        'loss': 'l1'}
        return fake_dataset, model_config
    return _single_setup


@pytest.fixture()
def get_single_pipeline():
    """ Creates instance of Pipeline that is configured to use given model
    with passed parameters.

    Parameters
    ----------
    current_config : dict
        Dictionary with parameters of model.

    model_name : TFModel
        Model class.

    Returns
    -------
    Pipeline
        Test pipeline that consists of initialization of needed variables and
        given model and preparing for training with given config.
    """
    def _get_single_pipeline(current_config):
        test_pipeline = (Pipeline()
                         .init_variable('current_loss')
                         .init_model('dynamic', SingleModel,
                                     'TestModelSingle', current_config)
                         .to_array()
                         .train_model('TestModelSingle',
                                      fetches='loss',
                                      images=B('images'),
                                      labels=B('labels'),
                                      save_to=V('current_loss'))
                         )
        return test_pipeline
    return _get_single_pipeline


class MultiBatch(ImagesBatch):
    components = 'images_1', 'images_2', 'labels'


class MultiModel(TFModel):
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body/block_1'] = {}
        config['body/block_2'] = {}
        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        kwargs = cls.fill_params('body', **kwargs)
        with tf.variable_scope(name):
            input_1, input_2 = inputs
            x_1 = cls.block_1(input_1, **kwargs)
            x_2 = cls.block_2(input_2, **kwargs)
            dct = {**{'data_format': 'channels_last'},
                   **cls.fill_params('common', **kwargs),
                   **cls.fill_params('body', **kwargs),
                   **cls.fill_params('body', **kwargs)['block_1']}
            axis = cls.channels_axis(dct['data_format'])
            output = tf.concat([x_1, x_2], axis=axis)
        return output

    @classmethod
    def block_1(cls, input_1, **kwargs):
        kwargs = cls.fill_params('body/block_1', **kwargs)
        kwargs = {**kwargs, **kwargs['block_1']}

        with tf.variable_scope('block_1'):
            x_1 = conv_block(input_1, layout='cna',
                             filters=8, kernel_size=11,
                             strides=3, padding='valid', name='1', **kwargs)
        return x_1

    @classmethod
    def block_2(cls, input_2, **kwargs):
        kwargs = cls.fill_params('body/block_2', **kwargs)
        kwargs = {**kwargs, **kwargs['block_2']}

        with tf.variable_scope('block_2'):
            x_2 = conv_block(input_2, layout='cna',
                             filters=8, kernel_size=11,
                             strides=3, padding='valid', name='2', **kwargs)
        return x_2


@pytest.fixture()
def multi_setup():

    def _multi_setup(d_f):
        if d_f == 'channels_last':
            shape_in = (32, 32, 4)
            shape_out = (8, 8, 16)
        elif d_f == 'channels_first':
            shape_in = (4, 32, 32)
            shape_out = (16, 8, 8)

        size = 50
        batch_shape = (size,) + shape_in
        output = (size,) + shape_out

        images_array_1 = np.random.random(batch_shape)
        images_array_2 = np.random.random(batch_shape)
        labels_array = np.random.choice(5, size=output)
        data = images_array_1, images_array_2, labels_array
        fake_dataset = Dataset(index=size,
                               batch_class=MultiBatch,
                               preloaded=data)

        model_config = {'inputs': {'images_1': {'shape': shape_in},
                                   'images_2': {'shape': shape_in},
                                   'labels': {'classes': 5, 'shape': shape_out}},
                        'initial_block/inputs': ['images_1', 'images_2'],
                        'head': {'layout': 'a'},
                        'loss': 'l1'}
        return fake_dataset, model_config
    return _multi_setup


@pytest.fixture()
def get_multi_pipeline():
    def _get_multi_pipeline(current_config):
        test_pipeline = (Pipeline()
                         .init_variable('current_loss')
                         .init_model('dynamic', MultiModel,
                                     'TestModelMulti', current_config)
                         .to_array(src='images_1', dst='images_1')
                         .to_array(src='images_2', dst='images_2')
                         .train_model('TestModelMulti',
                                      fetches='loss',
                                      images_1=B('images_1'),
                                      images_2=B('images_2'),
                                      labels=B('labels'),
                                      save_to=V('current_loss'))
                         )
        return test_pipeline
    return _get_multi_pipeline


@pytest.fixture()
def model_noinitial_setup():

    def _model_noinitial_setup(d_f):

        if d_f == 'channels_last':
            shape_in = (20, 20, 2)

        elif d_f == 'channels_first':
            shape_in = (2, 20, 20)

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
                        'initial_block/inputs': 'images',
                        'body/block': {'pool_size': 1, 'pool_strides':1},
                        'common': {'padding': 'valid'}}
        return fake_dataset, model_config
    return _model_noinitial_setup

@pytest.fixture()
def get_model_noinitial_pipeline():

    def _get_model_noinitial_pipeline(model_class, current_config):
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
    return _get_model_noinitial_pipeline

@pytest.fixture()
def setup(single_setup, multi_setup, model_noinitial_setup):

    def _setup(mode, d_f, model_class=None):
        if mode == 'single_input':
            return single_setup(d_f)
        if mode == 'multi_input':
            return multi_setup(d_f)
        if mode in ['vgg']:
            return model_noinitial_setup(d_f)

    return _setup


@pytest.fixture
def get_pipeline(get_single_pipeline, get_multi_pipeline, get_model_noinitial_pipeline):

    def _get_pipeline(mode, current_config, model_class=None):
        if mode == 'single_input':
            return get_single_pipeline(current_config)
        if mode == 'multi_input':
            return get_multi_pipeline(current_config)
        if mode == 'vgg':
            return get_model_noinitial_pipeline(VGG7, current_config)

    return _get_pipeline


@pytest.mark.parametrize('mode', ['multi_input', 'vgg', 'single_input'])
class TestPlural:
    def test_last_last_default(self, mode, setup, get_pipeline):
        """ Default value for 'data_format' is 'channels_last'. """
        fake_dataset, config = setup(mode, d_f='channels_last')

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_last_common(self, mode, setup, get_pipeline):
        """ We can explicitly pass 'data_format', it has no effect in this case. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_last'}
        else:
            config['common'].update({'data_format': 'channels_last'})

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_first_root(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config' does nothing. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        config['data_format'] = 'channels_first'
        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_first_common(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config/common' raises ValueError. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_first'}
        else:
            config['common'].update({'data_format': 'channels_first'})

        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)

    def test_last_first_body(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config/body' raises ValueError. """
        fake_dataset, config = setup(mode, d_f='channels_last')

        if 'body' not in config:
            config['body'] = {'data_format': 'channels_first'}
        else:
            config['body'].update({'data_format': 'channels_first'})

        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)

    def test_first_last_default(self, mode, setup, get_pipeline):
        """ Default 'data_format' is 'channels_last', so passing data with
        other format raises ValueError.
        """
        fake_dataset, config = setup(mode, d_f='channels_first')
        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)

    def test_first_first_common(self, mode, setup, get_pipeline):
        """ That is intended way to communicate 'data_format' with model. """
        fake_dataset, config = setup(mode, d_f='channels_first')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_first'}
        else:
            config['common'].update({'data_format': 'channels_first'})
        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7


    def test_first_first_body(self, mode, setup, get_pipeline):
        """ Explicitly passing 'data_format' to 'config/body'. """
        fake_dataset, config = setup(mode, d_f='channels_first')

        if 'body' not in config:
            config['body'] = {'data_format': 'channels_first'}
        else:
            config['body'].update({'data_format': 'channels_first'})

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7


def test_first_first_blocks(setup, get_pipeline):
    """ 'data_format' can be passed to a particular block in body,
    thgough it should be explicitly stated in model definition.
    """
    fake_dataset, config = setup('multi_input', d_f='channels_first')

    config['body/block_1'] = {'data_format': 'channels_first'}
    config['body/block_2'] = {'data_format': 'channels_first'}

    test_pipeline = get_pipeline('multi_input', current_config=config)
    total_pipeline = test_pipeline << fake_dataset
    n_b = total_pipeline.next_batch(7, n_epochs=None)
    assert len(n_b.index) == 7

def test_first_cross_blocks(setup, get_pipeline):
    """ Explicitly passing wrong 'data_format' raises ValueError. """
    fake_dataset, config = setup('multi_input', d_f='channels_first')

    config['body/block_1'] = {'data_format': 'channels_last'}
    config['body/block_2'] = {'data_format': 'channels_first'}
    try:
        test_pipeline = get_pipeline('multi_input', current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        total_pipeline.next_batch(7, n_epochs=None)
        pytest.fail("Should not have worked")
    except ValueError as excinfo:
        assert 'Negative dimension size' in str(excinfo)