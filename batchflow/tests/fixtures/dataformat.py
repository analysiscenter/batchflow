""" Fixtures for 'data_format' tests.
General idea is to create such combination of neural network architecture
and input data, so that passing wrong 'data_format' would inevitably raise ValueError.

For example, 'single_setup' and 'get_single_pipeline' fixtures are creating input data of
32x32x4 size. That array is passed through convolutional layer with 'kernel_size' of 11, 'valid' padding,
and that is possible only in 'channels_last' mode. Switching it to 'channels_first' would result
in an atempt of creating array with negative dimension sinse 4-11 is less than 0.

For even more confidence in that test, we use multiple layers, and each of them follows
the same pattern: if the 'data_format' is wrong, it would raise ValueError.

Last correctness check is performed at the loss evaluation step: we use l1 loss,
which is possible to calculate only for equal-sized arrays. Since we know all the dimensions
and their transforms layer-to-layer, we can use 'ground_truth' labels
with desired dimensions as one of the testing methods.

Model with multiple imputs is tested in the same way, except for the fact that
'tf.concat' is used as additional measure of correct performance.

For predefined models like VGG7, ResNet18 and so on it is necessary to change
padding to 'valid'.

For every type of tested models (e.g. 'single_input', 'vgg') there are two
corresponding fixtures: '<prefix>_setup' and 'get_<prefix>_pipeline'. It is
necessary to split them as it allows us to modify 'config' between creating
it and actually using for 'train_model' action. That is exactly how testing is performed.

There are documentation strings for 'single_setup' and 'get_single_pipeline' fixtures.
Others follow same structure and have same parameters and returns.

There are also convinient 'setup' and 'get_pipeline' fixtures, that allow
to pass 'mode' argument to choose exact type of architecture to use. That allows
to parametrize tests.
"""
# pylint:  disable=import-error, wrong-import-position
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
from ...models.tf.layers import conv_block
from ...models.tf.base import TFModel
from ...models.tf import VGG7, ResNet18, Inception_v1


AVAILABLE_MODELS = ['single_input', 'multi_input',
                    'vgg', 'resnet', 'inception']



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

    Returns
    -------
    Pipeline
        Test pipeline that consists of initialization of needed variables and
        model and preparing for training with given config.
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
def vgg_setup():

    def _vgg_setup(d_f):
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

    return _vgg_setup

@pytest.fixture()
def get_vgg_pipeline():

    def _get_vgg_pipeline(model_class, current_config):
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

    return _get_vgg_pipeline



@pytest.fixture()
def model_setup():

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
                        'initial_block/inputs': 'images',
                        'body/block': {'pool_size': 1, 'pool_strides':1},
                        'initial_block': {'padding': 'valid'}}
        return fake_dataset, model_config

    return _model_setup

@pytest.fixture()
def get_model_pipeline():

    def _get_model_pipeline(model_class, current_config):
        if current_config.get('body'):
            if current_config['body'].get('data_format'):
                current_config['initial_block']['data_format'] = current_config['body']['data_format']
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



@pytest.fixture()
def setup(single_setup, multi_setup,
          vgg_setup, model_setup):

    def _setup(mode, d_f):
        if mode == 'single_input':
            return single_setup(d_f)
        if mode == 'multi_input':
            return multi_setup(d_f)
        if mode == 'vgg':
            return vgg_setup(d_f)
        if mode in AVAILABLE_MODELS[2:]:
            return model_setup(d_f)
        return None

    return _setup

@pytest.fixture
def get_pipeline(get_single_pipeline, get_multi_pipeline,
                 get_vgg_pipeline, get_model_pipeline):

    def _get_pipeline(mode, current_config):
        if mode == 'single_input':
            return get_single_pipeline(current_config)
        if mode == 'multi_input':
            return get_multi_pipeline(current_config)
        if mode == 'vgg':
            return get_vgg_pipeline(VGG7, current_config)
        if mode == 'resnet':
            return get_model_pipeline(ResNet18, current_config)
        if mode == 'inception':
            return get_model_pipeline(Inception_v1, current_config)
        return None

    return _get_pipeline
