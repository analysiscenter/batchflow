""" Test for model saving and loading """
# pylint: disable=import-error, no-name-in-module

import pickle

import pytest
import dill

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C, I

from batchflow.models.torch import VGG7 as TORCH_VGG7

PATH = 'my_mdl'
BATCH_SIZE = 20


@pytest.mark.slow
@pytest.mark.parametrize('model_class',
                         [pytest.param(TORCH_VGG7, id='torch')])
class TestModelSaveLoad:
    """
    Ensure that a model can be saved and loaded.
    """
    @staticmethod
    @pytest.fixture()
    def save_path(tmp_path):
        """
        Make path in temporary pytest folder for model to be saved to and loaded from
        """
        return str((tmp_path / PATH).absolute())

    @staticmethod
    @pytest.fixture
    def pipelines(model_setup_images_clf):
        """
        make pipelines for model loading and saving, that are compatible with given `model_class`
        """
        def _pipelines(model_class):
            config = {}
            dataset, model_config = model_setup_images_clf(data_format='channels_first')

            save_pipeline = (Pipeline()
                             .init_variable('predictions', default=[])
                             .init_model('dynamic', model_class, 'model', model_config)
                             .to_array(dtype='float32')
                             .predict_model('model', B.images,
                                            fetches='predictions',
                                            save_to=V('predictions', mode='a')))
            load_pipeline = (Pipeline()
                             .init_variable('predictions', default=[])
                             .to_array(dtype=C('dtype'))
                             .predict_model('model', B.images,
                                            fetches='predictions',
                                            save_to=V('predictions', mode='a')))

            save_pipeline = (save_pipeline << dataset) << config
            load_pipeline = (load_pipeline << dataset) << config
            return save_pipeline, load_pipeline

        return _pipelines


    def test_run(self, save_path, pipelines, model_class):
        """
        Check model loading and saving during pipeline iterations

        A model is initialised in save_pipeline, then for each batch:
            predictions are obtained and saved;
            current model state is saved;
            the model is trained.
        After that in load_pipeline for each batch in same dataset:
            the model from corresponding iteration is loaded;
            predictions are obtained and saved.

        Predictions from save_pipeline and from load_pipeline should be equal
        """
        save_pipeline, load_pipeline = pipelines(model_class)

        save_tmpl = (Pipeline()
                     .save_model('model', path=save_path + I("current").str())
                     .train_model('model', B('images'), B('labels'), fetches='loss'))

        save_pipeline = save_pipeline + save_tmpl
        save_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_tmpl = (Pipeline()
                     .load_model('dynamic', C('model_class'), 'model', path=save_path + I("current").str()))

        load_pipeline = load_tmpl + load_pipeline
        load_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    def test_now(self, save_path, pipelines, model_class):
        """
        Test model loading and saving with `save_model_now`  and `load_model_now`
        """
        save_pipeline, load_pipeline = pipelines(model_class)

        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')
        save_pipeline.save_model_now('model', path=save_path)

        load_pipeline.load_model_now('dynamic', C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    def test_after_before(self, save_path, pipelines, model_class):
        """
        Test model saving in pipeline.after and loading in pipeline.before
        """
        save_pipeline, load_pipeline = pipelines(model_class)

        save_pipeline.after.save_model('model', path=save_path)
        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_pipeline.before.load_model('dynamic', C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    @pytest.mark.parametrize('pickle_module', [None, dill, pickle])
    def test_bare_model(self, save_path, model_class, pickle_module):
        """
        Test model saving and loading without pipeline
        """
        pickle_args = {'pickle_module': pickle_module}

        num_classes = 10
        dataset_size = 10
        image_shape = (2, 100, 100)

        model_config = {'inputs/images/shape': image_shape,
                        'inputs/labels/classes': num_classes,
                        'initial_block/inputs': 'images'}
        model_save = model_class(config=model_config)

        batch_shape = (dataset_size, *image_shape)
        images_array = np.random.random(batch_shape)

        args = (images_array.astype('float32'),)
        kwargs = dict(fetches='predictions')

        saved_predictions = model_save.predict(*args, **kwargs)
        model_save.save(path=save_path, **pickle_args)

        model_load = model_class(config=model_config)
        model_load.load(path=save_path, **pickle_args)
        loaded_predictions = model_load.predict(*args, **kwargs)

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()
