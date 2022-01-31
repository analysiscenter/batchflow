""" Test for model saving and loading """
# pylint: disable=import-error, no-name-in-module

import pickle

import pytest
import dill

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C, I

from batchflow.models.torch import TorchModel, VGG7

PATH = 'my_mdl'
BATCH_SIZE = 20


@pytest.mark.slow
@pytest.mark.parametrize('model_class',
                         [VGG7])
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
            config = {'model_class': model_class}
            dataset, model_config = model_setup_images_clf(data_format='channels_first')

            train_template = (Pipeline()
                .init_model('model', model_class, 'dynamic', model_config)
                .to_array(dtype='float32')
                .train_model('model', inputs=B('images'), targets=B('labels'), outputs='loss')
            )

            predict_template = (Pipeline()
                .init_variable('predictions', default=[])
                .to_array(dtype='float32')
                .predict_model('model', inputs=B('images'),
                            outputs='predictions',
                            save_to=V('predictions', mode='a'))
            )

            train_pipeline = (train_template + predict_template) << dataset << config
            inference_pipeline = predict_template << dataset << config

            return train_pipeline, inference_pipeline

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
        save_tmpl = (Pipeline()
                     .save_model('model', path=save_path + I("current").str())
                    )

        load_tmpl = (Pipeline()
                     .load_model('model', C('model_class'), 'dynamic', path=save_path + I("current").str())
                    )

        train_pipeline, predict_pipeline = pipelines(model_class)

        train_pipeline = train_pipeline + save_tmpl
        train_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = train_pipeline.v('predictions')

        predict_pipeline = load_tmpl + predict_pipeline
        predict_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = predict_pipeline.v('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    def test_now(self, save_path, pipelines, model_class):
        """
        Test model loading and saving with `save_model_now`  and `load_model_now`.
        """
        train_pipeline, predict_pipeline = pipelines(model_class)

        train_pipeline.next_batch(BATCH_SIZE, n_epochs=1)
        saved_predictions = train_pipeline.v('predictions')
        train_pipeline.save_model_now('model', path=save_path)

        predict_pipeline.load_model_now('model', C('model_class'), 'dynamic', path=save_path)
        predict_pipeline.next_batch(BATCH_SIZE, n_epochs=1)
        loaded_predictions = predict_pipeline.v('predictions')

        assert (saved_predictions[0] == loaded_predictions[0]).all()

    def test_after_before(self, save_path, pipelines, model_class):
        """
        Test model saving in pipeline.after and loading in pipeline.before
        """
        train_pipeline, predict_pipeline = pipelines(model_class)

        train_pipeline.after.save_model('model', path=save_path)
        train_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = train_pipeline.get_variable('predictions')

        predict_pipeline.before.load_model('model', C('model_class'), 'dynamic', path=save_path)
        predict_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = predict_pipeline.get_variable('predictions')

        assert (saved_predictions[-1] == loaded_predictions[-1]).all()

    def test_outer(self, save_path, pipelines, model_class):
        """
        Test model saving in pipeline.after and loading from created Model instance.
        """
        train_pipeline, predict_pipeline = pipelines(model_class)

        train_pipeline.after.save_model('model', path=save_path)
        train_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = train_pipeline.get_variable('predictions')

        model = TorchModel(config={'load/path': save_path})

        load_tmpl = (Pipeline()
                     .init_model('model', source=model)
                    )

        predict_pipeline = load_tmpl + predict_pipeline

        predict_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = predict_pipeline.get_variable('predictions')

        assert (saved_predictions[-1] == loaded_predictions[-1]).all()

    @pytest.mark.parametrize('pickle_module', [None, dill, pickle])
    @pytest.mark.parametrize('outputs', ['predictions', 'sigmoid'])
    def test_bare_model(self, save_path, model_class, pickle_module, outputs):
        """
        Test model saving and loading without pipeline
        """
        pickle_args = {'pickle_module': pickle_module}

        num_classes = 10
        dataset_size = 10
        image_shape = (2, 100, 100)

        model_config = {
            'classes': num_classes,
            'inputs_shapes': image_shape,
            'output': 'sigmoid'
        }

        model_save = model_class(config=model_config)

        batch_shape = (dataset_size, *image_shape)
        images_array = np.random.random(batch_shape)

        args = (images_array.astype('float32'),)
        kwargs = dict(outputs=outputs)

        saved_predictions = model_save.predict(*args, **kwargs)
        model_save.save(path=save_path, **pickle_args)

        model_load = model_class()
        model_load.load(path=save_path, **pickle_args)
        loaded_predictions = model_load.predict(*args, **kwargs)

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()
