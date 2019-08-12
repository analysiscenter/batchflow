""" Test for model saving and loading """

import pytest

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C, I

from batchflow.models.tf import VGG7 as TF_VGG7


PATH = 'my_mdl'
BATCH_SIZE = 20


@pytest.mark.slow
@pytest.mark.parametrize('mode',
                         [pytest.param('static', marks=pytest.mark.skip(reason="doesn't work")),
                          'dynamic'])
class TestModelSaveLoad:
    """
    Ensure that a model can be saved and loaded.
    """

    @pytest.fixture()
    def save_path(self, tmp_path):
        """
        Make path in temporary pytest folder for model to be saved to and loaded from
        """
        return str((tmp_path / PATH).absolute())

    @pytest.fixture()
    def load_save_ppls(self, model_setup_images_clf, mode):
        """
        Create pipelines with models to be saved and to be loaded
        Same dataset with no shuffling should be used, so that on same iterations
        the pipelines get same data
        """
        dataset, model_config = model_setup_images_clf(image_shape=(100, 100, 2))
        config = {'model_class': TF_VGG7, 'model_config': model_config}

        save_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .init_model(mode, C('model_class'), 'model', C('model_config'))
                         .to_array()
                         .predict_model('model', images=B('images'), labels=B('labels'),
                                        fetches='predictions', save_to=V('predictions', mode='a')))
        load_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .to_array()
                         .predict_model('model', images=B('images'), labels=B('labels'),
                                        fetches='predictions', save_to=V('predictions', mode='a')))

        save_pipeline = (save_pipeline << dataset) << config
        load_pipeline = (load_pipeline << dataset) << config
        return save_pipeline, load_pipeline

    def test_run(self, load_save_ppls, mode, save_path):
        """
        Check model loading and saving during pipeline iterations

        A model is initialised in save_pipeline, then for each batch:
            predictions are obtained and saved;
            path for model saving is constructed using current iteration number;
            current model state is saved;
            the model is trained.
        After that in load_pipeline for each batch in same dataset:
            path for model loading is constructed using current iteration number;
            the model is loaded;
            predictions are obtained and saved.

        Predictions from save_pipeline and from load_pipeline should be equal
        """
        save_pipeline, load_pipeline = load_save_ppls

        def make_path_name(batch, path, iteration):
            return "{}_{}".format(path, iteration)

        save_tmpl = (Pipeline()
                     .init_variable('save_path')
                     .call(make_path_name, save_to=V("save_path"),
                           kwargs=dict(path=save_path, iteration=I("current")))
                     .save_model('model', path=V("save_path"))
                     .train_model('model', images=B('images'), labels=B('labels')))

        save_pipeline = save_pipeline + save_tmpl
        save_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_tmpl = (Pipeline()
                     .init_variable('save_path')
                     .call(make_path_name, save_to=V("save_path"),
                           kwargs=dict(path=save_path, iteration=I("current")))
                     .load_model(mode, C('model_class'), 'model', path=V("save_path")))

        load_pipeline = load_tmpl + load_pipeline
        load_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    def test_tf_now(self, load_save_ppls, mode, save_path):
        """
        Test model loading and saving with `save_model_now`  and `load_model_now`
        """
        save_pipeline, load_pipeline = load_save_ppls

        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')
        save_pipeline.save_model_now('model', path=save_path)

        load_pipeline.load_model_now(mode, C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    def test_tf_after_before(self, load_save_ppls, mode, save_path):
        """
        Test model saving in pipeline.after and loading in pipeline.before
        """
        save_pipeline, load_pipeline = load_save_ppls

        save_pipeline.after.save_model('model', path=save_path)
        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_pipeline.before.load_model(mode, C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()
