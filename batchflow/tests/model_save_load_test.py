""" Test for model saving and loading """
# pylint: disable=import-error, no-name-in-module

import pytest

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C, I

from batchflow.models.tf import VGG7 as TF_VGG7
from batchflow.models.torch import VGG7 as TORCH_VGG7

PATH = 'my_mdl'
BATCH_SIZE = 20


@pytest.mark.slow
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

    @pytest.fixture
    def tf_pipelines(self, model_setup_images_clf):
        """
        Create pipelines with TF models to be saved and to be loaded
        Same dataset with no shuffling should be used, so that on same iterations
        the pipelines get same data
        """
        dataset, model_config = model_setup_images_clf('channels_last')
        config = {'model_class': TF_VGG7, 'model_config': model_config}

        save_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .init_model('dynamic', C('model_class'), 'model', C('model_config'))
                         .to_array()
                         .predict_model('model',
                                        fetches='predictions', save_to=V('predictions', mode='a'),
                                        images=B('images')))
        load_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .to_array()
                         .predict_model('model',
                                        fetches='predictions', save_to=V('predictions', mode='a'),
                                        images=B('images')))

        save_pipeline = (save_pipeline << dataset) << config
        load_pipeline = (load_pipeline << dataset) << config
        return save_pipeline, load_pipeline

    @pytest.fixture
    def torch_pipelines(self, model_setup_images_clf):
        """
        Create pipelines with Torch models to be saved and to be loaded
        see :meth:`~.TestModelSaveLoad.tf_pipelines` for details
        """
        dataset, model_config = model_setup_images_clf('channels_first')
        config = {'model_class': TORCH_VGG7, 'model_config': model_config}

        save_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .init_model('dynamic', C('model_class'), 'model', C('model_config'))
                         .to_array(channels='first', dtype='float32')
                         .predict_model('model', B('images'),
                                        fetches='predictions', save_to=V('predictions', mode='a')))
        load_pipeline = (Pipeline()
                         .init_variable('predictions', init_on_each_run=list)
                         .to_array(channels='first', dtype='float32')
                         .predict_model('model', B('images'),
                                        fetches='predictions', save_to=V('predictions', mode='a')))

        save_pipeline = (save_pipeline << dataset) << config
        load_pipeline = (load_pipeline << dataset) << config
        return save_pipeline, load_pipeline

    @pytest.fixture
    def fixture_dict(self, tf_pipelines, torch_pipelines):
        """
        workaround for passing parametrized fixtures as parameters
        """
        return dict(tf_pipelines=tf_pipelines, torch_pipelines=torch_pipelines)

    @pytest.mark.parametrize('fixture_name, model_args, model_kwargs',
                             [pytest.param('tf_pipelines', (), dict(images=B('images'), labels=B('labels')),
                                           id="tf"),
                              pytest.param('torch_pipelines', (B('images'), B('labels')), dict(fetches='loss'),
                                           id='torch')])
    def test_run(self, fixture_dict, fixture_name, save_path, model_args, model_kwargs):
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
        save_pipeline, load_pipeline = fixture_dict[fixture_name]

        save_tmpl = (Pipeline()
                     .save_model('model', path=save_path + I("current").str())
                     .train_model('model', *model_args, **model_kwargs))

        save_pipeline = save_pipeline + save_tmpl
        save_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_tmpl = (Pipeline()
                     .load_model('dynamic', C('model_class'), 'model', path=save_path + I("current").str()))

        load_pipeline = load_tmpl + load_pipeline
        load_pipeline.run(BATCH_SIZE, n_epochs=1, bar=True)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    @pytest.mark.parametrize('fixture_name',
                             [pytest.param('tf_pipelines',
                                           id="tf"),
                              pytest.param('torch_pipelines',
                                           id='torch')])
    def test_now(self, fixture_dict, fixture_name, save_path):
        """
        Test model loading and saving with `save_model_now`  and `load_model_now`
        """
        save_pipeline, load_pipeline = fixture_dict[fixture_name]

        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')
        save_pipeline.save_model_now('model', path=save_path)

        load_pipeline.load_model_now('dynamic', C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()

    @pytest.mark.parametrize('fixture_name',
                             [pytest.param('tf_pipelines',
                                           id="tf"),
                              pytest.param('torch_pipelines',
                                           id='torch')])
    def test_after_before(self, fixture_dict, fixture_name, save_path):
        """
        Test model saving in pipeline.after and loading in pipeline.before
        """
        save_pipeline, load_pipeline = fixture_dict[fixture_name]

        save_pipeline.after.save_model('model', path=save_path)
        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_pipeline.before.load_model('dynamic', C('model_class'), 'model', path=save_path)
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (np.concatenate(saved_predictions) == np.concatenate(loaded_predictions)).all()
