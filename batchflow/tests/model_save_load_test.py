""" Test for model saving and loading """
# pylint: disable=import-error, no-name-in-module

import pytest

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C, I

from batchflow.models.tf import VGG7 as TF_VGG7, TFModel
from batchflow.models.torch import VGG7 as TORCH_VGG7, TorchModel

PATH = 'my_mdl'
BATCH_SIZE = 20


@pytest.mark.slow
@pytest.mark.parametrize('model_class',
                         [pytest.param(TF_VGG7, id="tf"),
                          pytest.param(TORCH_VGG7, id='torch')])
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
    def pipelines(self, model_setup_images_clf):
        def _pipelines(model_class):
            config = {}
            data_format = predict_args = predict_kwargs = None
            if issubclass(model_class, TFModel):
                data_format = 'channels_last'
                config.update({'channels': 'last', 'dtype': None})
                predict_args = ()
                predict_kwargs = dict(images=B('images'))
            elif issubclass(model_class, TorchModel):
                data_format = 'channels_first'
                config.update({'channels': 'first', 'dtype': 'float32'})
                predict_args = (B('images'),)
                predict_kwargs = dict()

            dataset, model_config = model_setup_images_clf(data_format)
            config.update({'model_class': model_class, 'model_config': model_config})

            save_pipeline = (Pipeline()
                             .init_variable('predictions', init_on_each_run=list)
                             .init_model('dynamic', C('model_class'), 'model', C('model_config'))
                             .to_array(channels=C('channels'), dtype=C('dtype'))
                             .predict_model('model', *predict_args,
                                            fetches='predictions', save_to=V('predictions', mode='a'),
                                            **predict_kwargs))
            load_pipeline = (Pipeline()
                             .init_variable('predictions', init_on_each_run=list)
                             .to_array(channels=C('channels'), dtype=C('dtype'))
                             .predict_model('model', *predict_args,
                                            fetches='predictions', save_to=V('predictions', mode='a'),
                                            **predict_kwargs))

            save_pipeline = (save_pipeline << dataset) << config
            load_pipeline = (load_pipeline << dataset) << config
            return save_pipeline, load_pipeline

        return _pipelines

    @staticmethod
    def train_args(model_class):
        args = kwargs = None
        if issubclass(model_class, TFModel):
            args = ()
            kwargs = dict(images=B('images'), labels=B('labels'))
        elif issubclass(model_class, TorchModel):
            args = (B('images'), B('labels'))
            kwargs = dict(fetches='loss')

        return args, kwargs

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

        train_args, train_kwargs = self.train_args(model_class)

        save_tmpl = (Pipeline()
                     .save_model('model', path=save_path + I("current").str())
                     .train_model('model', *train_args, **train_kwargs))

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
