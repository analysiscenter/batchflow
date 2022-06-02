""" Test for creation and running torch models. """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

import numpy as np

from batchflow import Pipeline
from batchflow import B, V, C
from batchflow.models.torch import VGG7, EfficientNetB0

MODELS = [VGG7, EfficientNetB0]
IM_SHAPE = [100, EfficientNetB0.resolution]


@pytest.fixture()
def pipeline():
    """ Creates a pipeline configured to use a given model with a specified configuration.

    Notes
    -----
    Pipeline can be executed only if its config contains the following parameters:

    model_class : TorchModel
        Architecture of model. List of available models is defined at 'AVAILABLE_MODELS'.

    model_config : Config
       Model parameters.

    Returns
    -------
    Pipeline
        A pipeline that contains model initialization and training with a given config.
    """

    test_pipeline = (Pipeline()
                     .init_variable('current_loss')
                     .init_model('model', C('model_class'),
                                 'dynamic', C('model_config'))
                     .to_array(dtype='float32')
                     .train_model('model',
                                  inputs=B('images'),
                                  targets=B('labels'),
                                  outputs='loss',
                                  save_to=V('current_loss'))
                     )
    return test_pipeline


@pytest.mark.slow
@pytest.mark.parametrize('model, image_shape', list(zip(MODELS, IM_SHAPE)))
class TestModels:
    """ Ensure that a model can be built and trained.

    There is a following pattern in every test:
        First of all, we get 'data' and 'config' via 'model_setup' fixture.
        Then we optionally modify 'config'. In this case we modify only 'model' argument.
        Finally, we assert that our modification was actually applied to a model by attempting
        to build and train it with a small batch.
    """

    def test_data_format(self, model, image_shape, model_setup_images_clf, pipeline):
        """ We can explicitly pass 'data_format' to inputs or common."""
        dataset, model_config = model_setup_images_clf('channels_first', image_shape=image_shape)
        config = {'model_class': model, 'model_config': model_config}
        test_pipeline = (pipeline << dataset) << config
        batch = test_pipeline.next_batch(2, n_epochs=None)

        assert len(batch) == 2

    @pytest.mark.parametrize('fetches, save_to', [
        ['loss', V('current_loss', mode='a')],
        [['loss', 'predictions'], V('output', mode='a')],
        [['loss', 'predictions'], [V('current_loss', mode='a'), V('predictions', mode='a')]]
    ])
    @pytest.mark.parametrize('microbatch', [4, 2])
    def test_fetches(self, model, image_shape, fetches, save_to, microbatch, model_setup_images_clf, pipeline):
        """ Check different combinations of 'fetches' and 'save_to'. """
        dataset, model_config = model_setup_images_clf('channels_first', image_shape=image_shape)
        pipeline = (Pipeline()
                    .init_variable('current_loss', [])
                    .init_variable('predictions', [])
                    .init_variable('output', [])
                    .init_model('model', C('model_class'),
                                'dynamic', C('model_config'))
                    .to_array(dtype='float32')
                    .train_model('model', inputs=B('images'), targets=B('labels'), outputs=fetches, save_to=save_to)
                    )

        batch_size = 4
        model_config['microbatch'] = microbatch

        config = {'model_class': model, 'model_config': model_config}
        test_pipeline = (pipeline << dataset) << config

        for _ in range(10):
            test_pipeline.next_batch(batch_size, n_epochs=None)

        if len(test_pipeline.v('current_loss')) > 0:
            loss = test_pipeline.v('current_loss')

        if len(test_pipeline.v('predictions')) > 0:
            predictions = test_pipeline.v('predictions')

        if len(test_pipeline.v('output')) > 0:
            loss = [item[0] for item in test_pipeline.v('output')]
            predictions = [item[1] for item in test_pipeline.v('output')]

        assert len(loss) == 10
        if 'predictions' in fetches:
            assert np.concatenate(predictions, axis=0).shape == (batch_size * 10, 10)
