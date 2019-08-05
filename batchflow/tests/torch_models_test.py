""" Test for creation and running torch models. """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from batchflow import Pipeline, ImagesBatch, Dataset
from batchflow import B, V, C
from batchflow.models.torch import VGG7

MODELS = [VGG7]

@pytest.fixture()
def model_setup():
    """ Pytest fixture to generate fake dataset and model config.

    Returns
    -------
    tuple
        an instance of Dataset
        a model config
    """
    def _model_setup():
        size = 50
        image_shape = (2, 100, 100)
        batch_shape = (size, *image_shape)
        images_array = np.random.random(batch_shape)
        labels_array = np.random.choice(10, size=size)
        data = images_array, labels_array
        dataset = Dataset(index=size,
                          batch_class=ImagesBatch,
                          preloaded=data)

        model_config = {'inputs/images/shape': image_shape,
                        'inputs/labels/classes': 10,
                        'initial_block/inputs': 'images'}
        return dataset, model_config

    return _model_setup

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
                     .init_model('dynamic', C('model_class'),
                                 'model', C('model_config'))
                     .to_array(channels='first', dtype='float32')
                     .train_model('model',
                                  B('images'),
                                  B('labels'),
                                  fetches='loss',
                                  save_to=V('current_loss'))
                     )
    return test_pipeline

@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
class Test_models:
    """ Ensure that a model can be built and trained.

    There is a following pattern in every test:
        First of all, we get 'data' and 'config' via 'model_setup' fixture.
        Then we optionally modify 'config'. In this case we modify only 'model' argument.
        Finally, we assert that our modification was actually applied to a model by attempting
        to build and train it with a small batch.
    """
    @pytest.mark.parametrize('decay', [None, 'exp'])
    def test_data_format(self, model, model_setup, pipeline, decay):
        """ We can explicitly pass 'data_format' to inputs or common."""
        dataset, model_config = model_setup()
        model_config.update(decay=decay, n_iters=25)
        config = {'model_class': model, 'model_config': model_config}
        test_pipeline = (pipeline << dataset) << config
        batch = test_pipeline.next_batch(2, n_epochs=None)

        assert len(batch) == 2
