""" Test for model saving and loading """

import pytest

from batchflow import Pipeline
from batchflow import B, V, C

from batchflow.models.tf import VGG7 as TF_VGG7


PATH = 'my_mdl'
BATCH_SIZE = 20

save_pipeline = (Pipeline()
                 .init_variable('predictions')
                 .init_model('dynamic', C('model_class'), 'model', C('model_config'))
                 .to_array()
                 .predict_model('model', images=B('images'), labels=B('labels'),
                                fetches='predictions', save_to=V('predictions', mode='w'))
                 .save_model('model', path=PATH))

load_pipeline = (Pipeline()
                 .init_variable('predictions')
                 .load_model('dynamic', C('model_class'), 'model', path=PATH)
                 .to_array()
                 .predict_model('model', images=B('images'), labels=B('labels'),
                                fetches='predictions', save_to=V('predictions', mode='w'))
                 )

PIPELINES = [(load_pipeline, save_pipeline)]


@pytest.mark.slow
class TestModelSaveLoad:
    """ Ensure that a model can be built and trained.

    There is a following pattern in every test:
        First of all, we get 'data' and 'config' via 'model_setup' fixture.
        Then we optionally modify 'config'. In this case we modify only 'model' argument.
        Finally, we assert that our modification was actually applied to a model by attempting
        to build and train it with a small batch.
    """

    @pytest.mark.parametrize('pipelines', PIPELINES)
    def test_tf_model_load_save(self, pipelines, model_setup_images_clf):
        """ We can explicitly pass 'data_format' to inputs or common."""

        dataset, model_config = model_setup_images_clf(image_shape=(100, 100, 2))
        config = {'model_class': TF_VGG7, 'model_config': model_config}

        load_pipeline, save_pipeline = pipelines

        save_pipeline = (save_pipeline << dataset) << config
        save_pipeline.run(BATCH_SIZE, n_epochs=1)
        saved_predictions = save_pipeline.get_variable('predictions')

        load_pipeline = (load_pipeline << dataset) << config
        load_pipeline.run(BATCH_SIZE, n_epochs=1)
        loaded_predictions = load_pipeline.get_variable('predictions')

        assert (saved_predictions == loaded_predictions).all()
