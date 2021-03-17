""" Test for passing information about data format via config. """
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

from tensorflow.test import is_gpu_available

from batchflow import Pipeline
from batchflow import B, V, C
from batchflow.models.tf import VGG7, ResNet18, Inception_v1, MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small


MODELS = [VGG7, ResNet18, Inception_v1, MobileNet, MobileNet_v2, MobileNet_v3, MobileNet_v3_small]
LOCATIONS = set(['initial_block', 'body', 'block', 'head'])
NO_GPU = pytest.mark.skipif(not is_gpu_available(), reason='No GPU')


@pytest.fixture()
def pipeline():
    """ Creates a pipeline configured to use a given model with a specified configuration.

    Notes
    -----
    Pipeline can be executed only if its config contains the following parameters:

    model_class : TFModel
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
                     .to_array()
                     .train_model('model',
                                  fetches='loss',
                                  images=B('images'),
                                  labels=B('labels'),
                                  save_to=V('current_loss'))
                     )
    return test_pipeline


@pytest.mark.skip(reason="Test is outdated")
@pytest.mark.parametrize('model_and_config',
                         ['single_config', 'multi_config',],
                         indirect=['model_and_config'])
class Test_dataformat():
    """ This class holds tests that are checking ability to pass
    'data_format' to different places in model.

    There is a following pattern in every test:
        First of all, we get class and 'config' of our model via 'model_and_config'
        fixture.
        Then we optionally modify 'config'. In most of them only 'location' is changed.
        Finally, we assert that our modification was actually communicated to desired place.
    """
    @pytest.mark.parametrize('location', LOCATIONS)
    def test_default_dataformat(self, location, model_and_config, single_config):
        """ Default value for 'data_format' is 'channels_last'. """
        if model_and_config == single_config:
            model_class, config = model_and_config
            model_args = model_class(config).model_args
            assert model_args[location + '/data_format'] == 'channels_last'

    @pytest.mark.parametrize('location', LOCATIONS)
    def test_common_dataformat(self, location, model_and_config):
        """ Easiest way to change 'data_format' for every part of network is to
        pass it to 'common'.
        """
        model_class, config = model_and_config
        config['common/data_format'] = 'channels_first'
        model_args = model_class(config).model_args
        assert model_args[location + '/data_format'] == 'channels_first'

    @pytest.mark.parametrize('location', LOCATIONS - set(['block']))
    def test_loc_dataformat(self, location, model_and_config):
        """ 'data_format' can be passed directly to desired location. """
        model_class, config = model_and_config
        destination = location + '/data_format'
        config[destination] = 'channels_first'
        model_args = model_class(config).model_args
        assert model_args[destination] == 'channels_first'
        for loc in LOCATIONS - set([location, 'block']):
            assert 'channels_first' not in model_args[loc].values()

    def test_block_dataformat(self, model_and_config):
        """ Parameters, passed to inner parts take priority over outers. """
        model_class, config = model_and_config
        config['body/data_format'] = 'channels_last'
        config['body/block/data_format'] = 'channels_first'
        model_args = model_class(config).model_args
        assert model_args['block/data_format'] == 'channels_first'


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
class Test_models:
    """ Ensure that a model with given 'data_format' can be built and trained.

    There is a following pattern in every test:
        First of all, we get 'data' and 'config' via 'model_setup_images_clf' fixture.
        Then we optionally modify 'config'. In most cases it is done only at 'location'.
        Finally, we assert that our modification was actually applied to a model by attempting
        to build and train it with a small batch.
    """
    @pytest.mark.parametrize('location', ['common', 'inputs/images'])
    @pytest.mark.parametrize('data_format',
                             [None,
                              pytest.param('channels_first', marks=NO_GPU),
                              'channels_last'])
    def test_data_format(self, model, model_setup_images_clf, pipeline, location, data_format):
        """ We can explicitly pass 'data_format' to inputs or common

        Notes
        -----
        If `data_format` is None, use a default value.

        `channels_first` might not work on CPU as some convolutional and pooling kernels are not implemented yet.
        """
        expected_data_format = data_format or 'channels_last'
        dataset, model_config = model_setup_images_clf(data_format=expected_data_format)
        if data_format:
            model_config[location + '/data_format'] = data_format
        config = {'model_class': model, 'model_config': model_config}
        test_pipeline = (pipeline << dataset).set_config(config)
        batch = test_pipeline.next_batch(2, n_epochs=None)
        model = test_pipeline.get_model_by_name('model')

        assert model.data_format('images') == expected_data_format
        assert len(batch) == 2
