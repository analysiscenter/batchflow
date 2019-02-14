""" Tests for data_format functionality.
Each test follows the same pattern:
    First of all, 'fake_data' and 'config' are created via 'setup'
    fixture. These are Dataset instance and dictionary. They serve
    as placeholder for real data to train model on and simplest possible
    configuration for 'train_model' action respectively.

    After that, 'config' is changed by passing 'data_format' option
    to different keys in it. This step is optional as we want
    to test default behavior aswell.

    Finally, 'test_pipeline' is created via 'get_pipeline' fixture. It is
    then applied to 'fake_dataset'. 'next_batch' is used to actually run it.

The idea behind testing protocol is that it is possible to create such architecture
of neural network (i.e. number of layers, kernels, filters and so on) and
such input data (i.e. size of images and number of channels) that passing
wrong 'data_format' parameter would inevitably raise ValueError.

For example, if we try to propagate 28x28x3 image through convolutional
layer when 'kernel_size' equals 5 and 'data_format' is 'channels_first',
it would result in negative dimension size as (3-5) is less than 0.
For more imformation about exact architecture check fixtures file.

Name of test functions has following structure:
    test_<format of data>_<format passed to config>_<how format is passed>.
    For example, test_last_first_common mean that created data has 'channels_last'
    format and 'channels_first' is passed to 'config/common/data_format'.

Most of the tests are used against multiple architectures, list of which is
defined in TEST_MODELS.
"""
# pylint:  disable=import-error, wrong-import-position
# pylint: disable=missing-docstring, redefined-outer-name
import pytest


TEST_MODELS = ['multi_input', 'single_input',
               'vgg', 'resnet', 'inception']


@pytest.mark.parametrize('mode', TEST_MODELS)
class TestPlural:
    def test_last_last_default(self, mode, setup, get_pipeline):
        """ Default value for 'data_format' is 'channels_last'. """
        fake_dataset, config = setup(mode, d_f='channels_last')

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_last_common(self, mode, setup, get_pipeline):
        """ We can explicitly pass 'data_format', it has no effect in this case. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_last'}
        else:
            config['common'].update({'data_format': 'channels_last'})

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_first_root(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config' does nothing. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        config['data_format'] = 'channels_first'
        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7

    def test_last_first_common(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config/common' raises ValueError. """
        fake_dataset, config = setup(mode, d_f='channels_last')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_first'}
        else:
            config['common'].update({'data_format': 'channels_first'})

        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)
            assert 'layer-0' in str(excinfo)

    def test_last_first_body(self, mode, setup, get_pipeline):
        """ Explicitly passing wrong 'data_format' to 'config/body' raises ValueError. """
        fake_dataset, config = setup(mode, d_f='channels_last')

        if 'body' not in config:
            config['body'] = {'data_format': 'channels_first'}
        else:
            config['body'].update({'data_format': 'channels_first'})

        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)
            assert 'layer-0' in str(excinfo)

    def test_first_last_default(self, mode, setup, get_pipeline):
        """ Default 'data_format' is 'channels_last', so passing data with
        other format raises ValueError.
        """
        fake_dataset, config = setup(mode, d_f='channels_first')
        try:
            test_pipeline = get_pipeline(mode, current_config=config)
            total_pipeline = test_pipeline << fake_dataset
            total_pipeline.next_batch(7, n_epochs=None)
            pytest.fail("Should not have worked")
        except ValueError as excinfo:
            assert 'Negative dimension size' in str(excinfo)

    def test_first_first_common(self, mode, setup, get_pipeline):
        """ That is intended way to communicate 'data_format' with model. """
        fake_dataset, config = setup(mode, d_f='channels_first')
        if 'common' not in config:
            config['common'] = {'data_format': 'channels_first'}
        else:
            config['common'].update({'data_format': 'channels_first'})
        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7


    def test_first_first_body(self, mode, setup, get_pipeline):
        """ Explicitly passing 'data_format' to 'config/body'. """
        fake_dataset, config = setup(mode, d_f='channels_first')

        if 'body' not in config:
            config['body'] = {'data_format': 'channels_first'}
        else:
            config['body'].update({'data_format': 'channels_first'})

        test_pipeline = get_pipeline(mode, current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        n_b = total_pipeline.next_batch(7, n_epochs=None)
        assert len(n_b.index) == 7


def test_first_first_blocks(setup, get_pipeline):
    """ 'data_format' can be passed to a particular block in body,
    though it should be explicitly stated in model definition.
    """
    fake_dataset, config = setup('multi_input', d_f='channels_first')

    config['body/block_1'] = {'data_format': 'channels_first'}
    config['body/block_2'] = {'data_format': 'channels_first'}

    test_pipeline = get_pipeline('multi_input', current_config=config)
    total_pipeline = test_pipeline << fake_dataset
    n_b = total_pipeline.next_batch(7, n_epochs=None)
    assert len(n_b.index) == 7

def test_first_cross_blocks(setup, get_pipeline):
    """ Explicitly passing wrong 'data_format' raises ValueError. """
    fake_dataset, config = setup('multi_input', d_f='channels_first')

    config['body/block_1'] = {'data_format': 'channels_last'}
    config['body/block_2'] = {'data_format': 'channels_first'}
    try:
        test_pipeline = get_pipeline('multi_input', current_config=config)
        total_pipeline = test_pipeline << fake_dataset
        total_pipeline.next_batch(7, n_epochs=None)
        pytest.fail("Should not have worked")
    except ValueError as excinfo:
        assert 'Negative dimension size' in str(excinfo)
        assert 'layer-0' in str(excinfo)
