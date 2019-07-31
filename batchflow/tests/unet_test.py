""" Test for EncoderDecoder model architecture.
First of all, we define possible types of encoders, embeddings and decoders.
Later every combination of encoder, embedding, decoder is combined into one model and we initialize it.
"""
# pylint: disable=import-error, no-name-in-module
# pylint: disable=redefined-outer-name
import pytest

from batchflow.models.tf import UNet

INITIAL_BLOCK = [
    {},
    {'filters': 32},
]

ENCODERS = [
    {},
    {'num_stages': 2},
    {'num_stages': 4, 'filters': [[64, 128], [128, 256], [256, 512], [512, 1024]]},
]


EMBEDDINGS = [
    {},
]


DECODERS = [
    {},
]


@pytest.fixture()
def base_config():
    """ Fixture to hold default configuration. """
    config = {
        'inputs': {'images': {'shape': (16, 16, 1)},
                   'masks': {'name': 'targets', 'shape': (16, 16, 1)}},
        'initial_block': {'inputs': 'images'},
        'loss': 'mse'
    }
    return config


@pytest.mark.slow
@pytest.mark.parametrize('initial_block', INITIAL_BLOCK)
@pytest.mark.parametrize('decoder', DECODERS)
@pytest.mark.parametrize('embedding', EMBEDDINGS)
@pytest.mark.parametrize('encoder', ENCODERS)
def test_first(base_config, initial_block, encoder, embedding, decoder):
    """ Create encoder-decoder architecture from every possible combination
    of encoder, embedding, decoder, listed in global variables defined above.
    """
    base_config.update({'body/encoder': encoder,
                        'body/embedding': embedding,
                        'body/decoder': decoder})
    base_config['initial_block'].update(initial_block)

    print(base_config)

    _ = UNet(base_config)
