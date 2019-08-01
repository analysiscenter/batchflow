""" Test for model saving and loading """

import pytest
import numpy as np

from batchflow import Pipeline, ImagesBatch, Dataset
from batchflow import B, V, C

from batchflow.models.tf import VGG7 as TF_VGG7


# TODO Extract to utilities
# @pytest.fixture()
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

# TODO Create Pipeline, init model, predict on fake dataset, save predictions, save model
# TODO load saved model, predict on same dataset, compare predictions to saved ones
# TODO think about before/after/inside pipeline
