""" Pytest configuration. """
# pylint: disable=invalid-name, unused-import
import pytest
import numpy as np

from batchflow import Dataset, ImagesBatch


@pytest.fixture()
def model_setup_images_clf():
    """ Pytest fixture to generate fake dataset and model config for image classification

    Parameters
    ----------
    data_format : 'channels_last' or 'channels_first'

    Returns
    -------
    tuple
        an instance of Dataset
        a model config
    """
    def _model_setup(data_format, image_shape=100):
        dataset_size = 50
        num_classes = 10

        if data_format == 'channels_last':
            image_shape = (image_shape, image_shape, 2)
        else:
            image_shape = (2, image_shape, image_shape)

        batch_shape = (dataset_size, *image_shape)
        images_array = np.random.random(batch_shape)
        labels_array = np.random.choice(num_classes, size=dataset_size)
        data = images_array, labels_array
        dataset = Dataset(index=dataset_size,
                          batch_class=ImagesBatch,
                          preloaded=data)

        model_config = {'classes': 10}
        return dataset, model_config

    return _model_setup
