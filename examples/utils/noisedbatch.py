#pylint:disable=attribute-defined-outside-init

"""Batch class for generating images with noised MNIST digits."""
import numpy as np

from dataset import ImagesBatch, action, inbatch_parallel


class NoisedBatch(ImagesBatch):
    """Batch class for LinkNet."""

    components = 'images', 'labels', 'coordinates', 'noise'

    @action
    def normalize_images(self):
        """Normalize pixel values to (0, 1)."""
        self.images = self.images / 255.
        return self

    @action
    @inbatch_parallel(init='images', post='_assemble', components='noise')
    def create_noise(self, image, *args):
        """Create noise at MNIST image."""
        image_size = self.images.shape[1]
        filt = self.images.shape[-1]
        noise = (args[0] * np.random.random((image_size, image_size, filt)) * image.max())
        return (noise,)

    @action
    @inbatch_parallel(init='indices', post='_assemble')
    def add_noise(self, ind):
        """Add noise at MNIST image."""
        if self.images.shape[-1] == 1:
            return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)
        else:
            return (np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0),)
