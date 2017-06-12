""" Contains Batch classes for images """

import os

try:
    import blosc
except ImportError:
    pass
import numpy as np
import PIL.Image

from . import action, inbatch_parallel


class ImagesBatch(Batch):
    """ Batch class for 2D images """
    @property
    def images(self):
        """ Images """
        return self.data[0] if self.data is not None else None

    @images.setter
    def images(self, value):
        self._data[0] = value

    @property
    def labels(self):
        """ Labels for images """
        return self.data[1] if self.data is not None else None

    @labels.setter
    def labels(self, value):
        self._data[1] = value

    @property
    def masks(self):
        """ Masks for images """
        return self.data[2] if self.data is not None else None

    @masks.setter
    def masks(self, value):
        self._data[3] = value

    def get_image(self, *args, **kwargs):
        return [self.images[i] for i in range(self.indices)]

    def _assemble_batch(self, all_res, *args, **kwargs):
        if any_action_failed(all_res):
            raise ValueError("Could not assemble the batch", self.get_errors(all_res))
        self.images = np.concatenate(all_res)
        return self

    @inbatch_parallel(init='get_image', post='_assemble_batch')
    def resize(self, image, shape):
        """ Resize all images in the batch
        Uses a very fast implementation from Pillow-SIMD """
        return PIL.Image.fromarray(image).resize(shape, PIL.Image.ANTIALIAS)

    @action
    def load(self, src, fmt=None):
        """ Load data """
        if fmt is None:
            if isinstance(src, tuple):
                self._data = tuple(src[i][self.indices] if len(src) > i else None for i in range(3))
            else:
                self._data = src[self.indices], None, None
        else:
            raise ValueError("Unsupported format:", fmt)
        return self

    @action
    def dump(self, dst, fmt=None):
        """ Saves data to a file or array """
        _ = dst, fmt
        return self
