""" Contains Batch classes for images """

import os   # pylint: disable=unused-import

try:
    import blosc   # pylint: disable=unused-import
except ImportError:
    pass
import numpy as np
import PIL.Image

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed


class ImagesBatch(Batch):
    """ Batch class for 2D images """
    @property
    def data(self):
        data = super().data
        return data if data is not None else tuple([None, None, None])

    @property
    def images(self):
        """ Images """
        return self.data[0] if self.data is not None else None

    @images.setter
    def images(self, value):
        """ Set images """
        data = list(self.data)
        data[0] = value
        self._data = data

    @property
    def labels(self):
        """ Labels for images """
        return self.data[1] if self.data is not None else None

    @labels.setter
    def labels(self, value):
        """ Set labels """
        data = list(self.data)
        data[1] = value
        self._data = data

    @property
    def masks(self):
        """ Masks for images """
        return self.data[2] if self.data is not None else None

    @masks.setter
    def masks(self, value):
        """ Set masks """
        data = list(self.data)
        data[3] = value
        self._data = data

    def get_image(self, *args, **kwargs):
        _ = args, kwargs
        return [self.images[i] for i in range(len(self.indices))]

    def _assemble_batch(self, all_res, *args, **kwargs):
        _ = args, kwargs
        if any_action_failed(all_res):
            raise ValueError("Could not assemble the batch", self.get_errors(all_res))

        if isinstance(all_res[0], PIL.Image.Image):
            self.images = all_res
        else:
            self.images = np.transpose(np.dstack(all_res), (2, 0, 1))
        return self

    @inbatch_parallel(init='get_image', post='_assemble_batch')
    def resize(self, image, shape, method=None):
        """ Resize all images in the batch
        if batch contains PIL images or if method is 'PIL',
        uses PIL.Image.resize, otherwise scipy.ndimage.zoom
        We recommend to install a very fast Pillow-SIMD fork """
        if isinstance(image, PIL.Image.Image):
            return image.resize(shape, PIL.Image.ANTIALIAS)
        else:
            if method == 'PIL':
                new_image = PIL.Image.fromarray(image).resize(shape, PIL.Image.ANTIALIAS)
                new_arr = np.fromstring(new_image.tobytes(), dtype=image.dtype)
                if len(image.shape) == 2:
                    new_arr = new_arr.reshape(new_image.height, new_image.width)
                elif len(image.shape) == 3:
                    new_arr = new_arr.reshape(new_image.height, new_image.width, -1)
                return new_arr
            else:
                factor = 1. * np.asarray(shape) / np.asarray(image.shape)
                return scipy.ndimage.zoom(image, factor, order=3)

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

    @action
    @inbatch_parallel(init='indices')
    def apply_transform(self, ix, dst, fn, *args, **kwargs):
        dst_attr = getattr(self, dst)
        pos = self.index.get_pos(ix)
        dst_attr[pos] = fn(dst_attr[pos], *args, **kwargs)
        return self
