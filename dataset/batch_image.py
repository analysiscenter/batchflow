""" Contains Batch classes for images """

import os   # pylint: disable=unused-import
import traceback

try:
    import blosc   # pylint: disable=unused-import
except ImportError:
    pass
import numpy as np
try:
    import PIL.Image
except ImportError:
    pass
try:
    import scipy.ndimage
except ImportError:
    pass
try:
    from numba import njit
except ImportError:
    pass

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed


@njit(nogil=True)
def random_crop_numba(images, shape):
    """ Fill-in new_images with random crops from images """
    new_images = np.zeros((images.shape[0],) + shape, dtype=images.dtype)
    if images.shape[2] - shape[0] > 0:
        origin_x = np.random.randint(0, images.shape[2] - shape[0], size=images.shape[0])
    else:
        origin_x = np.zeros(images.shape[0], dtype=np.array(images.shape).dtype)
    if images.shape[1] - shape[1] > 0:
        origin_y = np.random.randint(0, images.shape[1] - shape[1], size=images.shape[0])
    else:
        origin_y = np.zeros(images.shape[0], dtype=np.array(images.shape).dtype)
    for i in range(images.shape[0]):
        x = slice(origin_x[i], origin_x[i] + shape[0])
        y = slice(origin_y[i], origin_y[i] + shape[1])
        new_images[i, :, :] = images[i, y, x]
    return new_images


class BasicImagesBatch(Batch):
    """ Batch class for 2D images """
    @property
    def components(self):
        return "images", "labels"

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        raise NotImplementedError("Use ImagesBatch")

    @action
    def load(self, src, fmt=None):
        """ Load data """
        return super().load(src, fmt)

    @action
    def dump(self, dst, fmt=None):
        """ Saves data to a file or a memory object """
        _ = dst, fmt
        return self

    @action
    def noop(self):
        """ Do nothing """
        return self

    @action
    def crop(self, component='images', origin=None, shape=None):
        """ Crop all images in the batch
        Args:
            component: string - a component name which data should be cropped
            origin: tuple - a starting point in the form of (x, y)
            shape: tuple - a crop size in the form of (width, height)
        """
        if origin is not None or shape is not None:
            origin = origin if origin is not None else (0, 0)
            self._crop(component, origin, shape)
        return self

    @action
    def random_crop(self, component='images', shape=None):
        """ Crop all images to a given shape and a random origin
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)

        Origin will be chosen at random to fit the required shape
        """
        if shape is None:
            raise ValueError("shape cannot be None")
        else:
            self._random_crop(component, shape)
        return self

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def resize(self, ix, component='images', shape=(64, 64)):
        """ Resize all images in the batch to the given shape
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)
        """
        return self._resize_one(ix, component, shape)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_scale(self, ix, component='images', factor=(0.9, 1.1), preserve_shape=True):
        """ Scale the content of each image in the batch with a random scale factor """
        _factor = np.random.uniform(factor[0], factor[1])
        image = self.get(ix, component)
        shape = image.shape[1:3]
        shape = np.asarray(shape) * _factor
        new_image = self._resize_one(ix, component, shape)
        if preserve_shape:
            new_image = self._crop_image(new_image, (0, 0), image.shape)
        return new_image

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def rotate(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate all images in the batch at the given angle
        Args:
            component: string - a component name which data should be rotated
            angle: float - in radians
            preserve_shape: bool - whether to keep shape after rotating
                                   (always True for images as arrays, can be False for PIL.Images)
        """
        return self._rotate_one(ix, component, angle, preserve_shape, **kwargs)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_rotate(self, ix, component='images', angle=None, **kwargs):
        """ Rotate each image in the batch at a random angle
        Args:
            component: string - a component name which data should be rotated
            angle: tuple - an angle range in the form of (min_angle, max_angle), in radians
        """
        angle = angle if angle is not None else (-np.pi, np.pi)
        _angle = np.random.uniform(angle[0], angle[1])
        preserve_shape = kwargs.pop('preserve_shape', True)
        return self._rotate_one(ix, component, _angle, preserve_shape=preserve_shape, **kwargs)

    def flip(self, axis=None):
        """ Flip images
        Args:
            axis: 'h' for horizontal (left/right) flip
                  'v' for vertical (up/down) flip
            direction: 'l', 'r', 'u', 'd'
        """
        if axis == 'h':
            return self.fliplr()
        elif axis == 'v':
            return self.flipud()
        else:
            raise ValueError("Parameter axis can be 'h' or 'v' only")


class ImagesBatch(BasicImagesBatch):
    """ Batch class for 2D images

    images are stored as numpy arrays (N, H, W) or (N, H, W, C)

    """
    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        component = kwargs.get('component', 'images')
        try:
            new_images = np.stack(all_res)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                min_shape = np.array([x.shape for x in all_res]).min(axis=0)
                all_res = [arr[:min_shape[0], :min_shape[1]].copy() for arr in all_res]
                new_images = np.stack(all_res)
        setattr(self, component, new_images)
        return self

    @action
    def convert_to_pil(self):
        """ Convert batch data to PIL.Image format """
        if self.images is None:
            new_images = None
        else:
            new_images = np.asarray(list(None for _ in self.indices))
            self.apply_transform(new_images, 'images', PIL.Image.fromarray)
        new_data = (new_images, self.labels)
        new_batch = ImagesPILBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    def _resize_one(self, ix, component='images', shape=(64, 64)):
        """ Resize one image """
        image = self.get(ix, component)
        factor = 1. * np.asarray([shape[1], shape[0]]) / np.asarray(image.shape[:2])
        return scipy.ndimage.interpolation.zoom(image, factor, order=3)

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        kwargs['reshape'] = not preserve_shape
        new_image = scipy.ndimage.interpolation.rotate(image, angle, **kwargs)
        return new_image

    def _crop(self, component='images', origin=None, shape=None):
        """ Crop all images in the batch
        Args:
            component: string - a component name which data should be cropped
            origin: tuple - a starting point in the form of (x, y)
            shape: tuple - a crop size in the form of (width, height)
        """
        if origin is not None or shape is not None:
            origin = origin if origin is not None else (0, 0)
            images = self.get(None, component)

            if shape is None:
                shape = images.shape[2], images.shape[1]
            if np.all(np.array(origin) + np.array(shape) > np.array(images.shape[1:3])):
                shape = images.shape[2] - origin[0], images.shape[1] - origin[1]

            x = slice(origin[0], origin[0] + shape[0])
            y = slice(origin[1], origin[1] + shape[1])
            new_images = images[:, y, x].copy()
            setattr(self, component, new_images)

    def _crop_image(self, image, origin, shape):
        return image[origin[1]:origin[1] + shape[1], origin[0]:origin[1] + shape[0]].copy()

    def _random_crop(self, component='images', shape=None):
        if shape is not None:
            images = self.get(None, component)
            new_images = random_crop_numba(images, shape)
            setattr(self, component, new_images)
        return self

    @action
    def fliplr(self, component='images'):
        """ Flip image horizontaly (left / right) """
        images = self.get(component)
        setattr(self, component, images[:, :, ::-1])
        return self

    @action
    def flipud(self, component='images'):
        """ Flip image verticaly (up / down) """
        images = self.get(component)
        setattr(self, component, images[:, ::-1])
        return self


class ImagesPILBatch(BasicImagesBatch):
    """ Batch class for 2D images in PIL format """
    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        component = kwargs.get('component', 'images')
        new_data = np.array(all_res, dtype='object')
        setattr(self, component, new_data)
        return self

    @action
    def convert_to_array(self, dtype=np.uint8):
        """ Convert images from PIL.Image format to an array """
        if self.images is not None:
            new_images = list(None for _ in self.indices)
            self.apply_transform(new_images, 'images', self._convert_to_array_one, dtype=dtype)
            new_images = np.stack(new_images)
        else:
            new_images = None
        new_data = new_images, self.labels
        new_batch = ImagesBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    def _convert_to_array_one(self, image, dtype=np.uint8):
        if image is not None:
            arr = np.fromstring(image.tobytes(), dtype=dtype)
            if image.palette is None:
                new_shape = (image.height, image.width)
            else:
                new_shape = (image.height, image.width, -1)
            new_image = arr.reshape(*new_shape)
        else:
            new_image = None
        return new_image

    def _resize_one(self, ix, component='images', shape=(64, 64), **kwargs):
        """ Resize one image """
        _ = kwargs
        image = self.get(ix, component)
        new_image = image.resize(shape, PIL.Image.ANTIALIAS)
        return new_image

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        kwargs['expand'] = not preserve_shape
        new_image = image.rotate(angle, **kwargs)
        return new_image

    def _crop_image(self, image, origin, shape):
        """ Crop one image """
        origin_x, origin_y = origin
        shape = shape if shape is not None else (image.width - origin_x, image.height - origin_y)
        box = origin_x, origin_y, origin_x + shape[0], origin_y + shape[1]
        return image.crop(box)

    @inbatch_parallel('indices', post='assemble')
    def _crop(self, ix, component='images', origin=(0, 0), shape=None):
        """ Crop all images """
        image = self.get(ix, component)
        return self._crop_image(image, origin, shape)

    @inbatch_parallel('indices', post='assemble')
    def _random_crop(self, ix, component='images', shape=None):
        """ Crop all images with a given shape and a random origin """
        image = self.get(ix, component)
        origin_x = np.random.randint(0, image.width - shape[0])
        origin_y = np.random.randint(0, image.height - shape[1])
        return self._crop_image(image, (origin_x, origin_y), shape)

    @action
    @inbatch_parallel('indices', post='assemble')
    def fliplr(self, ix, component='images'):
        """ Flip image horizontaly (left / right) """
        image = self.get(ix, component)
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    @action
    @inbatch_parallel('indices', post='assemble')
    def flipud(self, ix, component='images'):
        """ Flip image verticaly (up / down) """
        image = self.get(ix, component)
        return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
