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
    import cv2
except ImportError:
    pass
try:
    from numba import njit, jit
except ImportError:
    pass

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed


@njit(nogil=True)
def crop_numba(images, origin, shape=None):
    """ Fill-in new_images with crops from images """
    if shape is None:
        shape = images.shape[2] - origin[0], images.shape[1] - origin[1]
    if np.all(np.array(origin) + np.array(shape) > np.array(images.shape[1:3])):
        shape = images.shape[2] - origin[0], images.shape[1] - origin[1]
    new_images = np.zeros((images.shape[0],) + shape, dtype=images.dtype)
    x = slice(origin[0], origin[0] + shape[0])
    y = slice(origin[1], origin[1] + shape[1])
    new_images[:] = images[:, y, x]
    return new_images

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
        return "images", "labels", "masks"

    @action
    def load(self, src, fmt=None):
        """ Load data """
        return super().load(src, fmt)

    @action
    def dump(self, dst, fmt=None):
        """ Saves data to a file or a memory object """
        _ = dst, fmt
        return self


class ImagesBatch(BasicImagesBatch):
    """ Batch class for 2D images """
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
        if self.masks is None:
            new_masks = None
        else:
            new_masks = np.asarray(list(None for _ in self.indices))
            self.apply_transform(new_masks, 'masks', PIL.Image.fromarray)
        new_data = (new_images, self.labels, new_masks)
        new_batch = ImagesPILBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def resize(self, ix, component='images', shape=(64, 64), method=None):
        """ Resize all images in the batch to the given shape """
        return self._resize_one(ix, component, shape, method)

    def _resize_one(self, ix, component='images', shape=(64, 64), method=None):
        """ Resize all images in the batch
        if batch contains PIL images or if method is 'PIL',
        uses PIL.Image.resize, otherwise scipy.ndimage.zoom
        We recommend to install a very fast Pillow-SIMD fork """
        image = self.get(ix, component)

        if method == 'cv2':
            new_shape = shape[1], shape[0]
            return cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
        else:
            factor = 1. * np.asarray(shape[::-1]) / np.asarray(image.shape[:2])
            return scipy.ndimage.interpolation.zoom(image, factor, order=3)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_scale(self, ix, component='images', factor=(0.9, 1.1), preserve_shape=True, method=None):
        """ Scale the content of each image in the batch with a random scale factor """
        _factor = np.random.uniform(factor[0], factor[1])
        image = self.get(ix, component)
        shape = image.shape[1:3]
        shape = np.asarray(shape) * _factor
        new_image = self._resize_one(ix, component, shape, method)
        if preserve_shape:
            new_image = new_image[:image.shape[1], :image.shape[0]]
        return new_image

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def rotate(self, ix, component='images', angle=0, preserve_shape=True, method=None, **kwargs):
        """ Rotate all images in the batch at the given angle
        Args:
            component: string - a component name which data should be rotated
            angle: float - in radians
            preserve_shape: bool - whether to keep shape after rotating
                                   (always True for images as arrays, can be False for PIL.Images)
        """
        return self._rotate_one(ix, component, angle, preserve_shape, method, **kwargs)

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, method=None, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        _ = method
        kwargs['reshape'] = not preserve_shape
        new_image = scipy.ndimage.interpolation.rotate(image, angle, **kwargs)
        return new_image

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
            images = self.get(None, component)
            if shape is None:
                shape = images.shape[2], images.shape[1]
            new_images = crop_numba(images, origin, shape)
            setattr(self, component, new_images)
        return self

    @action
    def random_crop(self, component='images', shape=None):
        """ Crop all images to a given shape and a random origin
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)

        Origin will be chosen at random to fit the required shape
        """
        if shape is not None:
            images = self.get(None, component)
            new_images = random_crop_numba(images, shape)
            setattr(self, component, new_images)
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
        if self.masks is not None:
            new_masks = list(None for _ in self.indices)
            self.apply_transform(new_masks, 'masks', self._convert_to_array_one, dtype=dtype)
            new_masks = np.stack(new_images)
        else:
            new_masks = None
        new_data = new_images, self.labels, new_masks
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
        """ Resize all images in the batch """
        _ = kwargs
        image = self.get(ix, component)
        new_image = image.resize(shape, PIL.Image.ANTIALIAS)
        return new_image

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def resize(self, ix, component='images', shape=(64, 64), **kwargs):
        """ Resize all images in the batch to the given shape """
        return self._resize_one(ix, component, shape, **kwargs)


    @action
    @inbatch_parallel(init='indices', post='assemble')
    def random_scale(self, ix, component='images', factor=(0.9, 1.1), preserve_shape=True, **kwargs):
        """ Scale the content of each image in the batch with a random scale factor """
        _factor = np.random.uniform(factor[0], factor[1])

        image = self.get(ix, component)
        shape = image.width, image.height
        shape = np.asarray(shape) * _factor
        new_image = self._resize_one(ix, component, shape, **kwargs)

        if preserve_shape:
            box = 0, 0, image.width, image.height
            new_image = new_image.crop(box).load()
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

    def _rotate_one(self, ix, component='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, component)
        kwargs['expand'] = not preserve_shape
        new_image = image.rotate(angle, **kwargs)
        return new_image

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

    def _crop_one(self, ix, component='images', origin=(0, 0), shape=None):
        image = self.get(ix, component)
        origin_x, origin_y = origin
        shape = shape if shape is not None else (image.width - origin_x, image.height - origin_y)
        box = origin_x, origin_y, origin_x + shape[0], origin_y + shape[1]
        return image.crop(box)

    @inbatch_parallel('indices', post='assemble')
    def _crop(self, ix, component='images', origin=(0, 0), shape=None):
        return self._crop_one(ix, component, origin, shape)

    @action
    def random_crop(self, component='images', shape=None):
        """ Crop all images to a given shape and a random origin
        Args:
            component: string - a component name which data should be cropped
            shape: tuple - a crop size in the form of (width, height)

        Origin will be chosen at random to fit the required shape
        """
        if shape is not None:
            self._random_crop(component, shape)
        return self

    @inbatch_parallel('indices', post='assemble')
    def _random_crop(self, ix, component='images', shape=None):
        image = self.get(ix, component)
        origin_x = np.random.randint(0, image.width - shape[0])
        origin_y = np.random.randint(0, image.height - shape[1])
        return self._crop_one(ix, component, (origin_x, origin_y), shape)
