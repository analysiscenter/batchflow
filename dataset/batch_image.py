""" Contains Batch classes for images """

import traceback

import numpy as np
try:
    import blosc   # pylint: disable=unused-import
except ImportError:
    pass
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
    from .decorators import njit

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed
from .utils import partialmethod


def transform_actions(prefix='', suffix=''):
    """decorator that transforms class's actions which start with prefix and end with suffix
    with BaseImagesBatch.apply_transform and adds modified actions to the class. It ignores methods that start and end
    with '__'
    """
    def decorator(cls):
        """decorates actions"""
        methods_to_add = {}
        for method_name, method in cls.__dict__.items():
            name_slice = slice(len(prefix), -len(suffix))
            if method_name.startswith(prefix) and method_name.endswith(suffix) and\
               not method_name.startswith('__') and not method_name.endswith('__'):
                methods_to_add[method_name[name_slice]] =\
                    partialmethod(cls.apply_probability_transform, transform=method.__func__)
        for method_name, method in methods_to_add.items():
            setattr(cls, method_name, method)
        return cls
    return decorator


class BaseImagesBatch(Batch):
    """ Batch class for 2D images """
    components = "images", "labels"

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = all_res, args, kwargs
        raise NotImplementedError("Must be implemented in a child class")

    def _assemble_load(self, all_res, *args, **kwargs):
        """ Build the batch data after loading data from files """
        _ = all_res, args, kwargs
        return self

    @action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """ Load data """
        return super().load(src, fmt, components, *args, **kwargs)

    @action
    def dump(self, dst=None, fmt=None, components=None, *args, **kwargs):
        """ Save data to a file or a memory object """
        return super().dump(dst, fmt, components=None, *args, **kwargs)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def apply_probability_transform(self, ix, components='images', transform=None, p=1., *args, **kwargs):
        """apply given transfrom to an image with given probability"""
        if p > 1 or p < 0:
            raise ValueError("probability must be in [0,1]")
        image = self.get(ix, components)
        if np.random.binomial(1, p):
            image = transform(self, image, *args, **kwargs).copy()
        return image


@transform_actions(prefix='_', suffix='_')
class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images

    images are stored as numpy arrays (N, H, W, C)

    """

    @property
    def image_shape(self):
        """: tuple - shape of the image """
        return self.images.shape[1:-1]

    def assemble_component(self, all_res, components='images', **kwargs):
        """ Assemble one component """
        try:
            new_images = np.stack(all_res)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                preserve_shape = kwargs.get('preserve_shape', False)
                if preserve_shape:
                    min_shape = np.array([x.shape for x in all_res]).min(axis=0)
                    all_res = [arr[:min_shape[0], :min_shape[1]].copy() for arr in all_res]
                    new_images = np.stack(all_res)
                else:
                    new_images = np.array(all_res, dtype=object)
            else:
                raise e
        setattr(self, components, new_images)

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        components = kwargs.get('components', 'images')
        if isinstance(components, (list, tuple)):
            all_res = list(zip(*all_res))
        else:
            components = [components]
            all_res = [all_res]
        for component, res in zip(components, all_res):
            self.assemble_component(res, component, **kwargs)
        return self

    @action
    def convert_to_pil(self, components='images'):
        """ Convert batch data to PIL.Image format """
        if self.images is None:
            new_images = None
        else:
            new_images = np.asarray(list(None for _ in self.indices))
            self.apply_transform(new_images, components, PIL.Image.fromarray)
        new_data = (new_images, self.labels)
        new_batch = ImagesPILBatch(np.arange(len(self)), preloaded=new_data)
        return new_batch

    @classmethod
    def _calc_origin(cls, image_shape, origin, background_shape):
        if isinstance(origin, str):
            if origin == 'top_left':
                origin = 0, 0
            elif origin == 'center':
                origin = np.maximum(0, np.asarray(background_shape) - image_shape) // 2
            elif origin == 'random':
                origin = (np.random.randint(background_shape[0]-image_shape[0]+1),
                          np.random.randint(background_shape[1]-image_shape[1]+1))
        return np.asarray(origin, dtype=np.int)

    @classmethod
    def _scale_(cls, image, factor, preserve_shape, origin='top_left'):
        """ Scale the content of each image in the batch

        Parameters
        -----------
        factor : float, tuple
            resulting shape is obtained as original_shape * factor
            - float - scale all axes with the given factor
            - tuple (factor_1, factort_2, ...) - scale each axis with the given factor separately

        preserve_shape : bool
            whether to preserve the shape of the image after scaling

        origin : {'center', 'top_left', 'random'}, tuple
            Relevant only if `preserve_shape` is True.
            Position of the scaled image with respect to the original one's shape.
            - 'center' - place the center of the rescaled image on the center of the original one and crop
                         the rescaled image accordingly
            - 'top_left' - place the upper-left corner of the rescaled image on the upper-left of the original one
                           and crop the rescaled image accordingly
            - 'random' - place the upper-left corner of the rescaled image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - tuple - place the upper-left corner of the rescaled image on the given position in the original one.
        """
        if np.any(np.asarray(factor) <= 0):
            raise ValueError("factor must be greater than 0")
        image_shape = image.shape[:-1]
        rescaled_shape = np.ceil(np.array(image_shape) * factor).astype(np.int16)
        rescaled_image = cls._resize_(image, rescaled_shape)
        if preserve_shape:
            # if isinstance(origin, str) and origin not in ['top_left', 'center', 'random']:
                # raise ValueError('str value of origin must be one of [\'top_left\', \'center\', \'random\']')
            # if isinstance(origin, str) and origin == 'random':
                # origin = (np.random.randint(image.shape[0]-rescaled_image.shape[0]+1),
                          # np.random.randint(image.shape[1]-rescaled_image.shape[1]+1))
            rescaled_image = cls._preserve_shape(image, rescaled_image, origin)
        return rescaled_image

    @classmethod
    def _crop_(cls, image, origin, shape):
        """ Crop an images
        Parameters
        ----------
        image : image in numpy format

        origin : tuple
            Upper-left corner of the cropping box. Can be one of:
            - tuple - a starting point in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         an image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position

        shape : tuple
            - tuple - crop size in the form of (rows, columns)
        """

        origin = cls._calc_origin(shape, origin, image.shape[:2])
        if np.all(origin + shape > image.shape[:2]):
            shape = image.shape[:2] - origin

        row_slice = slice(origin[0], origin[0] + shape[0])
        column_slice = slice(origin[1], origin[1] + shape[1])
        return image[row_slice, column_slice].copy()

    @classmethod
    def _put_on_background_(cls, image, background, origin):
        """puts image on background at origin

        Parameters
        ----------
        image : np.array

        background : np.array

        origin : tuple, str
            Upper-left corner of the cropping box. Can be one of:
            - tuple - a starting point in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         an image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position"""

        origin = cls._calc_origin(image.shape[:2], origin, background.shape[:2])
        image = cls._crop_(image, 'top_left', np.asarray(background.shape[:2]) - origin).copy()

        slice_rows = slice(origin[0], origin[0]+image.shape[0])
        slice_columns = slice(origin[1], origin[1]+image.shape[1])

        new_image = background.copy()
        new_image[slice_rows, slice_columns] = image
        return new_image

    @classmethod
    def _preserve_shape(cls, original_image, rescaled_image, origin='center'):
        """ Change the image shape by cropping and/or adding empty pixels to fit the given shape """
        return cls._put_on_background_(cls._crop_(rescaled_image,
                                                  'top_left' if origin != 'center' else 'center',
                                                  original_image.shape[:2]),
                                       np.zeros(original_image.shape, dtype=np.uint8),
                                       origin)

    @classmethod
    def _resize_(cls, image, shape=None, *args, **kwargs):
        """ Resize an image to the given shape

        Parameters
        ----------
        shape : tuple
            resulting shape in the following form: (number of rows, number of columns)
        """
        factor = np.asarray(shape) / np.asarray(image.shape[:2])
        if len(image.shape) > 2:
            factor = np.concatenate((factor, [1.] * len(image.shape[2:])))
        order = kwargs.pop('order', 0)
        new_image = scipy.ndimage.interpolation.zoom(image, factor, order, *args, **kwargs)
        return new_image


    @classmethod
    def _shift_(cls, image, *args, **kwargs):
        """actually a wrapper for scipy.ndimage.interpolation.shift"""
        return scipy.ndimage.interpolation.shift(image, *args, **kwargs)

    @classmethod
    def _rotate_(cls, image, angle, *args, **kwargs):
        """ Rotate an image

        Parameters
        -----------
        angle : float
                image is rotated by the given angle
        """
        new_image = scipy.ndimage.interpolation.rotate(image, angle, *args, **kwargs)
        return new_image.copy()

    @classmethod
    def _flip_(cls, image, mode):
        """ Flip images

        Parameters
        ----------
        image : image in numpy format

        mode : {'lr', 'ud'}
            - 'lr' - apply the left/right flip
            - 'ud' - apply the upside/down flip
        """
        if mode == 'lr':
            return image[:, ::-1]
        elif mode == 'ud':
            return image[::-1]

    @classmethod
    def _pad_(cls, image, *args, **kwargs):
        """ pad an image. args and kwargs are passed to np.pad
        """
        return np.pad(image, *args, **kwargs)

    @classmethod
    def _invert_(cls, image, channels='all'):
        """ invert channels

        Parameters
        ----------
        p : str, tuple of ints
            probabilities of inverting channels.
            - str - invert all channels
            - tuple - (first channel, second channel, ...) - invert given channels
        """
        if channels == 'all':
            channels = list(range(image.shape[-1]))
        inv_multiplier = np.zeros(image.shape[-1], dtype=np.float32)
        inv_multiplier[np.asarray(channels)] = 255
        return np.abs(np.ones(image.shape, dtype=np.float32)*inv_multiplier - image.astype(np.float32)).astype(np.uint8)

    @classmethod
    def _salt_and_pepper_(cls, image, mode=0.5, p_pixel=1, salt=255, pepper=0):
        """ set pixels' intensities to 0 (pepper) or 255 (salt) randomly. Each pixel is chosen
        uniformly with probability equals p_pixel

        Parameters
        ----------
        p_pixel : float, callable
            probability of applying this transform for one pixel

        mode : float
            probability of choosing pepper
        """
        p_pixel = np.asarray(p_pixel).reshape(-1)
        mode = np.asarray(mode).reshape(-1)
        if mode.shape != p_pixel.shape:
            raise ValueError('shapes of `p_pixel` and `mode` must coincide')
        if len(p_pixel) == 1:
            mask = np.random.binomial(1, p_pixel, size=image.shape[:2])
            noise_mask = np.random.binomial(1, mode, size=(mask.sum(), 1))
            noise = (1 - noise_mask) * pepper + salt * noise_mask
            image = image.copy()
            image[mask != 0] = noise
            return image
        else:
            if len(p_pixel) != image.shape[-1]:
                raise ValueError('`p_pixel` must be given to every channel if len(p_pixel) > 1')

    @classmethod
    def _threshold(cls, image, min_value=0., max_value=1., dtype=np.uint8):
        image[image < min_value] = min_value
        image[image > max_value] = max_value
        return image.astype(dtype)

    @classmethod
    def _multiply_(cls, image, multiplier, min_value=0., max_value=1.):
        """multiply each pixel by the given multiplier

        Parameters
        ----------
        image : image in numpy format

        multiplier : float

        min_value : actual pixel's value is equal max(value, min_value)

        max_value : actual pixel's value is equal min(value, max_value)
        """
        return cls._threshold(multiplier * image.astype(np.float), min_value, max_value, image.dtype)

    @classmethod
    def _add_(cls, image, term, min_value=0., max_value=1.):
        """add term to each pixel

        Parameters
        ----------
        image : image in numpy format

        term : float

        min_value : actual pixel's value is equal max(value, min_value)

        max_value : actual pixel's value is equal min(value, max_value)
        """
        return cls._threshold(term + image.astype(np.float), min_value, max_value, image.dtype)


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

@njit(nogil=True)
def calc_origin_numba(image_coord, shape_coord, crop):
    """ Return origin for preserve_shape """
    if crop == 'top_left':
        origin = 0
    elif crop == 'center':
        origin = np.abs(shape_coord - image_coord) // 2
    return origin

@njit(nogil=True)
def calc_coords_numba(image_coord, shape_coord, crop):
    """ Return coords for preserve_shape """
    if image_coord < shape_coord:
        new_image_origin = calc_origin_numba(image_coord, shape_coord, crop)
        image_origin = 0
        image_len = image_coord
    else:
        new_image_origin = 0
        image_origin = calc_origin_numba(image_coord, shape_coord, crop)
        image_len = shape_coord
    return image_origin, new_image_origin, image_len

@njit(nogil=True)
def preserve_shape_numba(image_shape, shape, crop='center'):
    """ Change the image shape by cropping and adding empty pixels to fit the given shape """
    x, new_x, len_x = calc_coords_numba(image_shape[0], shape[0], crop)
    y, new_y, len_y = calc_coords_numba(image_shape[1], shape[1], crop)
    new_x = new_x, new_x + len_x
    x = x, x + len_x
    new_y = new_y, new_y + len_y
    y = y, y + len_y

    return new_x, new_y, x, y


class ImagesPILBatch(BaseImagesBatch):
    """ Batch class for 2D images in PIL format """
    def get_image_size(self, image):
        """ Return image size (width, height) """
        return image.size

    def assemble(self, all_res, *args, **kwargs):
        """ Assemble the batch after a parallel action """
        _ = args, kwargs
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

        components = kwargs.get('components', 'images')
        if isinstance(components, (list, tuple)):
            all_res = list(zip(*all_res))
        else:
            components = [components]
            all_res = [all_res]
        for component, res in zip(components, all_res):
            new_data = np.array(res, dtype='object')
            setattr(self, component, new_data)
        return self

    @action
    def convert_to_array(self, dtype=np.uint8):
        """ Convert images from PIL.Image format to an array """
        if self.images is not None:
            new_images = list(None for _ in self.indices)
            self.apply_transform(self._convert_to_array_one, dst=new_images, src='images', dtype=dtype)
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

    def _resize_one(self, ix, components='images', shape=None, **kwargs):
        """ Resize one image """
        _ = kwargs
        image = self.get(ix, components)
        new_image = image.resize(shape, PIL.Image.ANTIALIAS)
        return new_image

    def _preserve_shape(self, image, shape, crop='center'):
        new_x, new_y, x, y = preserve_shape_numba(image.size, shape, crop)
        new_image = PIL.Image.new(image.mode, shape)
        box = x[0], y[0], x[1], y[1]
        new_image.paste(image.crop(box), (new_x[0], new_y[0]))
        return new_image

    def _rotate_one(self, ix, components='images', angle=0, preserve_shape=True, **kwargs):
        """ Rotate one image """
        image = self.get(ix, components)
        kwargs['expand'] = not preserve_shape
        new_image = image.rotate(angle, **kwargs)
        return new_image

    def _crop_image(self, image, origin, shape):
        """ Crop one image """
        if origin is None or origin == 'top_left':
            origin = 0, 0
        elif origin == 'center':
            origin = (image.width - shape[0]) // 2, (image.height - shape[1]) // 2
        origin_x, origin_y = origin
        shape = shape if shape is not None else (image.width - origin_x, image.height - origin_y)
        box = origin_x, origin_y, origin_x + shape[0], origin_y + shape[1]
        new_image = image.crop(box)
        return new_image

    @inbatch_parallel('indices', post='assemble')
    def _crop(self, ix, components='images', origin=None, shape=None):
        """ Crop all images """
        image = self.get(ix, components)
        new_image = self._crop_image(image, origin, shape)
        return new_image

    @inbatch_parallel('indices', post='assemble')
    def _random_crop(self, ix, components='images', shape=None):
        """ Crop all images with a given shape and a random origin """
        image = self.get(ix, components)
        origin_x = np.random.randint(0, image.width - shape[0])
        origin_y = np.random.randint(0, image.height - shape[1])
        new_image = self._crop_image(image, (origin_x, origin_y), shape)
        return new_image

    @action
    @inbatch_parallel('indices', post='assemble')
    def fliplr(self, ix, components='images'):
        """ Flip image horizontaly (left / right) """
        image = self.get(ix, components)
        return image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    @action
    @inbatch_parallel('indices', post='assemble')
    def flipud(self, ix, components='images'):
        """ Flip image verticaly (up / down) """
        image = self.get(ix, components)
        return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
