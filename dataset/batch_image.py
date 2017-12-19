""" Contains Batch classes for images """

import os   # pylint: disable=unused-import
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
# try:
#     from numba import njit
# except ImportError:
#     from .decorators import njit

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed

# @njit(nogil=True)
# def calc_coords(image_coord, shape_coord, crop):
#     """ Return coords for preserve_shape """
#     if image_coord < shape_coord:
#         new_image_origin = calc_origin(image_coord, shape_coord, crop)
#         image_origin = 0
#         image_len = image_coord
#     else:
#         new_image_origin = 0
#         image_origin = calc_origin(image_coord, shape_coord, crop)
#         image_len = shape_coord
#     return image_origin, new_image_origin, image_len

# @njit(nogil=True)
# def preserve_shape_numba(image_shape, shape, crop='center'):
#     """ Change the image shape by cropping and adding empty pixels to fit the given shape """
#     x, new_x, len_x = calc_coords(image_shape[0], shape[0], crop)
#     y, new_y, len_y = calc_coords(image_shape[1], shape[1], crop)
#     new_x = new_x, new_x + len_x
#     x = x, x + len_x
#     new_y = new_y, new_y + len_y
#     y = y, y + len_y
#     return new_x, new_y, x, y


class BaseArg():
    """Base class which wraps arguments passed to the batch's actions"""
    def __init__(self):
        pass

    def __call__(self):
        raise ValueError('must be implemented in a child class')


class ConstArg(BaseArg):
    """Wrapper for the constant argument"""
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class CallArg(BaseArg):
    """Wrapper for the callable argument"""
    def __init__(self, function, **kwargs):
        if not callable(function):
            raise ValueError('`function` must be a callable without arguments')
        self.function = lambda : function(**kwargs)

    def __call__(self):
        return self.function()


def convert_args_type(*args):
    """convert raw arguments to the instances of the BaseArg class's children"""
    args_to_return = []
    for argument in args:
        if isinstance(argument, BaseArg):
            args_to_return.append(argument)
        elif callable(argument):
            args_to_return.append(CallArg(argument))
        else:
            args_to_return.append(ConstArg(argument))
    return args_to_return if len(args_to_return) > 1 else args_to_return[0]


class BaseImagesBatch(Batch):
    """ Batch class for 2D images """
    components = "images", "labels"

    @property
    def image_shape(self):
        """: tuple - shape of the images """
        return self.images.shape[1:]

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

    @staticmethod
    def get_image_shape(image):
        """ Return an image size (rows, columns) """
        raise NotImplementedError("must be implemented in child classes")

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def crop(self, ix, components='images', origin='top_left', shape=None):
        """ Crop all images in the batch

        Parameters
        ----------
        components : str
            name of the component to crop

        origin : tuple, BaseArg
            Upper-left corner of the cropping box. Can be one of:
            - tuple - a starting point in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         an image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position

        shape : tuple, BaseArg
            - tuple - crop size in the form of (rows, columns)
        """

        origin, shape = convert_args_type(origin, shape)

        origin = origin()
        if isinstance(origin, str) and origin not in ['top_left', 'center', 'random']:
            raise ValueError('origin must be either in [\'top_left\', \'center\', \'random\'] or be a tuple')

        shape = shape()
        if shape is None:
            raise ValueError('shape must be specified')

        image = self.get(ix, components)


        if origin == 'random':
            origin = (np.random.randint(image.shape[0]-shape[0]+1),
                      np.random.randint(image.shape[1]-shape[1]+1))
        return self._crop_image(image, origin, shape)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def resize(self, ix, components='images', shape=(64, 64)):
        """ Resize all images in the batch to the given shape

        Parameters
        ----------
        components : str
            name of the component to resize

        shape : tuple, BaseArg
            resulting shape in the following form: (number of rows, number of columns)
        """

        shape = convert_args_type(shape)()
        return self._resize_image(self.get(ix, components), shape)

    @staticmethod
    def _preserve_shape(original_image, rescaled_image, origin='center'):
        """ Change the image shape by cropping and adding empty pixels to fit the given shape """
        raise NotImplementedError()

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def scale(self, ix, components='images', factor=None, preserve_shape=True, origin='center', p=1.):
        """ Scale the content of each image in the batch

        Parameters
        -----------
        components : str
            name of the component to scale

        p : float, BaseArg
            probability of applying this action to an element of the batch
            (0. - don't apply, .5 - apply to approximately one half of images, 1 - apply to all images)

        factor : float, tuple, callable, BaseArg
            resulting shape is obtained as original_shape * factor
            - float - scale all axes with the given factor
            - tuple (factor_1, factort_2, ...) - scale each axis with the given factor separately
            - callable - get factor's value from callable

        preserve_shape : bool, BaseArg
            whether to preserve the shape of the image after scaling

        origin : {'center', 'top_left', 'random'}, tuple, callable, BaseArg
            Relevant only if `preserve_shape` is True. Position of the scaled image with respect to the original one's shape.
            - 'center' - place the center of the rescaled image on the center of the original one and crop
                         the rescaled image accordingly
            - 'top_left' - place the upper-left corner of the rescaled image on the upper-left of the original one and crop
                         the rescaled image accordingly
            - 'random' - place the upper-left corner of the rescaled image on the randomly sampled position in the original one.
                         Position is sampled uniformly such that there is no need for cropping.
            - tuple - place the upper-left corner of the rescaled image on the given position in the original one.
            - callable - place the upper-left corner of the rescaled image on the given position in the original one.
                         Position is obtained from callable.
        """
        factor, preserve_shape, origin, p = convert_args_type(factor, preserve_shape, origin, p)

        p = p()
        if p > 1 or p < 0:
            raise ValueError("probability must be in [0,1]")

        image = self.get(ix, components)
        if np.random.random() < p:
            factor = factor()
            if np.any(np.asarray(factor) <= 0):
                raise ValueError("factor must be greater than 0")
            image_shape = self.get_image_shape(image)
            rescaled_shape = np.ceil(np.array(image_shape) * factor).astype(np.int16)
            rescaled_image = self._resize_image(image, rescaled_shape)
            if preserve_shape():
                origin = origin()
                if isinstance(origin, str) and origin not in ['top_left', 'center', 'random']:
                    raise ValueError('str value of origin must be one of [\'top_left\', \'center\', \'random\']')
                if isinstance(origin, str) and origin == 'random':
                    origin = (np.random.randint(image.shape[0]-rescaled_image.shape[0]+1),
                              np.random.randint(image.shape[1]-rescaled_image.shape[1]+1))
                rescaled_image = self._preserve_shape(image, rescaled_image, origin)
        return rescaled_image

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def rotate(self, ix, components='images', angle=30, p=1., **kwargs):
        """ Rotate each image in the batch

        Parameters
        -----------
        components : str
            name of the component to rotate

        p : float, BaseArg
            probability of applying this action to an element of the batch
            (0. - don't apply, .5 - apply to approximately one half of images, 1 - apply to all images)

        angle : float, callable, BaseArg
            float - image is rotated by the given angle
            callable - image is rotated by the angle sampled from callable
        """

        angle, p = convert_args_type(angle, p)

        p = p()
        if p > 1 or p < 0:
            raise ValueError("probability must be in [0,1]")

        image = self.get(ix, components)
        if np.random.random() < p:
            angle = angle()
            preserve_shape = kwargs.pop('preserve_shape', True)
            image = self._rotate_image(image, angle, preserve_shape=preserve_shape, **kwargs)
        return image


    @action
    @inbatch_parallel(init='indices', post='assemble')
    def flip(self, ix, components='images', mode='lr', p=1):
        """ Flip images

        Parameters
        ----------
        components : str
            name of the component to flip

        mode : {'lr', 'ud'}, float, callable
            probability of applying the left\right flip.
            - 'lr' - apply the left\right flip (alias for mode=1)
            - 'ud' - apply the upside\down flip (alias for mode=0)
            - float - apply the left\right flip with the given probability
            - callable - apply the left\right flip with the probability sampled from callable

        p : float, BaseArg
            probability of applying this action to an element of the batch
            (0. - don't apply, .5 - apply to approximately one half of images, 1 - apply to all images)
        """
        p, mode = convert_args_type(p, mode)

        p = p()
        if p > 1 or p < 0:
            raise ValueError("probability must be in [0,1]")

        image = self.get(ix, components)
        if np.random.random() < p:
            mode = mode()
            mode_is_number = isinstance(mode, (int, float))
            if (mode not in ['lr', 'ud'] and not mode_is_number)\
                    or (mode_is_number and (mode < 0 or mode > 1)):
                raise ValueError("`mode` must be one of 'lr', 'ud' or float in [0,1] but {0} is passed".format(mode))
            mode = mode if mode_is_number else (1 if mode == 'lr' else 0)
            return self._flip_image(image, 'lr' if np.random.random() < mode else 'ud')
        return image


    @action
    @inbatch_parallel(init='indices', post='assemble')
    def pad(self, ix, components='images', **kwargs):
        """ pad an image. kwargs are passed to the corresponding function in a child class
        """
        return self._pad(self.get(ix, components), **kwargs)

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def invert(self, ix, components='images', p=1):
        """ invert channels

        Parameters
        ----------
        p : float, tuple, callable
            the probabilities of inverting channels.
            - float - invert all channels at once with the given probability
            - tuple - (first channel invert prob, second channel invert prob, ...) - invert each channel
                      with the given probability (or don't, if the probability is not given)
        """
        p = convert_args_type(p)()

        p = np.asarray(p).reshape(-1)
        if np.any((np.asarray(p) > 1) + (np.asarray(p) < 0)):
            raise ValueError("probability must be in [0,1]")

        channels = ()
        image = self.get(ix, components)
        if len(p) == 1:
            if np.any(np.random.random() < p):
                channels = list(range(image.shape[-1]))
        else:
            channels = np.where(np.random.uniform(size=len(p)) > p)[0]

        if len(channels) > 0:
            return self._invert(image, channels)
        return image

    @action
    def normalize(self, components='images'):
        """divide pixel's intencities by 255"""

        setattr(self, components, self.get(None, components) / 255.)
        return self



#------------------------------------------------------------------------------------------------


class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images

    images are stored as numpy arrays (N, H, W) or (N, H, W, C)

    """
    @staticmethod
    def get_image_shape(image):
        """ Return image shape (rows, columns) """
        return image.shape[:2]

    def assemble_component(self, all_res, components='images'):
        """ Assemble one component """
        try:
            new_images = np.stack(all_res)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                min_shape = np.array([x.shape for x in all_res]).min(axis=0)
                all_res = [arr[:min_shape[0], :min_shape[1]].copy() for arr in all_res]
                new_images = np.stack(all_res)
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
            self.assemble_component(res, component)
        return self

    # @action
    # def convert_to_pil(self, components='images'):
    #     """ Convert batch data to PIL.Image format """
    #     if self.images is None:
    #         new_images = None
    #     else:
    #         new_images = np.asarray(list(None for _ in self.indices))
    #         self.apply_transform(new_images, components, PIL.Image.fromarray)
    #     new_data = (new_images, self.labels)
    #     new_batch = ImagesPILBatch(np.arange(len(self)), preloaded=new_data)
    #     return new_batch

    @staticmethod
    def _calc_origin(image_shape, origin, shape):
        if isinstance(origin, str):
            if origin == 'top_left':
                origin = 0, 0
            elif origin == 'center':
                origin = np.maximum(0, image_shape - np.asarray(shape)) // 2
        return np.asarray(origin, dtype=np.int)


    @staticmethod
    def _crop_image(image, origin, shape):
        origin = ImagesBatch._calc_origin(image.shape[:2], origin, shape)

        if np.all(origin + shape > image.shape[:2]):
            shape = image.shape[:2] - origin

        row_slice = slice(origin[0], origin[0] + shape[0])
        column_slice = slice(origin[1], origin[1] + shape[1])
        return image[row_slice, column_slice].copy()


    @staticmethod
    def _put_on_background(background, image, origin):
        if not isinstance(origin, str):
            b_origin = origin
            image = ImagesBatch._crop_image(image, 'top_left', np.asarray(background.shape[:2]) - origin).copy()
        else:
            b_origin = ImagesBatch._calc_origin(  background.shape[:2], origin, image.shape[:2])
            origin = ImagesBatch._calc_origin(image.shape[:2], origin,  background.shape[:2])
            image = ImagesBatch._crop_image(image, origin, background.shape[:2]).copy()

        slice_rows = slice(b_origin[0], b_origin[0]+image.shape[0])
        slice_columns = slice(b_origin[1], b_origin[1]+image.shape[1])
        new_image = background.copy()
        new_image[slice_rows, slice_columns] = image
        return new_image


    @staticmethod
    def _preserve_shape(original_image, rescaled_image, origin='center'):
        """ Change the image shape by cropping and/or adding empty pixels to fit the given shape """
        new_image = np.zeros(original_image.shape, dtype=np.uint8)
        return ImagesBatch._put_on_background(new_image, rescaled_image, origin)


    @staticmethod
    def _resize_image(image, shape=None):
        """ Resize an image """
        factor = np.asarray(shape) / np.asarray(image.shape[:2])
        if len(image.shape) > 2:
            factor = np.concatenate((factor, [1.] * len(image.shape[2:])))
        new_image = scipy.ndimage.interpolation.zoom(image, factor, order=3)
        return new_image

    @staticmethod
    def _rotate_image(image, angle, preserve_shape, **kwargs):
        """ Rotate an image """
        kwargs['reshape'] = not preserve_shape
        new_image = scipy.ndimage.interpolation.rotate(image, angle, **kwargs)
        return new_image.copy()

    @staticmethod
    def _flip_image(image, mode):
        if mode == 'lr':
            return image[:, ::-1]
        elif mode == 'ud':
            return image[::-1]

    @staticmethod
    def _pad(image, **kwargs):
        return np.pad(image, **kwargs)

    @staticmethod
    def _invert(image, channels):
        inv_multiplier = np.zeros(image.shape[-1], dtype=np.int16)
        inv_multiplier[np.asarray(channels)] = 255
        return np.abs(np.ones(image.shape, dtype=np.int16)*inv_multiplier - image.astype(np.int16)).astype(np.uint8)



#------------------------------------------------------------------------------------------------


class ImagesPILBatch(BaseImagesBatch):
    """ Batch class for 2D images in PIL format """
    @staticmethod
    def get_image_shape(image):
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

    def _resize_image(self, ix, components='images', shape=None, **kwargs):
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

    def _rotate_image(self, ix, components='images', angle=0, preserve_shape=True, **kwargs):
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
