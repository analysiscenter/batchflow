""" Contains Batch classes for images """
import os
from numbers import Number
from functools import wraps

try:
    from imageio import imread
except ImportError:
    from scipy.ndimage import imread

import numpy as np
#pylint: disable=ungrouped-imports
import scipy.ndimage
from scipy.misc import imsave, imresize

from .batch import Batch
from .decorators import action, inbatch_parallel


def get_scipy_transforms():
    """ Returns ``dict`` {'function_name' : function} of functions from scipy.ndimage.

    Function is included if it has 'input : ndarray' or 'input : array_like' in its docstring.
    """

    scipy_transformations = {}
    hooks = ['input : ndarray', 'input : array_like']
    for function_name in scipy.ndimage.__dict__['__all__']:
        function = getattr(scipy.ndimage, function_name)
        doc = getattr(function, '__doc__')
        if doc is not None and (hooks[0] in doc or hooks[1] in doc):
            scipy_transformations[function_name] = function
    return scipy_transformations


def transform_actions(prefix='', suffix='', wrapper=None):
    """ Transforms classmethods that have names like <prefix><name><suffix> to pipeline's actions executed in parallel.

    First, it finds all *class methods* which names have the form <prefix><method_name><suffix>
    (ignores those that start and end with '__').

    Then, all found classmethods are decorated through ``wrapper`` and resulting
    methods are added to the class with the names of the form <method_name>.

    Parameters
    ----------
    prefix : str
    suffix : str
    wrapper : str
        name of the wrapper inside ``Batch`` class

    Examples
    --------
    >>> from dataset import ImagesBatch
    >>> @transform_actions(prefix='_', suffix='_')
    ... class MyImagesBatch(ImagesBatch):
    ...     @classmethod
    ...     def _flip_(cls, image):
    ...             return image[:,::-1]

    Note that if you only want to redefine actions you still have to decorate your class.

    >>> from dataset.opensets import CIFAR10
    >>> dataset = CIFAR10(batch_class=MyImagesBatch, path='.')

    Now dataset.pipeline has flip action that operates as described above.
    If you want to apply an action with some probability, then specify ``p`` parameter:

    >>> from dataset import Pipeline
    >>> pipeline = (Pipeline()
    ...                 ...preprocessing...
    ...                 .flip(p=0.7)
    ...                 ...postprocessing...

    Now each image will be flipped with probability 0.7.
    """
    def _decorator(cls):
        for method_name, method in cls.__dict__.copy().items():
            if method_name.startswith(prefix) and method_name.endswith(suffix) and\
               not method_name.startswith('__') and not method_name.endswith('__'):
                def _wrapper():
                    #pylint: disable=cell-var-from-loop
                    wrapped_method = method
                    @wraps(wrapped_method)
                    def _func(self, *args, src='images', dst='images', **kwargs):
                        return getattr(cls, wrapper)(self, wrapped_method, src=src, dst=dst,
                                                     use_self=True, *args, **kwargs)
                    return _func
                name_slice = slice(len(prefix), -len(suffix))
                wrapped_method_name = method_name[name_slice]
                setattr(cls, wrapped_method_name, action(_wrapper()))
        return cls
    return _decorator


def add_methods(transformations=None, prefix='_', suffix='_'):
    """ Bounds given functions to a decorated class

    All bounded methods' names will be extended with ``prefix`` and ``suffix``.
    For example, if ``transformations``={'method_name': method}, ``suffix``='_all' and ``prefix``='_'
    then a decorated class will have '_method_name_all' method.

    Parameters
    ----------
    transformations : dict
        dict of the form {'method_name' : function_to_bound} -- functions to bound to a class
    prefix : str
    suffix : str
    """

    def _decorator(cls):
        for func_name, func in transformations.items():
            def _method_decorator():
                #pylint: disable=cell-var-from-loop
                added_func = func
                @wraps(added_func)
                def _method(self, *args, **kwargs):
                    _ = self
                    return added_func(*args, **kwargs)
                return _method
            method_name = ''.join((prefix, func_name, suffix))
            added_method = _method_decorator()
            setattr(cls, method_name, added_method)
        return cls
    return _decorator


class BaseImagesBatch(Batch):
    """ Batch class for 2D images """
    components = "images", "labels"
    formats = set(['jpg', 'png', 'jpeg'])

    def _make_path(self, path, ix):
        """ Compose path.

        Parameters
        ----------
        path : str, None
        ix : str
            element's index (filename)

        Returns
        -------
        path : str
            Joined path if path is not None else element's path specified in the batch's index.
        """

        return self.index.get_fullpath(ix) if path is None else os.path.join(path, ix)

    def _load_image(self, ix, src=None, fmt='image', dst="images"):
        """ Loads image.

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, None
            path to the folder with an image. If src is None then it is determined from the index.
        dst : str
            Component to write images to.
        fmt : str
            Format of the an image

        Raises
        ------
        NotImplementedError
            If this method is not defined in a child class
        """

        _ = self, ix, src, dst, fmt
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def load(self, *args, src=None, fmt='image', components=None, **kwargs):
        """ Load data.

        .. note:: if `fmt='images'` than ``components`` must be a single component (str).
        .. note:: All parameters must be named only.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to download.
        components : str, sequence
            components to download.
        """

        if fmt.lower() in BaseImagesBatch.formats or fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=components)
        return super().load(src=src, fmt=fmt, components=components, *args, **kwargs)

    def _dump_image(self, ix, src='images', dst=None, fmt=None):
        """ Saves image to dst.

        .. note:: Please note that ``src`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str
            Component to get images from.
        dst : str
            Folder where to dump. If dst is None then it is determined from index.

        Raises
        ------
        NotImplementedError
            If this method is not defined in a child class
        """

        _ = self, ix, src, dst, fmt
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def dump(self, *args, dst=None, fmt=None, components="images", **kwargs):
        """ Dump data.

        .. note:: If `fmt='images'` than ``dst`` must be a single component (str).

        .. note:: All parameters must be named only.

        Parameters
        ----------
        dst : str, None
            Path to the folder where to dump. If dst is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to save.
        components : str, sequence
            Components to save.

        Returns
        -------
        self
        """

        if fmt == 'image':
            return self._dump_image(components, dst, fmt=None,)
        return super().dump(dst=dst, fmt=fmt, components=components, *args, **kwargs)


@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')
@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
@add_methods(transformations={**get_scipy_transforms(),
                              'pad': np.pad,
                              'resize': imresize}, prefix='_', suffix='_')
class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images.

    Images are stored as numpy arrays (N, H, W, C).
    """

    @classmethod
    def _get_image_shape(cls, image):
        return image.shape[:2]

    @property
    def image_shape(self):
        """: tuple - shape of the image"""
        if isinstance(self.images.dtype, object):
            _, shapes_count = np.unique([image.shape for image in self.images], return_counts=True, axis=0)
            if len(shapes_count) == 1:
                return self.images.shape[1:]
            else:
                raise RuntimeError('Images have different shapes')
        return self.images.shape[1:]

    @inbatch_parallel(init='indices', post='_assemble', target='t')
    def _load_image(self, ix, src=None, fmt="image", dst="images"):
        """ Loads image

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, None
            Path to the folder with an image. If src is None then it is determined from the index.
        dst : str
            Component to write images to.
        fmt : str
            Format of an image.

        Returns
        -------
        self
        """

        if ix.rfind('.') == -1:
            if fmt == "image":
                for image_format in BaseImagesBatch.formats:
                    try:
                        return imread(self._make_path(src, ix+'.'+image_format))
                    except OSError:
                        pass
                raise RuntimeError("Unknown image format")
            return imread(self._make_path(src, ix+'.'+fmt))
        return imread(self._make_path(src, ix))


    @inbatch_parallel(init='indices')
    def _dump_image(self, ix, src='images', dst=None, fmt=None):
        """ Saves image to dst.

        .. note:: Please note that ``src`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str
            Component to get images from.
        dst : str
            Folder where to dump. If dst is None then it is determined from index.

        Returns
        -------
        self
        """

        if ix.rfind('.') == -1:
            if fmt == "image":

                raise RuntimeError("Unknown image format")
            return imread(self._make_path(src, ix+'.'+fmt))
        return imread(self._make_path(src, ix))

        imsave(self._make_path(dst, ix+'.'+fmt), self.get(ix, src))

    def _assemble_component(self, result, *args, component='images', **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        result : sequence, array_like
            Results after inbatch_parallel.
        component : str
            component to assemble
        preserve_shape : bool
            If True then all images are cropped from the top left corner to have similar shapes.
            Shape is chosen to be minimal among given images.
        """

        try:
            new_images = np.stack(result)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                preserve_shape = kwargs.get('preserve_shape', False)
                if preserve_shape:
                    min_shape = np.array([self._get_image_shape(x) for x in result]).min(axis=0)
                    result = [arr[:min_shape[0], :min_shape[1]].copy() for arr in result]
                    new_images = np.stack(result)
                else:
                    new_images = np.array(result, dtype=object)
            else:
                raise e
        setattr(self, component, new_images)

    def _calc_origin(self, image_shape, origin, background_shape):
        """ Calculate coordinate of the input image with respect to the background.

        Parameters
        ----------
        image_shape : sequence
            shape of the input image.
        origin : array_like, sequence, {'center', 'top_left', 'random'}
            Position of the input image with respect to the background.
            - 'center' - place the center of the input image on the center of the background and crop
                         the input image accordingly.
            - 'top_left' - place the upper-left corner of the input image on the upper-left of the background
                           and crop the input image accordingly.
            - 'random' - place the upper-left corner of the input image on the randomly sampled position
                         in the background. Position is sampled uniformly such that there is no need for cropping.
            - other - place the upper-left corner of the input image on the given position in the background.
        background_shape : sequence
            shape of the background image.

        Returns
        -------
        sequence : calculated origin in the form (row, column)
        """

        if isinstance(origin, str):
            if origin == 'top_left':
                origin = 0, 0
            elif origin == 'center':
                origin = np.maximum(0, np.asarray(background_shape) - image_shape) // 2
            elif origin == 'random':
                origin = (np.random.randint(background_shape[0]-image_shape[0]+1),
                          np.random.randint(background_shape[1]-image_shape[1]+1))
        return np.asarray(origin, dtype=np.int)

    def _scale_(self, image, factor, preserve_shape=False, origin='center'):
        """ Scale the content of each image in the batch.

        Resulting shape is obtained as original_shape * factor.

        Parameters
        -----------
        factor : float, sequence
            resulting shape is obtained as original_shape * factor
            - float - scale all axes with the given factor
            - sequence (factor_1, factort_2, ...) - scale each axis with the given factor separately

        preserve_shape : bool
            whether to preserve the shape of the image after scaling

        origin : {'center', 'top_left', 'random'}, sequence
            Relevant only if `preserve_shape` is True.
            Position of the scaled image with respect to the original one's shape.
            - 'center' - place the center of the rescaled image on the center of the original one and crop
                         the rescaled image accordingly
            - 'top_left' - place the upper-left corner of the rescaled image on the upper-left of the original one
                           and crop the rescaled image accordingly
            - 'random' - place the upper-left corner of the rescaled image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - sequence - place the upper-left corner of the rescaled image on the given position in the original one.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        Returns
        -------
        self
        """

        if np.any(np.asarray(factor) <= 0):
            raise ValueError("factor must be greater than 0")
        rescaled_shape = np.ceil(np.array(self._get_image_shape(image)) * factor).astype(np.int16)
        rescaled_image = self._resize_(image, rescaled_shape)
        if preserve_shape:
            rescaled_image = self._preserve_shape(image, rescaled_image, origin)
        return rescaled_image

    def _crop_(self, image, origin, shape):
        """ Crop an image.

        Extract image data from the window of the size given by `shape` and placed at `origin`.

        Parameters
        ----------
        image : np.ndarray
        origin : sequence, str
            Upper-left corner of the cropping box. Can be one of:
            - sequence - corner's coordinates in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         an image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position
        shape : sequence
            - sequence - crop size in the form of (rows, columns)
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        image_shape = self._get_image_shape(image)
        origin = self._calc_origin(shape, origin, image_shape)
        if np.all(origin + shape > image_shape):
            shape = image_shape - origin

        row_slice = slice(origin[0], origin[0] + shape[0])
        column_slice = slice(origin[1], origin[1] + shape[1])
        return image[row_slice, column_slice].copy()

    def _put_on_background_(self, image, background, origin):
        """ Put an image on a background at given origin

        Parameters
        ----------
        background : np.ndarray
        origin : sequence, str
            Upper-left corner of the cropping box. Can be one of:
            - sequence - corner's coordinates in the form of (row, column).
            - 'top_left' - crop an image such that upper-left corners of an image and the cropping box coincide.
            - 'center' - crop an image such that centers of an image and the cropping box coincide.
            - 'random' - place the upper-left corner of the cropping box at a random position.

        Returns
        -------
        self
        """

        image_shape = self._get_image_shape(image)
        background_shape = self._get_image_shape(background)
        origin = self._calc_origin(image_shape, origin, background_shape)
        image = self._crop_(image, 'top_left', np.asarray(background_shape) - origin).copy()

        slice_rows = slice(origin[0], origin[0]+image_shape[0])
        slice_columns = slice(origin[1], origin[1]+image_shape[1])

        new_image = background.copy()
        new_image[slice_rows, slice_columns] = image
        return new_image

    def _preserve_shape(self, original_image, transformed_image, origin='center'):
        """ Change the transformed image's shape by cropping and adding empty pixels to fit the shape of original image.

        Parameters
        ----------
        original_image : np.ndarray
        transformed_image : np.ndarray
        origin : {'center', 'top_left', 'random'}, sequence
            Position of the transformed image with respect to the original one's shape.
            - 'center' - place the center of the transformed image on the center of the original one and crop
                         the transformed image accordingly.
            - 'top_left' - place the upper-left corner of the transformed image on the upper-left of the original one
                           and crop the transformed image accordingly.
            - 'random' - place the upper-left corner of the transformed image on the randomly sampled position
                         in the original one. Position is sampled uniformly such that there is no need for cropping.
            - sequence - place the upper-left corner of the transformed image on the given position in the original one.

        Returns
        -------
        np.ndarray : image after described actions
        """

        return self._put_on_background_(self._crop_(transformed_image,
                                                    'top_left' if origin != 'center' else 'center',
                                                    self._get_image_shape(original_image)),
                                        np.zeros(original_image.shape, dtype=np.uint8),
                                        origin)

    def _flip_all(self, images=None, indices=None, mode='lr'):
        """ Flip images in the batch.

        Parameters
        ----------
        mode : {'lr', 'ud'}
            - 'lr' - apply the left/right flip
            - 'ud' - apply the upside/down flip
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        images = images.copy()
        if mode == 'lr':
            images[indices] = images[indices, :, ::-1]
        elif mode == 'ud':
            images[indices] = images[indices, ::-1]
        return images

    def _invert_(self, image, channels='all'):
        """ Invert givn channels.

        Parameters
        ----------
        channels : int, sequence
            Indices of the channels to invert.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        image = image.copy()
        if channels == 'all':
            channels = list(range(image.shape[-1]))
        max_intencity = 255 if np.issubdtype(image.dtype, np.integer) else 1.
        image[..., channels] = max_intencity - image[..., channels]
        return image

    def _salt_(self, image, p_noise=.015, color=255, size=(1, 1)):
        """ Set random pixel on image to givan value.

        Every pixel will be set to ``color`` value with probability ``p_noise``.

        Parameters
        ----------
        p_noise : float
            Probability of salting a pixel.
        color : float, int, sequence, callable
            Color's value.
            - int, float, sequence -- value of color
            - callable -- color is sampled for every chosen pixel (rules are the same as for int, float and sequence)
        size : int, sequence of int, callable
            Size of salt
            - int -- square salt with side ``size``
            - sequence -- recangular salt in the form (row, columns)
            - callable -- size is sampled for every chosen pixel (rules are the same as for int and sequence)
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        image = image.copy()
        mask_size = np.asarray(self._get_image_shape(image))
        mask_salt = np.random.binomial(1, p_noise, size=mask_size).astype(bool)
        if isinstance(size, (tuple, int)) and (size == (1, 1) or size == 1) and not callable(color):
            image[mask_salt] = color
        else:
            size_lambda = size if callable(size) else lambda: size
            color_lambda = color if callable(color) else lambda: color
            mask_salt = np.where(mask_salt)
            for i in range(len(mask_salt[0])):
                current_size = size_lambda()
                current_size = (current_size, current_size) if isinstance(current_size, Number) else current_size
                left_top = np.asarray((mask_salt[0][i], mask_salt[1][i]))
                right_bottom = np.minimum(left_top + current_size, self._get_image_shape(image))
                image[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]] = color_lambda()
        return image

    def _threshold_(self, image, low=0., high=1., dtype=np.uint8):
        """ Truncate image's pixels.

        Parameters
        ----------
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        dtype : np.dtype
            dtype of truncated images.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        image = image.copy()
        if isinstance(low, Number):
            image[image < low] = low
        else:
            if len(low) != image.shape[-1]:
                raise RuntimeError("``len(low)`` must coincide with the number of channels")
            for channel, low_channel in enumerate(low):
                pixels_to_truncate = image[..., channel] < low_channel
                image[..., channel][pixels_to_truncate] = low_channel
        if isinstance(high, Number):
            image[image > high] = high
        else:
            if len(high) != image.shape[-1]:
                raise RuntimeError("``len(high)`` must coincide with the number of channels")

            for channel, high_channel in enumerate(high):
                pixels_to_truncate = image[..., channel] > high_channel
                image[..., channel][pixels_to_truncate] = high_channel
        return image.astype(dtype)

    def _multiply_(self, image, multiplier=1., low=0., high=1., preserve_type=True):
        """ Multiply each pixel by the given multiplier.

        Parameters
        ----------
        multiplier : float, sequence
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        dtype = image.dtype if preserve_type else np.float
        return self._threshold_(multiplier * image.astype(np.float), low, high, dtype)

    def _add_(self, image, term=0., low=0., high=1., preserve_type=True):
        """ Add term to each pixel.

        Parameters
        ----------
        term : float, sequence
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        dtype = image.dtype if preserve_type else np.float
        return self._threshold_(term + image.astype(np.float), low, high, dtype)

    def _to_greyscale_all(self, images, indices, keepdims=True):
        """ Set image's pixels to their mean among all channels

        .. note:: Images' shape must provide last axis for channels

        Parameters
        ----------
        keepdims : bool
            Whether to preserve the number of channels
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """
        _ = indices
        return images.mean(axis=-1, keepdims=keepdims).astype(images.dtype)

    def _posterize_all(self, images, indices, colors_number=3):
        """ Posterizes images.

        More concretely, it quantizes pixels' values so that they have``colors_number`` colours.

        Parameters
        ----------
        colors : int
            Number of colours.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        images = images.copy()
        dtype = images.dtype
        max_bin = 256 if np.issubdtype(dtype, np.integer) else 1.0001
        max_intencity = 255 if np.issubdtype(dtype, np.integer) else 1.

        bins = np.linspace(0, max_bin, colors_number+1)
        color_indices = np.digitize(images[indices], bins) - 1
        colors = np.linspace(0, max_intencity, colors_number)

        images[indices] = colors[color_indices]
        return images

    def _fill_crop_(self, image, origins, shapes, colors):
        """ Fills given areas ('crops') with color

        .. note:: It is assumed that ``origins``, ``shapes`` and ``colors`` have the same length.

        Parameters
        ----------
        origins : sequence
            Every element of this sequence is the upper-left corner of a filled box. Can be one of:
            - sequence - corner's coordinates in the form of (row, column).
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the filled box coincide.
            - 'center' - crop an image such that centers of
                         an image and the filled box coincide.
            - 'random' - place the upper-left corner of the filled box at a random position.
        shapes : sequence
            Every element of this sequence is the shape of a filled box. Can be one of:
            - sequence - crop size in the form of (rows, columns)
            - int - shape has squared form
        colors : sequence
            Every element of this sequence is the colour of a filled box. Can be one of:
            - sequence - (r,g,b) form
            - number - grayscale
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        def _get_shape(shape):
            return (shape, shape) if isinstance(shape, Number) else shape

        def _get_origin(shape, origin):
            if isinstance(origin, str):
                origin = self._calc_origin(shape, origin, image_shape)
            return origin
        image = image.copy()
        image_shape = self._get_image_shape(image)
        for origin, shape, color in zip(origins, shapes, colors):
            shape = _get_shape(shape)
            origin = _get_origin(shape, origin)
            right_bottom = (min(origin[0] + shape[0], image_shape[0]),
                            min(origin[1] + shape[1], image_shape[1]))
            image[origin[0]:right_bottom[0], origin[1]:right_bottom[1]] = color

        return image

    def _assemble_patches(self, patches, *args, dst, **kwargs):
        """ Assembles patches after parallel execution.

        Parameters
        ----------
        patches : sequence
            Patches to gather. pathces.shape must be like (batch.size, patches_i, patch_height, patch_width, n_channels)
        dst : str
            Component to put patches in.
        """

        _ = args, kwargs
        new_items = np.concatenate(patches)
        setattr(self, dst, new_items)

    @action
    @inbatch_parallel(init='indices', post='_assemble_patches')
    def split_to_patches(self, ix, patch_shape, stride=1, droplast=False, src='images', dst=None):
        """ Splits image to patches.

        Small images with the same shape (``patch_shape``) are cropped from the original one with stride ``stride``.

        Parameters
        ----------
        patch_shape : int, sequence
            Patch's shape in the from (rows, columns). If int is given then patches have square shape.
        stride : int, square
            Step of the moving window from which patches are cropped. If int is given then the window has square shape.
        droplast : bool
            Whether to drop patches whose window covers area out of the image.
            If False is passed then these patches are cropped from the edge of an image. See more in tutorials.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        _ = dst
        image = self.get(ix, src)
        image_shape = self._get_image_shape(image)
        stride = (stride, stride) if isinstance(stride, Number) else stride
        patch_shape = (patch_shape, patch_shape) if isinstance(patch_shape, Number) else patch_shape
        patches = []

        def _iterate_columns(row_from, row_to):
            column = 0
            while column < image_shape[1]-patch_shape[1]+1:
                patches.append(image[row_from:row_to, column:column+patch_shape[1]])
                column += stride[1]
            if not droplast and column + patch_shape[1] != image_shape[1]:
                patches.append(image[row_from:row_to, image_shape[1]-patch_shape[1]:image_shape[1]])

        row = 0
        while row < image_shape[0]-patch_shape[0]+1:
            _iterate_columns(row, row+patch_shape[0])
            row += stride[0]
        if not droplast and row + patch_shape[0] != image_shape[0]:
            _iterate_columns(image_shape[0]-patch_shape[0], image_shape[0])

        return np.stack(patches)

    def _additive_noise_all(self, images, indices, noise, low=0, high=1.):
        """ Add additive noise to images.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """

        images = images.copy()
        noisy_images = images[indices]
        images[indices] = self._threshold_(images[indices]+noise(size=noisy_images.shape),
                                           low, high, dtype=images.dtype)
        return images


    def _multiplicative_noise_all(self, images, indices, noise, low=0, high=1.):
        """ Add multiplicativa noise to images.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Returns
        -------
        self
        """
        images = images.copy()
        noisy_images = images[indices]
        images[indices] = self._threshold_(images[indices]*noise(size=noisy_images.shape),
                                           low, high, dtype=images.dtype)
        return images
