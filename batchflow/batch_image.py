""" Contains Batch classes for images """
import os
import warnings
from numbers import Number

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageChops
import PIL.ImageFilter
import PIL.ImageEnhance
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .batch import Batch
from .decorators import action, apply_parallel, inbatch_parallel
from .dsindex import FilesIndex


class BaseImagesBatch(Batch):
    """ Batch class for 2D images.

    Note, that if any class method is wrapped with `@apply_parallel` decorator
    than for inner calls (i.e. from other class methods) should be used version
    of desired method with underscores. (For example, if there is a decorated
    `method` than you need to call `_method_` from inside of `other_method`).
    Same is applicable for all child classes of :class:`batch.Batch`.
    """
    components = "images", "labels", "masks"
    # Class-specific defaults for :meth:`.Batch.apply_parallel`
    apply_defaults = dict(target='for',
                          post='_assemble',
                          src='images',
                          dst='images',
                          )

    def _make_path(self, ix, src=None):
        """ Compose path.

        Parameters
        ----------
        ix : str
            element's index (filename)
        src : str
            Path to folder with images. Used if `self.index` is not `FilesIndex`.

        Returns
        -------
        path : str
            Full path to an element.
        """

        if isinstance(src, FilesIndex):
            path = src.get_fullpath(ix)
        elif isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(ix)
        else:
            path = os.path.join(src, str(ix))
        return path

    def _load_image(self, ix, src=None, fmt=None, dst="images"):
        """ Loads image.

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
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
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data.

        .. note:: if `fmt='images'` than ``components`` must be a single component (str).
        .. note:: All parameters must be named only.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from the index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to download.
        dst : str, sequence
            components to download.
        """
        if fmt == 'image':
            return self._load_image(src, fmt=fmt, dst=dst)
        return super().load(src=src, fmt=fmt, dst=dst, *args, **kwargs)


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
        ext: str
            Format to save images to.

        Returns
        -------
        self
        """
        if fmt == 'image':
            return self._dump_image(components, dst, fmt=kwargs.pop('ext'))
        return super().dump(dst=dst, fmt=fmt, components=components, *args, **kwargs)


class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images.

    Images are stored as numpy arrays of PIL.Image.

    PIL.Image has the following system of coordinates::

                           X
          0 -------------- >
          |
          |
          |  images's pixels
          |
          |
        Y v

    Pixel's position is defined as (x, y)

    Note, that if any class method is wrapped with `@apply_parallel` decorator
    than for inner calls (i.e. from other class methods) should be used version
    of desired method with underscores. (For example, if there is a decorated
    `method` than you need to call `_method_` from inside of `other_method`).
    Same is applicable for all child classes of :class:`batch.Batch`.
    """

    @classmethod
    def _get_image_shape(cls, image):
        if isinstance(image, PIL.Image.Image):
            return image.size
        return image.shape[:2]

    @property
    def image_shape(self):
        """: tuple - shape of the image"""
        _, shapes_count = np.unique([image.size for image in self.images], return_counts=True, axis=0)
        if len(shapes_count) == 1:
            if isinstance(self.images[0], PIL.Image.Image):
                return (*self.images[0].size, len(self.images[0].getbands()))
            return self.images[0].shape
        raise RuntimeError('Images have different shapes')

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_image(self, ix, src=None, fmt=None, dst="images"):
        """ Loads image

        .. note:: Please note that ``dst`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str, dataset.FilesIndex, None
            Path to the folder with an image. If src is None then it is determined from the index.
        dst : str
            Component to write images to.
        fmt : str
            Format of an image.
        """
        return PIL.Image.open(self._make_path(ix, src))

    @inbatch_parallel(init='indices')
    def _dump_image(self, ix, src='images', dst=None, fmt=None):
        """ Saves image to dst.

        .. note:: Please note that ``src`` must be ``str`` only, sequence is not allowed here.

        Parameters
        ----------
        src : str
            Component to get images from.
        dst : str
            Folder where to dump.
        fmt : str
            Format of saved image.
        """
        if dst is None:
            raise RuntimeError('You must specify `dst`')
        image = self.get(ix, src)
        ix = str(ix) + '.' + fmt if fmt is not None else str(ix)
        image.save(os.path.join(dst, ix))

    def _assemble_component(self, result, *args, component='images', **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        result : sequence, array_like
            Results after inbatch_parallel.
        component : str
            component to assemble
        """
        _ = args, kwargs
        if isinstance(result[0], PIL.Image.Image):
            setattr(self, component, np.asarray(result, dtype=object))
        else:
            try:
                setattr(self, component, np.stack(result))
            except ValueError:
                array_result = np.empty(len(result), dtype=object)
                array_result[:] = result
                setattr(self, component, array_result)

    @apply_parallel
    def to_pil(self, image, mode=None):
        """converts images in Batch to PIL format

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        """
        if isinstance(image, PIL.Image.Image):
            return image

        if mode is None:
            if len(image.shape) == 2:
                mode = 'L'
            elif len(image.shape) == 3:
                if image.shape[-1] == 3:
                    mode = 'RGB'
                elif image.shape[-1] == 1:
                    mode = 'L'
                    image = image[:, :, 0]
                elif image.shape[-1] == 2:
                    mode = 'LA'
                elif image.shape[-1] == 4:
                    mode = 'RGBA'
            else:
                raise ValueError('Unknown image type as image has', image.shape[-1], 'channels')
        elif mode == 'L' and len(image.shape) == 3:
            image = image[..., 0]
        return PIL.Image.fromarray(image, mode)

    def _calc_origin(self, image_shape, origin, background_shape):
        """ Calculate coordinate of the input image with respect to the background.

        Parameters
        ----------
        image_shape : sequence
            shape of the input image.
        origin : array_like, sequence, {'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random'}
            Position of the input image with respect to the background. Can be one of:
                - 'center' - place the center of the input image on the center of the background and crop
                  the input image accordingly.
                - 'top_left' - place the upper-left corner of the input image on the upper-left of the background
                  and crop the input image accordingly.
                - 'top_right' - crop an image such that upper-right corners of
                  an image and the cropping box coincide
                - 'bottom_left' - crop an image such that lower-left corners of
                  an image and the cropping box coincide
                - 'bottom_right' - crop an image such that lower-right corners of
                  an image and the cropping box coincide
                - 'random' - place the upper-left corner of the input image on the randomly sampled position
                  in the background. Position is sampled uniformly such that there is no need for cropping.
                - other - sequence of ints or sequence of floats in [0, 1) interval;
                  place the upper-left corner of the input image on the given position in the background.
                  If `origin` is a sequence of floats in [0, 1), it defines a relative position of
                  the origin in a valid region of image.

        background_shape : sequence
            shape of the background image.

        Returns
        -------
        sequence : calculated origin in the form (column, row)
        """
        if isinstance(origin, str):
            if origin == 'top_left':
                origin = 0, 0
            elif origin == 'top_right':
                origin = (background_shape[0]-image_shape[0]+1, 0)
            elif origin == 'bottom_left':
                origin = (0, background_shape[1]-image_shape[1]+1)
            elif origin == 'bottom_right':
                origin = (background_shape[0]-image_shape[0]+1,
                          background_shape[1]-image_shape[1]+1)
            elif origin == 'center':
                origin = np.maximum(0, np.asarray(background_shape) - image_shape) // 2
            elif origin == 'random':
                origin = (np.random.randint(background_shape[0]-image_shape[0]+1),
                          np.random.randint(background_shape[1]-image_shape[1]+1))
            else:
                raise ValueError("If string, origin should be one of ['center', 'top_left', 'top_right', "
                                 "'bottom_left', 'bottom_right', 'random']. Got '{}'.".format(origin))
        elif all(0 <= elem < 1 for elem in origin):
            region = ((background_shape[0]-image_shape[0]+1),
                      (background_shape[1]-image_shape[1]+1))
            origin = np.asarray(origin) * region
        elif not all(isinstance(elem, int) for elem in origin):
            raise ValueError('If not a string, origin should be either a sequence of ints or sequence of '
                             'floats in [0, 1) interval. Got {}'.format(origin))

        return np.asarray(origin, dtype=np.int)

    @apply_parallel
    def scale(self, image, factor, preserve_shape=False, origin='center', resample=0):
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

        origin : array-like, {'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'random'}
            Relevant only if `preserve_shape` is True.
            If `scale` < 1, defines position of the scaled image with respect to the original one's shape.
            If `scale` > 1, defines position of cropping box.

            Can be one of:

            - 'center' - place the center of the input image on the center of the background and crop
              the input image accordingly.
            - 'top_left' - place the upper-left corner of the input image on the upper-left of the background
              and crop the input image accordingly.
            - 'top_right' - crop an image such that upper-right corners of
              an image and the cropping box coincide
            - 'bottom_left' - crop an image such that lower-left corners of
              an image and the cropping box coincide
            - 'bottom_right' - crop an image such that lower-right corners of
              an image and the cropping box coincide
            - 'random' - place the upper-left corner of the input image on the randomly sampled position
              in the background. Position is sampled uniformly such that there is no need for cropping.
            - array_like - sequence of ints or sequence of floats in [0, 1) interval;
              place the upper-left corner of the input image on the given position in the background.
              If `origin` is a sequence of floats in [0, 1), it defines a relative position
              of the origin in a valid region of image.

        resample: int
            Parameter passed to PIL.Image.resize. Interpolation order
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Notes
        -----
        Using 'random' option for origin with `src` as list with multiple elements will not result in same crop for each
        element, as origin will be sampled independently for each `src` element.
        To randomly sample same origin for a number of components, use `R` named expression for `origin` argument.

        Returns
        -------
        self
        """
        original_shape = self._get_image_shape(image)
        rescaled_shape = list(np.int32(np.ceil(np.asarray(original_shape)*factor)))
        rescaled_image = image.resize(rescaled_shape, resample=resample)
        if preserve_shape:
            rescaled_image = self._preserve_shape(original_shape, rescaled_image, origin)
        return rescaled_image

    @apply_parallel
    def crop(self, image, origin, shape, crop_boundaries=False):
        """ Crop an image.

        Extract image data from the window of the size given by `shape` and placed at `origin`.

        Parameters
        ----------
        origin : sequence, str
            Location of the cropping box. See :meth:`.ImagesBatch._calc_origin` for details.
        shape : sequence
            crop size in the form of (rows, columns)
        crop_boundaries : bool
            If `True` then crop is got only from image's area. Shape of the crop might diverge with the passed one
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Notes
        -----
        Using 'random' origin with `src` as list with multiple elements will not result in same crop for each
        element, as origin will be sampled independently for each `src` element.
        To randomly sample same origin for a number of components, use `R` named expression for `origin` argument.
        """
        origin = self._calc_origin(shape, origin, image.size)
        right_bottom = origin + shape

        if crop_boundaries:
            out_of_boundaries = origin < 0
            origin[out_of_boundaries] = 0

            image_shape = np.asarray(image.size)
            out_of_boundaries = right_bottom > image_shape
            right_bottom[out_of_boundaries] = image_shape[out_of_boundaries]

        return image.crop((*origin, *right_bottom))

    @apply_parallel
    def put_on_background(self, image, background, origin, mask=None):
        """ Put an image on a background at given origin

        Parameters
        ----------
        background : PIL.Image, np.ndarray of np.uint8
            Blank background to put image on.
        origin : sequence, str
            Location of the cropping box. See :meth:`.ImagesBatch._calc_origin` for details.
        mask : None, PIL.Image, np.ndarray of np.uint8
            mask passed to PIL.Image.paste

        Notes
        -----
        Using 'random' origin with `src` as list with multiple elements will not result in same crop for each
        element, as origin will be sampled independently for each `src` element.
        To randomly sample same origin for a number of components, use `R` named expression for `origin` argument.
        """
        if not isinstance(background, PIL.Image.Image):
            background = PIL.Image.fromarray(background)
        else:
            background = background.copy()

        if not isinstance(mask, PIL.Image.Image):
            mask = PIL.Image.fromarray(mask) if mask is not None else None

        origin = list(self._calc_origin(self._get_image_shape(image), origin,
                                        self._get_image_shape(background)))

        background.paste(image, origin, mask)

        return background

    def _preserve_shape(self, original_shape, transformed_image, origin='center'):
        """ Change the transformed image's shape by cropping and adding empty pixels to fit the shape of original image.

        Parameters
        ----------
        original_shape : sequence
        transformed_image : np.ndarray
        input_origin : array-like, {'center', 'top_left', 'random'}
            Position of the scaled image with respect to the original one's shape.
            - 'center' - place the center of the input image on the center of the background and crop
                         the input image accordingly.
            - 'top_left' - place the upper-left corner of the input image on the upper-left of the background
                           and crop the input image accordingly.
            - 'top_right' - crop an image such that upper-right corners of
                            an image and the cropping box coincide
            - 'bottom_left' - crop an image such that lower-left corners of
                              an image and the cropping box coincide
            - 'bottom_right' - crop an image such that lower-right corners of
                               an image and the cropping box coincide
            - 'random' - place the upper-left corner of the input image on the randomly sampled position
                         in the background. Position is sampled uniformly such that there is no need for cropping.
            - array_like - sequence of ints or sequence of floats in [0, 1) interval;
                           place the upper-left corner of the input image on the given position in the background.
                           If `origin` is a sequence of floats in [0, 1), it defines a relative position
                           of the origin in a valid region of image.
        crop_origin: array-like, {'center', 'top_left', 'random'}
            Position of crop from transformed image.
            Has same values as `input_origin`.

        Returns
        -------
        np.ndarray : image after described actions
        """
        transformed_shape = self._get_image_shape(transformed_image)
        if np.any(np.array(transformed_shape) < np.array(original_shape)):
            n_channels = len(transformed_image.getbands())
            if n_channels == 1:
                background = np.zeros(original_shape, dtype=np.uint8)
            else:
                background = np.zeros((*original_shape, n_channels), dtype=np.uint8)
            return self._put_on_background_(transformed_image, background, origin)
        return self._crop_(transformed_image, origin, original_shape, True)

    @apply_parallel
    def filter(self, image, mode, *args, **kwargs):
        """ Filters an image. Calls ``image.filter(getattr(PIL.ImageFilter, mode)(*args, **kwargs))``.

        For more details see `ImageFilter <http://pillow.readthedocs.io/en/stable/reference/ImageFilter.html>_`.

        Parameters
        ----------
        mode : str
            Name of the filter.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.filter(getattr(PIL.ImageFilter, mode)(*args, **kwargs))

    @apply_parallel
    def transform(self, image, *args, **kwargs):
        """ Calls ``image.transform(*args, **kwargs)``.

        For more information see
        `<http://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform>_`.

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        size = kwargs.pop('size', self._get_image_shape(image))
        return image.transform(*args, size=size, **kwargs)

    @apply_parallel
    def resize(self, image, size, *args, **kwargs):
        """ Calls ``image.resize(*args, **kwargs)``.

        For more details see `<https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize>_`.

        Parameters
        ----------
        size : tuple
            the resulting size of the image. If one of the components of tuple is None,
            corresponding dimension will be proportionally resized.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if size[0] is None and size[1] is None:
            raise ValueError('At least one component of the parameter "size" must be a number.')
        if size[0] is None:
            new_size = (int(image.size[0] * size[1] / image.size[1]), size[1])
        elif size[1] is None:
            new_size = (size[0], int(image.size[1] * size[0] / image.size[0]))
        else:
            new_size = size

        return image.resize(new_size, *args, **kwargs)

    @apply_parallel
    def shift(self, image, offset, mode='const'):
        """ Shifts an image.

        Parameters
        ----------
        offset : (Number, Number)
        mode : {'const', 'wrap'}
            How to fill borders
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if mode == 'const':
            image = image.transform(size=image.size,
                                    method=PIL.Image.AFFINE,
                                    data=(1, 0, -offset[0], 0, 1, -offset[1]))
        elif mode == 'wrap':
            image = PIL.ImageChops.offset(image, *offset)
        else:
            raise ValueError("mode must be one of ['const', 'wrap']")
        return image

    @apply_parallel
    def pad(self, image, *args, **kwargs):
        """ Calls ``PIL.ImageOps.expand``.

        For more details see `<http://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.expand>`_.

        Parameters
        ----------
        offset : sequence
            Size of the borders in pixels. The order is (left, top, right, bottom).
        mode : {'const', 'wrap'}
            Filling mode
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return PIL.ImageOps.expand(image, *args, **kwargs)

    @apply_parallel
    def rotate(self, image, *args, **kwargs):
        """ Rotates an image.

            kwargs are passed to PIL.Image.rotate

        Parameters
        ----------
        angle: Number
            In degrees counter clockwise.
        resample: int
            Interpolation order
        expand: bool
            Whether to expand the output to hold the whole image. Default is False.
        center: (Number, Number)
            Center of rotation. Default is the center of the image.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.rotate(*args, **kwargs)

    @apply_parallel
    def flip(self, image, mode='lr'):
        """ Flips image.

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
        """
        if mode == 'lr':
            return PIL.ImageOps.mirror(image)
        return PIL.ImageOps.flip(image)

    @apply_parallel
    def invert(self, image, channels='all'):
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
        """
        if channels == 'all':
            image = PIL.ImageChops.invert(image)
        else:
            bands = list(image.split())
            channels = (channels,) if isinstance(channels, Number) else channels
            for channel in channels:
                bands[channel] = PIL.ImageChops.invert(bands[channel])
            image = PIL.Image.merge('RGB', bands)
        return image

    @apply_parallel
    def salt(self, image, p_noise=.015, color=255, size=(1, 1)):
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
        """
        mask_size = np.asarray(self._get_image_shape(image))
        mask_salt = np.random.binomial(1, p_noise, size=mask_size).astype(bool)
        image = np.array(image)
        if isinstance(size, (tuple, int)) and size in [1, (1, 1)] and not callable(color):
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

        return PIL.Image.fromarray(image)

    @apply_parallel
    def clip(self, image, low=0, high=255):
        """ Truncate image's pixels.

        Parameters
        ----------
        low : int, float, sequence
            Actual pixel's value is equal max(value, low). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        high : int, float, sequence
            Actual pixel's value is equal min(value, high). If sequence is given, then its length must coincide
            with the number of channels in an image and each channel is thresholded separately
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if isinstance(low, Number):
            low = tuple([low]*3)
        if isinstance(high, Number):
            high = tuple([high]*3)

        high = PIL.Image.new('RGB', image.size, high)
        low = PIL.Image.new('RGB', image.size, low)
        return PIL.ImageChops.lighter(PIL.ImageChops.darker(image, high), low)

    @apply_parallel
    def enhance(self, image, layout='hcbs', factor=(1, 1, 1, 1)):
        """ Apply enhancements from PIL.ImageEnhance to the image.

        Parameters
        ----------
        layout : str
            defines layout of operations, default is `hcbs`:
            h - color
            c - contrast
            b - brightness
            s - sharpness

        factor : float or tuple of float
            factor of enhancement for each operation listed in `layout`.
        """
        enhancements = {
            'h': 'Color',
            'c': 'Contrast',
            'b': 'Brightness',
            's': 'Sharpness'
        }

        if isinstance(factor, float):
            factor = (factor,) * len(layout)
        if len(layout) != len(factor):
            raise ValueError("'layout' and 'factor' should be of same length!")

        for alias, multiplier in zip(layout, factor):
            enhancement = enhancements.get(alias)
            if enhancement is None:
                raise ValueError('Unknown enhancement alias: ', alias)
            image = getattr(PIL.ImageEnhance, enhancement)(image).enhance(multiplier)

        return image

    @apply_parallel
    def multiply(self, image, multiplier=1., clip=False, preserve_type=False):
        """ Multiply each pixel by the given multiplier.

        Parameters
        ----------
        multiplier : float, sequence
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        multiplier = np.float32(multiplier)
        if isinstance(image, PIL.Image.Image):
            if preserve_type is False:
                warnings.warn("Note that some info might be lost during `multiply` transformation since PIL.image "
                              "stores data as `np.uint8`. To suppress this warning, use `preserve_type=True` or "
                              "consider using `to_array` action before multiplication.")
            return PIL.Image.fromarray(np.clip(multiplier*np.asarray(image), 0, 255).astype(np.uint8))
        dtype = image.dtype if preserve_type else np.float
        if clip:
            image = np.clip(multiplier*image, 0, 255 if dtype == np.uint8 else 1.)
        else:
            image = multiplier * image
        return image.astype(dtype)

    @apply_parallel
    def add(self, image, term=1., clip=False, preserve_type=False):
        """ Add term to each pixel.

        Parameters
        ----------
        term : float, sequence
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        term = np.float32(term)
        if isinstance(image, PIL.Image.Image):
            return PIL.Image.fromarray(np.clip(term+np.asarray(image), 0, 255).astype(np.uint8))
        dtype = image.dtype if preserve_type else np.float
        if clip:
            image = np.clip(term+image, 0, 255 if dtype == np.uint8 else 1.)
        else:
            image = term + image
        return image.astype(dtype)

    @apply_parallel
    def pil_convert(self, image, mode="L"):
        """ Convert image. Actually calls ``image.convert(mode)``.

        Parameters
        ----------
        mode : str
            Pass 'L' to convert to grayscale
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return image.convert(mode)

    @apply_parallel
    def posterize(self, image, bits=4):
        """ Posterizes image.

        More concretely, it quantizes pixels' values so that they have``2^bits`` colors

        Parameters
        ----------
        bits : int
            Number of bits used to store a color's component.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        return PIL.ImageOps.posterize(image, bits)

    @apply_parallel
    def cutout(self, image, origin, shape, color):
        """ Fills given areas with color

        .. note:: It is assumed that ``origins``, ``shapes`` and ``colors`` have the same length.

        Parameters
        ----------
        origin : sequence, str
            Location of the cropping box. See :meth:`.ImagesBatch._calc_origin` for details.
        shape : sequence, int
            Shape of a filled box. Can be one of:
                - sequence - crop size in the form of (rows, columns)
                - int - shape has squared form

        color : sequence, number
            Color of a filled box. Can be one of:

            - sequence - (r,g,b) form
            - number - grayscale
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.

        Notes
        -----
        Using 'random' origin with `src` as list with multiple elements will not result in same crop for each
        element, as origin will be sampled independently for each `src` element.
        To randomly sample same origin for a number of components, use `R` named expression for `origin` argument.
        """
        image = image.copy()
        shape = (shape, shape) if isinstance(shape, Number) else shape
        origin = self._calc_origin(shape, origin, self._get_image_shape(image))
        color = (color, color, color) if isinstance(color, Number) else color
        image.paste(PIL.Image.new('RGB', tuple(shape), tuple(color)), tuple(origin))
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
    def split_to_patches(self, ix, patch_shape, stride=1, drop_last=False, src='images', dst=None):
        """ Splits image to patches.

        Small images with the same shape (``patch_shape``) are cropped from the original one with stride ``stride``.

        Parameters
        ----------
        patch_shape : int, sequence
            Patch's shape in the from (rows, columns). If int is given then patches have square shape.
        stride : int, square
            Step of the moving window from which patches are cropped. If int is given then the window has square shape.
        drop_last : bool
            Whether to drop patches whose window covers area out of the image.
            If False is passed then these patches are cropped from the edge of an image. See more in tutorials.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        _ = dst
        image = self.get(ix, src)
        image_shape = self._get_image_shape(image)
        image = np.array(image)
        stride = (stride, stride) if isinstance(stride, Number) else stride
        patch_shape = (patch_shape, patch_shape) if isinstance(patch_shape, Number) else patch_shape
        patches = []

        def _iterate_columns(row_from, row_to):
            column = 0
            while column < image_shape[1]-patch_shape[1]+1:
                patches.append(PIL.Image.fromarray(image[row_from:row_to, column:column+patch_shape[1]]))
                column += stride[1]
            if not drop_last and column + patch_shape[1] != image_shape[1]:
                patches.append(PIL.Image.fromarray(image[row_from:row_to,
                                                         image_shape[1]-patch_shape[1]:image_shape[1]]))

        row = 0
        while row < image_shape[0]-patch_shape[0]+1:
            _iterate_columns(row, row+patch_shape[0])
            row += stride[0]
        if not drop_last and row + patch_shape[0] != image_shape[0]:
            _iterate_columns(image_shape[0]-patch_shape[0], image_shape[0])

        return np.array(patches, dtype=object)

    @apply_parallel
    def additive_noise(self, image, noise, clip=False, preserve_type=False):
        """ Add additive noise to an image.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        noise = noise(size=(*image.size, len(image.getbands())) if isinstance(image, PIL.Image.Image) else image.shape)
        return self._add_(image, noise, clip, preserve_type)

    @apply_parallel
    def multiplicative_noise(self, image, noise, clip=False, preserve_type=False):
        """ Add multiplicative noise to an image.

        Parameters
        ----------
        noise : callable
            Distribution. Must have ``size`` parameter.
        clip : bool
            whether to force image's pixels to be in [0, 255] or [0, 1.]
        preserve_type : bool
            Whether to preserve ``dtype`` of transformed images.
            If ``False`` is given then the resulting type will be ``np.float``.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        noise = noise(size=(*image.size, len(image.getbands())) if isinstance(image, PIL.Image.Image) else image.shape)
        return self._multiply_(image, noise, clip, preserve_type)

    @apply_parallel
    def elastic_transform(self, image, alpha, sigma, **kwargs):
        """ Deformation of images as described by Simard, Steinkraus and Platt, `Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis <http://cognitivemedium.com/assets/rmnist/Simard.pdf>_`.

        Code slightly differs from `<https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a>`_.

        Parameters
        ----------
        alpha : number
            maximum of vectors' norms.
        sigma : number
            Smooth factor.
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        p : float
            Probability of applying the transform. Default is 1.
        """
        image = np.array(image)
        # full shape is needed
        shape = image.shape
        if len(shape) == 2:
            image = image[..., None]
            shape = image.shape

        kwargs.setdefault('mode', 'constant')
        kwargs.setdefault('cval', 0)

        column_shift = gaussian_filter(np.random.uniform(-1, 1, size=shape), sigma, **kwargs) * alpha
        row_shift = gaussian_filter(np.random.uniform(-1, 1, size=shape), sigma, **kwargs) * alpha

        row, column, channel = np.meshgrid(range(shape[0]), range(shape[1]), range(shape[2]))

        indices = (column + column_shift, row + row_shift, channel)

        distored_image = map_coordinates(image, indices, order=1, mode='reflect')

        if shape[-1] == 1:
            return PIL.Image.fromarray(np.uint8(distored_image.reshape(image.shape))[..., 0])
        return PIL.Image.fromarray(np.uint8(distored_image.reshape(image.shape)))
