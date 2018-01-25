""" Contains Batch classes for images """
import os
import traceback
from numbers import Number

import numpy as np
try:
    import scipy.ndimage
    import scipy.misc.imsave
except ImportError:
    pass

from .batch import Batch
from .decorators import action, inbatch_parallel, any_action_failed


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
    def __decorator(cls):
        for method_name, method in cls.__dict__.copy().items():
            if method_name.startswith(prefix) and method_name.endswith(suffix) and\
               not method_name.startswith('__') and not method_name.endswith('__'):
                def __wrapper():
                    #pylint: disable=cell-var-from-loop
                    wrapped_method = method
                    def __func(self, src='images', dst='images', *args, **kwargs):
                        return getattr(cls, wrapper)(self, wrapped_method, src=src, dst=dst,
                                                     use_self=True, *args, **kwargs)
                    return __func
                name_slice = slice(len(prefix), -len(suffix))
                wrapped_method_name = method_name[name_slice]
                setattr(cls, wrapped_method_name, action(__wrapper()))
        return cls
    return __decorator


class BaseImagesBatch(Batch):
    """ Batch class for 2D images """
    components = "images", "labels"
    #pylint: disable=arguments-differ
    def _assemble(self, all_res, components='images', *args, **kwargs):
        """ Assemble the batch after a parallel action.

        Parameters
        ----------
        all_res : list
            processed components of the batch

        components : str, sequence
            names of the components to assemble

        Returns
        -------
        self
        """
        _ = all_res, args, kwargs, components
        raise NotImplementedError("Must be implemented in a child class")

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
            joined path if path is not None else element's path specified in the batch's index
        """
        return self.index.get_fullpath(ix) if path is None else os.path.join(path, ix)

    def _load_image(self, ix, src=None, components="images"):
        """ Loads image

        .. note:: only works with a single component

        Parameters
        ----------
        path : str, None
        ix : str
            element's index (filename)
        components : str
            component to load

        Returns
        -------
        np.ndarray : Loaded image.
        """
        _ = self, ix, src, components
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """ Load data.

        .. note:: if `fmt='images'` than there must be single component.

        Parameters
        ----------
        src : str, None
            Path to the folder with data. If src is None then path is determined from index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to download.
        components : str, sequence
            components to download.

        Returns
        -------
        self
        """
        if fmt == 'image':
            self._load_image(src, components)
        else:
            super().load(src, fmt, components, *args, **kwargs)
        return self

    def _dump_image(self, ix, dst=None, components='images'):
        """ Saves image to dst.

        .. note:: only works with a single component

        Parameters
        ----------
        dst : str
            Folder where to dump. If dst is None then it is determined from index.
        components : str
            component to save.
        """
        _ = self, ix, dst, components
        raise NotImplementedError("Must be implemented in a child class")

    @action
    def dump(self, dst=None, fmt=None, components="images", *args, **kwargs):
        """ Dump data.

        .. note:: if `fmt='images'` than there must be single component.


        Parameters
        ----------
        dst : str, None
            Path to the folder where to dump. If dst is None then path is determined from index.
        fmt : {'image', 'blosc', 'csv', 'hdf5', 'feather'}
            Format of the file to save.
        components : str, sequence
            components to save.

        Returns
        -------
        self
        """
        if fmt == 'image':
            self._dump_image(dst, components)
        return super().dump(dst, fmt, components, *args, **kwargs)


@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')
@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
class ImagesBatch(BaseImagesBatch):
    """ Batch class for 2D images.

    Images are stored as numpy arrays (N, H, W, C).
    """

    @classmethod
    def _get_image_shape(cls, image):
        return image.shape[:2]

    @property
    def image_shape(self):
        """: tuple - shape of the image """
        if isinstance(self.images.dtype, object):
            _, shapes_count = np.unique(list(map(lambda x: x.shape, self.images)),
                                        return_counts=True, axis=0)
            if len(shapes_count) == 1:
                return self.images.shape[1:]
            else:
                print(list(map(lambda x: x.shape, self.images)))
                raise RuntimeError('Images have different shapes')
        return self.images.shape[1:]

    @inbatch_parallel(init='indices', post='assemble')
    def _load_image(self, ix, src=None, components="images"):
        """ Wrapper for scipy.ndimage.open.

        .. note:: only works with a single component

        Parameters
        ----------
        path : str, None
        ix : str
            element's index (filename)
        components : str
            component to load

        Returns
        -------
        np.ndarray : Loaded image
        """
        return scipy.ndimage.open(self._make_path(src, ix))

    @inbatch_parallel(init='indices')
    def _dump_image(self, ix, dst=None, components='images'):
        """ Save image to dst.

        Actually a wrapper for scipy.misc.imsave.

        .. note:: only works with a single component

        Parameters
        ----------
        dst : str
            Folder where to dump. If dst is None then it is determined from index.
        components : str
            component to save.
        """
        scipy.misc.imsave(self._make_path(dst, ix), self.get(ix, components))

    def _assemble_component(self, all_res, components='images', **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        all_res : sequence, array_like
            Results after inbatch_parallel.
        components : str
            component to assemble
        preserve_shape : bool
            If True then all images are cropped from the top left corner to have similar shapes.
            Shape is chosen to be minimal among given images.
        """
        try:
            new_images = np.stack(all_res)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                preserve_shape = kwargs.get('preserve_shape', False)
                if preserve_shape:
                    min_shape = np.array([self._get_image_shape(x) for x in all_res]).min(axis=0)
                    all_res = [arr[:min_shape[0], :min_shape[1]].copy() for arr in all_res]
                    new_images = np.stack(all_res)
                else:
                    new_images = np.array(all_res, dtype=object)
            else:
                raise e
        setattr(self, components, new_images)

    def _assemble(self, all_res, components='images', *args, **kwargs):
        """ Assemble the batch after a parallel action.

        Parameters
        ----------
        all_res : sequence, array_like
            results from parallel action
        components : str, sequence
            components to assemble
        dst : str
            if `components` is not one of (list, tuple, str)
            then the value of `dst` will be used instead `components`

        Returns
        -------
        self
        """
        _ = args
        if any_action_failed(all_res):
            all_errors = self.get_errors(all_res)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")
        if not isinstance(components, (list, tuple, str)):
            components = kwargs.get('dst', 'images')
        if isinstance(components, (list, tuple)):
            all_res = list(zip(*all_res))
        else:
            components = [components]
            all_res = [all_res]
        for component, res in zip(components, all_res):
            self._assemble_component(res, component, **kwargs)
        return self

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
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
            image to scale
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

        Returns
        -------
        np.ndarray : rescaled image
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
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
        origin : sequence
            Upper-left corner of the cropping box. Can be one of:
            - sequence - a starting point in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of
                           an image and the cropping box coincide
            - 'center' - crop an image such that centers of
                         an image and the cropping box coincide
            - 'random' - place the upper-left corner of the cropping box at a random position
        shape : sequence
            - sequence - crop size in the form of (rows, columns)

        Returns
        -------
        np.ndarray : cropped image
        """
        image_shape = self._get_image_shape(image)
        origin = self._calc_origin(shape, origin, image_shape)
        if np.all(origin + shape > image_shape):
            shape = image_shape - origin

        row_slice = slice(origin[0], origin[0] + shape[0])
        column_slice = slice(origin[1], origin[1] + shape[1])
        return image[row_slice, column_slice].copy()

    def _put_on_background_(self, image, background, origin):
        """ Put an image on a background at origin

        Parameters
        ----------
        image : np.ndarray
        background : np.array
        origin : sequence, str
            Upper-left corner of the cropping box. Can be one of:
            - sequence - a starting point in the form of (row, column)
            - 'top_left' - crop an image such that upper-left corners of an image and the cropping box coincide.
            - 'center' - crop an image such that centers of an image and the cropping box coincide.
            - 'random' - place the upper-left corner of the cropping box at a random position.

        Returns
        -------
        np.ndarray : the image placed on the background
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
                         the transformed image accordingly
            - 'top_left' - place the upper-left corner of the transformed image on the upper-left of the original one
                           and crop the transformed image accordingly
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

    def _resize_(self, image, shape=None, order=0, *args, **kwargs):
        """ Resize an image to the given shape

        Actually a wrapper for scipy.ndimage.interpolation.zoom method. *args and **kwargs are passed to the last.

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
            image to resize
        shape : sequence
            resulting shape in the following form: (number of rows, number of columns)
        order : int
            The order of the spline interpolation, default is 0. The order has to be in the range 0-5.

        Returns
        -------
        np.ndarray : resized image
        """
        image_shape = self._get_image_shape(image)
        factor = np.asarray(shape) / np.asarray(image_shape)
        if len(image.shape) > 2:
            factor = np.concatenate((factor,
                                     [1.]*(len(image.shape)-len(image_shape))))
        new_image = scipy.ndimage.interpolation.zoom(image, factor, order=order, *args, **kwargs)
        return new_image

    def _shift_(self, image, order=0, *args, **kwargs):
        """ Shift an image.

        Actually a wrapper for scipy.ndimage.interpolation.shift. *args and **kwargs are passed to the last.

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
            image to shift
        shift : float or sequence
            The shift along the axes. If a float, shift is the same for each axis.
            If a sequence, shift should contain one value for each axis.
        order : int
            The order of the spline interpolation, default is 0. The order has to be in the range 0-5.

        Returns
        -------
        np.ndarray : shifted image
        """

        return scipy.ndimage.interpolation.shift(image, order=order, *args, **kwargs)

    def _rotate_(self, image, angle, order=0, *args, **kwargs):
        """ Rotate an image.

        Actually a wrapper for scipy.ndimage.interpolation.rotate. *args and **kwargs are passed to the last.

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
            image to rotate
        angle : float
            The rotation angle in degrees.
        order : int
            The order of the spline interpolation, default is 0. The order has to be in the range 0-5.

        Returns
        -------
        np.ndarray : shifted image
        """

        return scipy.ndimage.interpolation.rotate(image, angle=angle, order=order, *args, **kwargs)

    def _flip_all(self, images=None, indices=None, mode='lr'):
        """ Flip images in the batch.

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        images : np.ndarray
            batch of images
        indices : sequence
            indices of images to flip
        mode : {'lr', 'ud'}
            - 'lr' - apply the left/right flip
            - 'ud' - apply the upside/down flip
        """
        if mode == 'lr':
            images[indices] = images[indices, :, ::-1]
        elif mode == 'ud':
            images[indices] = images[indices, ::-1]
        return images

    def _pad_(self, image, *args, **kwargs):
        """ Pad an image.

        Actually a wrapper for np.pad.

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        pad_width : sequence, array_like, int
            Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N))
            unique pad widths for each axis. ((before, after),) yields same before and after pad for each axis. (pad,)
            or int is a shortcut for before = after = pad width for all axes.
        mode : str or function
            mode of padding. For more details see np.pad

        Returns
        -------
        np.ndarray : padded image
        """
        return np.pad(image, *args, **kwargs)

    def _invert_(self, image, channels='all'):
        """ Invert channels

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        channels : int, sequence
            channels indices to invert.

        Returns
        -------
        np.ndarray : inverted image
        """

        if channels == 'all':
            channels = list(range(image.shape[-1]))
        inv_multiplier = 255 if np.issubdtype(image.dtype, np.integer) else 1.
        image[..., channels] = inv_multiplier - image[..., channels]
        return image



    def _salt_(self, image, p_noise=.015, color=255, size=(1,1)):
        """ set random pixel on image to givan value

        every pixel will be set to ``color`` value with probability ``p_noise``

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
            image to flavour with species
        p_noise : float
            probability of salting a pixel
        color : float, int, sequence, callable
            color's value.
            - int, float, sequence -- value of color
            - callable -- color is sampled for every chosen pixel (rules are the same as for int, float and sequence)
        size : int, sequence of int, callable
            size of salt
            - int -- square salt with side ``size``
            - sequence -- recangular salt in the form (row, columns)
            - callable -- size is sampled for every chosen pixel (rules are the same as for int and sequence)

        Returns
        -------
        np.ndarray : flavoured image
        """
        mask_size = np.asarray(self._get_image_shape(image))
        mask_salt = np.random.binomial(1, p_noise, size=mask_size).astype(bool)
        if (size == (1,1) or size == 1) and not callable(color):
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
        """ truncate image's pixels

        Parameters
        ----------
        low : int, float, sequence
            lower threshold, if sequence is given then apply it to every channel
        high : int, float, sequence
            higher threshold, if sequence is given then apply it to every channel
        dtype : np.dtype
            dtype of returned images
        !!! ADD DOCS

        Returns
        -------
        np.ndarray : truncated images
        """
        if isinstance(low, Number):
            image[image < low] = low
        else:
            if len(low) != image.shape[-1]:
                raise RuntimeError("``len(low)`` must coincide with the number of channels")
            for channel, low_channel in enumerate(low):
                pixels_to_truncate = image[...,channel] < low_channel
                image[...,channel][pixels_to_truncate] = low_channel
        if isinstance(high, Number):
            image[image > high] = high
        else:
            if len(high) != image.shape[-1]:
                raise RuntimeError("``len(high)`` must coincide with the number of channels")

            for channel, high_channel in enumerate(high):
                pixels_to_truncate = image[...,channel] > high_channel
                image[...,channel][pixels_to_truncate] = high_channel
        return image.astype(dtype)

    def _multiply_(self, image, multiplier=1., low=0., high=1., preserve_type=True):
        """multiply each pixel by the given multiplier

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
        multiplier : float, sequence
        low : actual pixel's value is equal max(value, low)
        high : actual pixel's value is equal min(value, high)

        Returns
        -------
        np.ndarray : transformed image
        """
        dtype = image.dtype if preserve_type else np.float
        return self._threshold_(multiplier * image.astype(np.float), low, high, dtype)

    def _add_(self, image, term=0., low=0., high=1., preserve_type=True):
        """add term to each pixel

        Parameters
        ----------
        components : str
            component to get an image from
        p : float
            probability of applying the transforms
        image : np.ndarray
        term : float, sequence
        low : actual pixel's value is equal max(value, low)
        high : actual pixel's value is equal min(value, high)

        Returns
        -------
        np.ndarray : transformed image
        """
        dtype = image.dtype if preserve_type else np.float
        return self._threshold_(term + image.astype(np.float), low, high, dtype)
