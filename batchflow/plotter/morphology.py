"""Morphological operations implemented with numba to replace cv2 dependency."""

import numpy as np
from numba import njit, prange


@njit
def dilate(image, kernel, iterations=1):
    """Dilate an image using a structuring element.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to dilate.
    kernel : numpy.ndarray
        Structuring element (kernel) for dilation. Should contain 1s where
        the structuring element is active and 0s elsewhere.
    iterations : int, optional
        Number of times to apply the dilation. Default is 1.

    Returns
    -------
    numpy.ndarray
        Dilated image with the same shape and dtype as input.

    """
    result = image.copy()

    for _ in range(iterations):
        result = _single_dilate(result, kernel)

    return result

@njit
def erode(image, kernel, iterations=1):
    """Erode an image using a structuring element.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to erode.
    kernel : numpy.ndarray
        Structuring element (kernel) for erosion. Should contain 1s where
        the structuring element is active and 0s elsewhere.
    iterations : int, optional
        Number of times to apply the erosion. Default is 1.

    Returns
    -------
    numpy.ndarray
        Eroded image with the same shape and dtype as input.

    """
    result = image.copy()

    for _ in range(iterations):
        result = _single_erode(result, kernel)

    return result

@njit(parallel=True)
def _single_dilate(image, kernel):
    """Single iteration of dilation operation."""
    height, width = image.shape
    kh, kw = kernel.shape
    kh_half, kw_half = kh // 2, kw // 2

    # Create output array
    result = np.zeros_like(image)

    # Apply dilation - for each output pixel, find max in kernel neighborhood
    for i in prange(height):
        for j in range(width):
            max_val = image[i, j]  # Start with current pixel value

            for ki in range(kh):
                for kj in range(kw):
                    if kernel[ki, kj] > 0:  # Only consider active kernel elements
                        # Calculate the source image coordinates
                        img_i = i + ki - kh_half
                        img_j = j + kj - kw_half

                        # Check bounds
                        if 0 <= img_i < height and 0 <= img_j < width:
                            if image[img_i, img_j] > max_val:
                                max_val = image[img_i, img_j]

            result[i, j] = max_val

    return result

@njit(parallel=True)
def _single_erode(image, kernel):
    """Single iteration of erosion operation."""
    height, width = image.shape
    kh, kw = kernel.shape
    kh_half, kw_half = kh // 2, kw // 2

    # Create output array
    result = np.zeros_like(image)

    # Apply erosion - for each output pixel, find min in kernel neighborhood
    for i in prange(height):
        for j in range(width):
            min_val = image[i, j]  # Start with current pixel value

            for ki in range(kh):
                for kj in range(kw):
                    if kernel[ki, kj] > 0:  # Only consider active kernel elements
                        # Calculate the source image coordinates
                        img_i = i + ki - kh_half
                        img_j = j + kj - kw_half

                        # Check bounds - treat out of bounds as 0 for erosion
                        if 0 <= img_i < height and 0 <= img_j < width:
                            if image[img_i, img_j] < min_val:
                                min_val = image[img_i, img_j]
                        else:
                            # Outside bounds treated as 0, so erosion result should be 0
                            min_val = 0
                            break

            result[i, j] = min_val

    return result
