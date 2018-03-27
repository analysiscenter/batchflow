""" Custom batch class for storing mnist batch and models
"""
import sys

import numpy as np

sys.path.append('..')
from dataset.dataset import action, inbatch_parallel, ImagesBatch

class MnistBatch(ImagesBatch):
    """ Mnist batch and models
    Attributes
    ----------
    images: numpy array
    Array with images

    labels: numpy array
    Array with answers """

    components = 'images', 'labels'

    @action
    @inbatch_parallel(init='images', post='assemble', target='threads')
    def shift_flattened_pic(self, image, max_margin=8):
        """ Apply random shift to a flattened pic

        Parameters
        ----------
        ind: numpy array
        Array with indices, which need to shift

        max_margin: int
        Constit max value of margin that inamge may
        be shift

        Returns
        -------
        flattened shifted pic """
        padded = np.pad(image, pad_width=[[max_margin, max_margin], [max_margin, max_margin], [0, 0]],
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        return padded[slicing]
