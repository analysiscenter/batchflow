""" Contains custom losses """
import tensorflow as tf

from ..layers import flatten


def dice(targets, predictions):
    """ Dice coefficient """
    e = 1e-6
    intersection = flatten(targets * predictions)
    loss = -tf.reduce_mean((2. * intersection + e) / (flatten(targets) + flatten(predictions) + e))
    return loss
