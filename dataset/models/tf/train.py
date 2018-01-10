""" Helpers for training """
import tensorflow as tf


def piecewise_constant(global_step, *args, **kwargs):
    return tf.train.piecewise_constant(x=global_step, *args, **kwargs)
