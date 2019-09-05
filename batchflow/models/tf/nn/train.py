""" Helpers for training """
from math import pi

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from tensorflow.math import sin, asin, floor


def piecewise_constant(global_step, *args, **kwargs):
    """ Constant learning rate decay (uses global_step param instead of x) """
    return tf.train.piecewise_constant(global_step, *args, **kwargs)

def cyclic_learning_rate(global_step, learning_rate=0.05, max_lr=0.1, step_size=10, mode='sin', name=None):

    with ops.name_scope_v2(name or "CyclicLearningRate") as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        if mode == 'sin':
        # (max_lr-learning_rate)/2 * sin(pi*global_step/step_size - pi) + (max_lr+learning_rate)/2
            first_factor = math_ops.divide(math_ops.subtract(learning_rate, max_lr), 2.)
            second_factor = sin(math_ops.divide(math_ops.multiply(pi, global_step), step_size))
        elif mode == 'triangular':
        # ((max_lr-learning_rate)/pi) * asin(sin(((2*-pi)/step_size)*global_step)) + (max_lr+learning_rate)/2
            first_factor = math_ops.divide(math_ops.subtract(max_lr, learning_rate), pi)
            inside_sin = math_ops.multiply(math_ops.divide(math_ops.multiply(2., -pi), step_size), global_step)
            second_factor = asin(sin(inside_sin))
        elif mode == 'sawtooth':
        # (max_lr-learning_rate) * (global_step / step_size - floor(global_step / step_size)) + learning_rate
            first_factor = math_ops.subtract(max_lr, learning_rate)
            divided_global_step = math_ops.divide(global_step, step_size)
            second_factor = math_ops.subtract(divided_global_step, floor(divided_global_step))
            return math_ops.add(math_ops.multiply(first_factor, second_factor), learning_rate)

        second_comp = math_ops.divide(math_ops.add(learning_rate, max_lr), 2.)
        return math_ops.add(math_ops.multiply(first_factor, second_factor), second_comp)
