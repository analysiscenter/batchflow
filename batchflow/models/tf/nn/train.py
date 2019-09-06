""" Helpers for training """
from math import pi

import tensorflow as tf
from tensorflow.python.framework import ops # pylint: disable=no-name-in-module
from tensorflow.python.ops import math_ops # pylint: disable=no-name-in-module
from tensorflow.math import sin, asin, floor # pylint: disable=import-error


def piecewise_constant(global_step, *args, **kwargs):
    """ Constant learning rate decay (uses global_step param instead of x) """
    return tf.train.piecewise_constant(global_step, *args, **kwargs)

def cyclic_learning_rate(learning_rate, global_step, max_lr=0.1, step_size=10, mode='triangular', name=None):
    """Applies cyclic learning rate (CLR).
    https://arxiv.org/pdf/1506.01186.pdf.

    This function varies the learning rate between the
    minimum (learning_rate) and the maximum (max_lr).
    It returns the cyclic learning rate.

    Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number. The minimun learning rate boundary.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        Global step to use for the cyclic computation.  Must not be negative.
    max_lr:  A scalar. The maximum learning rate boundary.
        Default 0.1.
    step_size: A scalar. The number of iterations in half a cycle.
        Default 10.
    mode:
        If 'sin' or 'sine' or 'sine wave':
            Learning rate changes as a sine wave, starting
            from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.
        If 'triangular' or 'triangular wave' or 'zigzag':
            Default, linearly increasing then linearly decreasing the
            learning rate at each cycle. Learning rate starting
            from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.
        If 'sawtooth' or 'saw' or 'sawtooth wave' or 'saw wave':
            Linearly increasing to `max_lr` then again increasing from `learning_rate`
            learning rate at each cycle. Learning rate starting from `learning_rate`
            then increasing.
    name: String.  Optional name of the operation.  Defaults to
        'CyclicLearningRate'.

    Returns:
    A scalar `Tensor` of the same type as `learning_rate`. The cyclic
    learning rate.
    """
    with ops.name_scope_v2(name or "CyclicLearningRate"):
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        if mode in ('sin', 'sine', 'sine wave'):
        # (max_lr - learning_rate)/2 * sin(pi * global_step/step_size - pi) + (max_lr + learning_rate)/2
            first_factor = math_ops.divide(math_ops.subtract(learning_rate, max_lr), 2.)
            second_factor = sin(math_ops.divide(math_ops.multiply(pi, global_step), step_size))
            second_comp = math_ops.divide(math_ops.add(learning_rate, max_lr), 2.)
            return math_ops.add(math_ops.multiply(first_factor, second_factor), second_comp)
        if mode in ('triangular', 'triangular wave', 'zigzag'):
        # ((max_lr - learning_rate)/pi) * asin(sin(((2 * -pi)/step_size) * global_step)) + (max_lr + learning_rate)/2
            first_factor = math_ops.divide(math_ops.subtract(max_lr, learning_rate), pi)
            inside_sin = math_ops.multiply(math_ops.divide(math_ops.multiply(2., -pi), step_size), global_step)
            second_factor = asin(sin(inside_sin))
            second_comp = math_ops.divide(math_ops.add(learning_rate, max_lr), 2.)
            return math_ops.add(math_ops.multiply(first_factor, second_factor), second_comp)
        if mode in ('sawtooth', 'saw', 'sawtooth wave', 'saw wave'):
        # (max_lr - learning_rate) * (global_step/step_size - floor(global_step/step_size)) + learning_rate
            first_factor = math_ops.subtract(max_lr, learning_rate)
            divided_global_step = math_ops.divide(global_step, step_size)
            second_factor = math_ops.subtract(divided_global_step, floor(divided_global_step))
            return math_ops.add(math_ops.multiply(first_factor, second_factor), learning_rate)
