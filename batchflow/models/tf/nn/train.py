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
    """ Applies cyclic learning rate (CLR).
    https://arxiv.org/abs/1506.01186.
    This function varies the learning rate between the
    minimum (learning_rate) and the maximum (max_lr).
    It returns the decayed learning rate.

    Parameters
    ----------
    learning_rate: float of tf.Tensor
        the minimum learning rate boundary
    global_step: int of tf.Tensor.
        global step to use for the cyclic computation.  Must not be negative
    max_lr:  float
        the maximum learning rate boundary (default=0.1)
    step_size: int
        the number of iterations in half a cycle (default=10)
    mode:
        If 'sin' or 'sine' or 'sine wave':
            Learning rate changes as a sine wave, starting
            from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.

            Formula::
                (max_lr - learning_rate)/2 * sin(pi * global_step/step_size - pi) + (max_lr + learning_rate)/2

        If 'triangular' or 'triangular wave' or 'zigzag':
            Default, linearly increasing then linearly decreasing the
            learning rate at each cycle. Learning rate starting
            from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.

            Formula::

                ((max_lr - learning_rate)/pi) * asin(sin(2 * -pi/step_size * global_step)) + (max_lr + learning_rate)/2

        If 'sawtooth' or 'saw' or 'sawtooth wave' or 'saw wave':
            Learning rate linearly increasing from `learning_rate` to `max_lr`
            and then sharply drops to `learning_rate` at each cycle.
            Learning rate starting from `learning_rate` then increasing.

            Formula::

                (max_lr - learning_rate) * (global_step/step_size - floor(global_step/step_size)) + learning_rate

    name: str
        Optional name of the operation (default='CyclicLearningRate')

    Returns
    -------
    tf.Tensor
    """
    with ops.name_scope_v2(name or "CyclicLearningRate"):
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        if mode in ('sin', 'sine', 'sine wave'):
            first_factor = (learning_rate - max_lr) / 2.
            second_factor = sin((pi * global_step)/step_size)
            second_comp = (learning_rate + max_lr) / 2.
        elif mode in ('triangular', 'triangular wave', 'zigzag'):
            first_factor = (learning_rate-max_lr) / pi
            inside_sin = 2. * -pi  / step_size * global_step
            second_factor = asin(sin(inside_sin))
            second_comp = (learning_rate + max_lr) / 2.
        elif mode in ('sawtooth', 'saw', 'sawtooth wave', 'saw wave'):
            first_factor = learning_rate - max_lr
            divided_global_step = global_step / step_size
            second_factor = divided_global_step - floor(divided_global_step)
            second_comp = learning_rate
        return first_factor * second_factor + second_comp
