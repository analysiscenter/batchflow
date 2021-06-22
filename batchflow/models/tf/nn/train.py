""" Helpers for training """
from math import pi

import tensorflow.compat.v1 as tf

def piecewise_constant(global_step, *args, **kwargs):
    """ Constant learning rate decay (uses global_step param instead of x) """
    return tf.train.piecewise_constant(global_step, *args, **kwargs)

def cyclic_learning_rate(learning_rate, global_step, max_lr, step_size=10,
                         mode='tri', name='CyclicLearningRate'):
    """ This function varies the learning rate between the
    minimum (learning_rate) and the maximum (max_lr).
    It returns the decayed learning rate.

    Parameters
    ----------
    learning_rate : float or tf.Tensor
        The minimum learning rate boundary.
    global_step : int or tf.Tensor
        Global_step refers to the number of batches seen by the model.
        It is use for the cyclic computation. Must not be negative.
    max_lr : float or tf.Tensor
        The maximum learning rate boundary.
    step_size : int or tf.Tensor
        The number of iterations in half a cycle (the default is 10).
    mode : {'tri', 'sin', 'saw'}
        Set the learning rate change function.
    name : str
        Name of the operation (the default is 'CyclicLearningRate').

    Returns
    -------
    tf.Tensor

    Notes
    -----
    More detailed information about `mode`:

    If 'tri':
        Default, linearly increasing then linearly decreasing the
        learning rate at each cycle. Learning rate starting
        from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.
        See `Leslie N. Smith, Cyclical Learning Rates for Training Neural Networks
        <https://arxiv.org/abs/1506.01186>`_ for more information.

        It is computed as::

            decayed_learning_rate = abs(mod((global_step + step_size / 4) / step_size, 1) - 0.5) *
                                    2 * (max_lr - learning_rate) +
                                    learning_rate


    If 'sin':
        Learning rate changes as a sine wave, starting
        from (max_lr-learning_rate)/2 then decreasing to `learning_rate`.

        It is computed as::

            decayed_learning_rate = (learning_rate - max_lr) / 2 *
                                    sin(pi * global_step / step_size) +
                                    (max_lr + learning_rate) / 2


    If 'saw':
        Learning rate linearly increasing from `learning_rate` to `max_lr`
        and then sharply drops to `learning_rate` at each cycle.
        Learning rate starting from `learning_rate` then increasing.

        It is computed as::

            decayed_learning_rate = (max_lr - learning_rate) *
                                    (floor(global_step / step_size) - global_step / step_size) +
                                    learning_rate
    """
    with tf.name_scope(name):
        learning_rate = tf.cast(learning_rate, dtype=tf.float32)
        global_step = tf.cast(global_step, dtype=tf.float32)
        step_size = tf.cast(step_size, dtype=tf.float32)
        max_lr = tf.cast(max_lr, dtype=tf.float32)

        if mode == 'tri':
            periodic_comp = tf.mod((global_step + step_size / 4) / step_size, 1)
            first_factor = tf.abs(periodic_comp - 0.5)
            second_factor = 2 * (max_lr - learning_rate)
            second_comp = learning_rate
        elif mode == 'sin':
            first_factor = (learning_rate - max_lr) / 2.
            second_factor = tf.sin((pi * global_step) / step_size)
            second_comp = (learning_rate + max_lr) / 2.
        elif mode == 'saw':
            first_factor = max_lr - learning_rate
            second_factor = tf.mod(global_step / step_size, 1)
            second_comp = learning_rate
        return first_factor * second_factor + second_comp
