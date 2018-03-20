""" Contains custom losses """
import tensorflow as tf


def _dice(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
          loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS, _square=False):
    e = 1e-6
    predictions = tf.sigmoid(predictions)
    axis = tuple(range(1, targets.shape.ndims))

    if label_smoothing > 0:
        num_classes = targets.shape[-1]
        targets = targets * (1 - label_smoothing) + label_smoothing / num_classes

    intersection = tf.reduce_sum(targets * predictions, axis=axis)
    if _square:
        targets = tf.reduce_sum(tf.square(targets), axis=axis)
        predictions = tf.reduce_sum(tf.square(predictions), axis=axis)
    else:
        targets = tf.reduce_sum(targets, axis=axis)
        predictions = tf.reduce_sum(predictions, axis=axis)

    loss = -(2. * intersection + e) / (targets + predictions + e)
    loss = tf.losses.compute_weighted_loss(loss, weights, scope, loss_collection, reduction)
    return loss


def dice(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
         loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """ Dice coefficient

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction)


def dice2(targets, predictions, weights=1.0, label_smoothing=0, scope=None,
          loss_collection=tf.GraphKeys.LOSSES, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """ Dice coefficient with squares in denominator

    Parameters
    ----------
    targets : tf.Tensor
        tensor with target values

    predictions : tf.Tensor
        tensor with predicted logits

    Returns
    -------
    Tensor of the same type as targets.
    If reduction is NONE, this has the same shape as targets; otherwise, it is scalar.
    """
    return _dice(targets, predictions, weights, label_smoothing, scope, loss_collection, reduction, _square=True)
