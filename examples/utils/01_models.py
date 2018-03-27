"""File contains classes with main regression alghoritms"""
import tensorflow as tf

from dataset.dataset.models.tf import TFModel

class Regressions(TFModel):
    """ Class with logistic regression model """

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Body of our model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        """
        with tf.variable_scope(name):
            dense = tf.layers.dense(inputs, kwargs['units'], name='dense')

        tf.cast(tf.exp(dense), tf.int32, name='predicted_poisson')

        return dense
