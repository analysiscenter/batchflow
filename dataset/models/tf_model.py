# pylint: disable=undefined-variable
""" Contains base class for tensorflow models """

import os
import re
import json
import numpy as np
import tensorflow as tf

from .base import BaseModel


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'ce': tf.losses.softmax_cross_entropy,
    'crossentropy': tf.losses.softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'L1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': tf.train.piecewise_constant,
    'poly': tf.train.polynomial_decay
}



class TFModel(BaseModel):
    """ Base class for all tensorflow models """

    def __init__(self, *args, **kwargs):
        self.session = kwargs.get('session', None)
        self.graph = tf.Graph() if self.session is None else self.session.graph
        self._graph_context = None
        self.is_training = None
        self.global_step = None
        self.loss = None
        self.train_step = None
        self._attrs = []

        super().__init__(*args, **kwargs)


    def __enter__(self):
        """ Enter the model graph context """
        self._graph_context = self.graph.as_default()
        self._graph_context.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """ Exit the model graph context """
        return self._graph_context.__exit__(exception_type, exception_value, exception_traceback)

    def _build(self, *args, **kwargs):
        """ Define a model architecture

        This method must be implemented in ancestor classes:
        inside it a tensorflow model must be defined (placeholders, layers, loss, etc)

        Notes
        -----
        This method is executed within a self.graph context
        """
        _ = args, kwargs

    def build(self, *args, **kwargs):
        """ Build the model
        1. Define a model architecture by calling self._build(*args, **kwargs)
        2. Create an optimizer and define a train step
        3. Set UPDATE_OPS control dependency on train step
           (see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        """
        with self.graph.as_default():
            self.store_to_attr('is_training', tf.placeholder(tf.bool, name='is_training'))
            self.store_to_attr('global_step', tf.Variable(0, trainable=False, name='global_step'))

            self._build(*args, **kwargs)

            self._make_loss()
            self.store_to_attr('loss', tf.losses.get_total_loss())

            optimizer = self._make_optimizer()

            if self.train_step is None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.store_to_attr('train_step', optimizer.minimize(self.loss, global_step=self.global_step))
            else:
                self.store_to_attr('train_step', self.train_step)

            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())

    def _unpack_fn_from_config(self, param, default=None):
        par = self.get_from_config(param, default)

        if par is None:
            return None, None

        if isinstance(par, (tuple, list)):
            if len(par) == 0:
                par_name = None
            elif len(par) == 1:
                par_name, par_args = par[0], {}
            elif len(par) == 2:
                par_name, par_args = par
            else:
                par_name = par[0]
                par_args = par[1:]
        elif isinstance(par, dict):
            par_name, par_args = par.get('name', None), par.get('args', {})
        else:
            par_name, par_args = par, {}

        return par_name, par_args

    def _make_loss(self):
        """ Return a loss function from config """
        if len(tf.losses.get_losses()) == 0:
            loss = self.get_from_config("loss")
            if isinstance(loss, str) and hasattr(tf.losses, loss):
                loss = getattr(tf.losses, loss)
            elif isinstance(loss, str):
                loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
            elif callable(loss):
                pass
            elif loss is None:
                raise ValueError("Loss is not defined in the model %s" % self)
            else:
                raise ValueError("Unknown loss", loss)

            try:
                predictions = self.graph.get_tensor_by_name("predictions:0")
                targets = self.graph.get_tensor_by_name("targets:0")
            except IndexError:
                pass
            else:
                tf.losses.add_loss(loss(targets, predictions))

    def _make_decay(self):
        decay_name, decay_args = self._unpack_fn_from_config('decay')

        if decay_name is None:
            pass
        elif callable(decay_name):
            pass
        elif isinstance(decay_name, str) and hasattr(tf.train, decay_name):
            decay_name = getattr(tf.train, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def _make_optimizer(self):
        optimizer_name, optimizer_args = self._unpack_fn_from_config('optimizer', (tf.train.AdamOptimizer, {}))

        if callable(optimizer_name):
            pass
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name):
            optimizer_name = getattr(tf.train, optimizer_name)
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name + 'Optimizer'):
            optimizer_name = getattr(tf.train, optimizer_name + 'Optimizer')
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        decay_name, decay_args = self._make_decay()
        if decay_name is not None:
            optimizer_args['learning_rate'] = decay_name(**decay_args, global_step=self.global_step)

        optimizer = optimizer_name(**optimizer_args)

        return optimizer

    def get_number_of_trainable_vars(self):
        """ Return the number of trainable variable in the model graph """
        with self:
            arr = np.asarray([np.prod(self.get_shape(v)) for v in tf.trainable_variables()])
        return np.sum(arr)

    @staticmethod
    def num_channels(tensor):
        """ Return the number of channels (last dimension) in the tensor

        Parameters
        ----------
        tensor: tf.Variable or tf.Tensor

        Returns
        -------
        number of channels: int
        """
        return tensor.get_shape().as_list()[-1]

    @staticmethod
    def batch_size(tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor: tf.Variable or tf.Tensor

        Returns
        -------
        batch size: int
        """
        return tensor.get_shape().as_list()[0]

    @staticmethod
    def get_shape(tensor):
        """ Return full shape of the input tensor

        Parameters
        ----------
        tensor: tf.Variable or tf.Tensor

        Returns
        -------
        shape: tuple of ints
        """
        return tensor.get_shape().as_list()

    def train(self, fetches=None, feed_dict=None):   # pylint: disable=arguments-differ
        """ Train the model with the data provided """
        with self:
            feed_dict = feed_dict or {}
            _feed_dict = {}
            for placeholder_name, value in feed_dict.items():
                placeholder = self.graph.get_tensor_by_name(placeholder_name + ':0')
                _feed_dict.update({placeholder: value})
            _feed_dict.update({self.is_training: True})
            _fetches = fetches or tuple()
            _, output = self.session.run([self.train_step, _fetches], feed_dict=_feed_dict)
        return output

    def predict(self, fetches, feed_dict=None):      # pylint: disable=arguments-differ
        """ Get predictions on the data provided """
        with self:
            feed_dict = feed_dict or {}
            _feed_dict = {self.is_training: False}
            _feed_dict = {**feed_dict, **_feed_dict}
            output = self.session.run(fetches, _feed_dict)
        return output

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model.

        Parameters
        ----------
        path: str - a path to a directory where all model files will be stored

        Examples
        --------
        >>> tf_model = TFResNet34()
        >>> ... train the model
        >>> tf_model.save('/path/to/models/resnet34')
        The model will be saved to /path/to/models/resnet34
        """
        with self.graph.as_default():
            if not os.path.exists(path):
                os.makedirs(path)
            saver = tf.train.Saver()
            saver.save(self.session, os.path.join(path, 'model'), *args, global_step=self.global_step, **kwargs)
            with open(os.path.join(path, 'attrs.json'), 'w') as f:
                json.dump(self._attrs, f)

    def load(self, path, graph=None, checkpoint=None, *args, **kwargs):
        """ Load a tensorflow model from files

        Parameters
        ----------
        - path: str - a directory where a model is stored
        - graph: str - a filename for a metagraph file
        - checkpoint: str - a checkpoint file name or None to load the latest checkpoint

        Examples
        --------
        >>> tf_model = TFResNet34(load=True)
        >>> tf_model.load('/path/to/models/resnet34')
        """
        _ = args, kwargs
        self.session = tf.Session()

        with self.session.as_default():
            graph_path = os.path.join(path, graph or 'model-0.meta')
            saver = tf.train.import_meta_graph(graph_path)

            if checkpoint is None:
                checkpoint_path = tf.train.latest_checkpoint(path)
            else:
                checkpoint_path = os.path.join(path, checkpoint)

            saver.restore(self.session, checkpoint_path)
            self.graph = self.session.graph

        with open(os.path.join(path, 'attrs.json'), 'r') as json_file:
            self._attrs = json.load(json_file)
        with self.graph.as_default():
            for attr, graph_item in zip(self._attrs, tf.get_collection('attrs')):
                setattr(self, attr, graph_item)

    def store_to_attr(self, attr, graph_item):
        """ Make a graph item (variable or operation) accessible as a model attribute """
        with self.graph.as_default():
            setattr(self, attr, graph_item)
            self._attrs.append(attr)
            tf.get_collection_ref('attrs').append(graph_item)
