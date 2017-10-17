# pylint: disable=undefined-variable
""" Contains base class for tensorflow models """

import os
import re
import json
import numpy as np
import tensorflow as tf

from ..base import BaseModel
from .losses import dice


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'ce': tf.losses.softmax_cross_entropy,
    'crossentropy': tf.losses.softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'l1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss,
    'dice': dice
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': tf.train.piecewise_constant,
    'poly': tf.train.polynomial_decay
}


class TFModel(BaseModel):
    """ Base class for all tensorflow models

    Attributes
    ----------
    name : str - a model name
    config : dict - configuration parameters

    session : tf.Session
    graph : tf.Graph
    is_training : tf.Tensor
    global_step : tf.Tensor
    loss : tf.Tensor
    train_step : tf.Operation


    Configuration
    -------------
    session : dict - parameters for session creation (https://www.tensorflow.org/api_docs/python/tf/Session#__init__)

    loss - a loss function, might be one of:
        - short name ('mse', 'ce', 'l1', 'cos', 'hinge', 'huber', 'logloss', 'dice')
        - a function name from tf.losses (e.g. 'absolute_difference' or 'sparse_softmax_cross_entropy')
        - a callable

        Examples:
        ``{'loss': 'mse'}``
        ``{'loss': 'sigmoid_cross_entropy'}``
        ``{'loss': tf.losses.huber_loss}``
        ``{'loss': external_loss_fn}``

    decay - a learning rate decay algorithm might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, **other_args}

        where name might be one of
        - short name ('exp', 'invtime', 'naturalexp', 'const', 'poly')
        - a function name from tf.train (e.g. 'exponential_decay')
        - a callable

        Examples:
           ``{'decay': 'exp'}``
           ``{'decay': ('polynomial_decay', {'decay_steps':10000})}``
           ``{'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}``

    optimizer - an optimizer might be defined in one of three formats
            - name
            - tuple (name, args)
            - dict with keys 'name' and 'args'

            where name might be one of
            - short name (e.g. 'Adam', 'Adagrad')
            - a function name from tf.train (e.g. 'FtlrOptimizer')
            - a callable

        Examples:
            ``{'optimizer': 'Adam'}``
            ``{'optimizer': ('Ftlr', {'learning_rate_power': 0})}``
            ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
            ``{'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}``
            ``{'optimizer': some_optimizer_fn}``

    """

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
        return self._graph_context.__enter__()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """ Exit the model graph context """
        return self._graph_context.__exit__(exception_type, exception_value, exception_traceback)

    def _build(self, *args, **kwargs):
        """ Define a model architecture

        This method must be implemented in ancestor classes:
        inside it a tensorflow model must be defined (placeholders, layers, loss, etc)

        How to write your own _build method
        -----------------------------------
        1. Give names to all placeholders (you will need them later in `train` and `predict`)
        2. For dropout, batch norm, etc you might use a predefined `self.is_training` tensor.
        3. For learning rate decay and training control you might use a predefined `self.global_step` tensor.
        4. In many cases there is no need to write a loss function as it might be defined through config, e.g.
        5. However, for that to work you have to define operations with names `targets` and `predictions`.
           Their output tensors will be sent to a loss function.
        6. If you need to define your own loss function, use losses from `tf.losses` or call `tf.losses.add_loss(...)`
        7. In most cases there is no need to define an optimizer as well,
           since it might be defined through config, e.g.:
        8. If you need to write your own optimizer, assing `self.train_step` to the train step operation.
           Don't forget about UPDATE_OPS control dependency
           (see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)

        Notes
        -----
        This method is executed within a self.graph context
        """
        _ = args, kwargs

    def build(self, *args, **kwargs):
        """ Build the model

        1. Define is_training and global_step tensors
        2. Define a model architecture by calling self._build(*args, **kwargs)
        3. Create a loss function
        4. Create an optimizer and define a train step
        5. Set UPDATE_OPS control dependency on train step
           (see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
        6. Create a tensorflow session
        """
        with self.graph.as_default():
            self.store_to_attr('is_training', tf.placeholder(tf.bool, name='is_training'))
            self.store_to_attr('global_step', tf.Variable(0, trainable=False, name='global_step'))

            self._build(*args, **kwargs)

            self._make_loss()
            self.store_to_attr('loss', tf.losses.get_total_loss())

            if self.train_step is None:
                optimizer = self._make_optimizer()

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.store_to_attr('train_step', optimizer.minimize(self.loss, global_step=self.global_step))
            else:
                self.store_to_attr('train_step', self.train_step)

            session_config = self.get_from_config('session', {})
            self.session = tf.Session(**session_config)
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
                par_name, par_args = par[0], par[1:]
        elif isinstance(par, dict):
            par_name, par_args = par.pop('name', None), par
        else:
            par_name, par_args = par, {}

        return par_name, par_args

    def _make_loss(self):
        """ Return a loss function from config """
        if len(tf.losses.get_losses()) == 0:
            loss = self.get_from_config("loss")
            if loss is None:
                raise ValueError("Loss is not defined in the model %s" % self)
            elif isinstance(loss, str) and hasattr(tf.losses, loss):
                loss = getattr(tf.losses, loss)
            elif isinstance(loss, str):
                loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
            elif callable(loss):
                pass
            else:
                raise ValueError("Unknown loss", loss)

            try:
                predictions = self.graph.get_tensor_by_name("predictions:0")
                targets = self.graph.get_tensor_by_name("targets:0")
            except KeyError:
                raise KeyError("Model %s does not have 'predictions' or 'targets' tensors" % self.name)
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
        optimizer_name, optimizer_args = self._unpack_fn_from_config('optimizer', 'Adam')

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
        """ Return the number of channels (the length of the last dimension) in the tensor

        Parameters
        ----------
        tensor : tf.Variable or tf.Tensor

        Returns
        -------
        number of channels : int
        """
        return tensor.get_shape().as_list()[-1]

    @staticmethod
    def batch_size(tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : tf.Variable or tf.Tensor

        Returns
        -------
        batch size : int
        """
        return tensor.get_shape().as_list()[0]

    @staticmethod
    def get_shape(tensor):
        """ Return full shape of the input tensor

        Parameters
        ----------
        tensor : tf.Variable or tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        return tensor.get_shape().as_list()

    def _tensor_name(self, name):
        if isinstance(name, str):
            if hasattr(self, name):
                return getattr(self, name)
            elif ':' in name:
                return name
            return name + ':0'
        return name

    def _fill_feed_dict(self, feed_dict=None):
        feed_dict = feed_dict or {}
        _feed_dict = {}
        for placeholder_name, value in feed_dict.items():
            placeholder = self.graph.get_tensor_by_name(self._tensor_name(placeholder_name))
            _feed_dict.update({placeholder: value})
        _feed_dict.update({self.is_training: True})
        return _feed_dict

    def _fill_fetches(self, fetches=None, default=None):
        fetches = fetches or default
        if isinstance(fetches, str):
            _fetches = self._tensor_name(fetches)
        elif isinstance(fetches, (tuple, list)):
            _fetches = []
            for fetch in fetches:
                _fetches.append(self._tensor_name(fetch))
        elif isinstance(fetches, dict):
            _fetches = dict()
            for key, fetch in fetches.items():
                _fetches.update({key: self._tensor_name(fetch)})
        else:
            _fetches = fetches
        return _fetches

    def train(self, fetches=None, feed_dict=None):   # pylint: disable=arguments-differ
        """ Train the model with the data provided

        Parameters
        ----------
        fetches : an arbitrarily nested structure of `tf.Operation`s and `tf.Tensor`s
        feed_dict : a dict with input data, where key is a placeholder name and value is a numpy value

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        Tensorflow Session run (https://www.tensorflow.org/api_docs/python/tf/Session#run)
        """
        with self.graph.as_default():
            _feed_dict = self._fill_feed_dict(feed_dict)
            if fetches is None:
                _fetches = tuple()
            else:
                _fetches = self._fill_fetches(fetches, default=None)
            _, output = self.session.run([self.train_step, _fetches], feed_dict=_feed_dict)
        return output

    def predict(self, fetches=None, feed_dict=None):      # pylint: disable=arguments-differ
        """ Get predictions on the data provided

        Parameters
        ----------
        fetches : an arbitrarily nested structure of `tf.Operation`s and `tf.Tensor`s
        feed_dict : a dict with input data, where key is a placeholder name and value is a numpy value

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        Notes
        -----
        The only difference between `predict` and `train` is that `train` also executes a `train_step` operation
        which involves calculating and applying gradients and thus chainging model weights.

        See also
        --------
        Tensorflow Session run (https://www.tensorflow.org/api_docs/python/tf/Session#run)
        """
        with self.graph.as_default():
            _feed_dict = self._fill_feed_dict(feed_dict)
            _fetches = self._fill_fetches(fetches, default='predictions')
            output = self.session.run(_fetches, _feed_dict)
        return output

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model.

        Parameters
        ----------
        path : str - a path to a directory where all model files will be stored

        Examples
        --------
        >>> tf_model = ResNet34()

        Now train the model
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
        path : str - a directory where a model is stored
        graph : str - a filename for a metagraph file
        checkpoint : str - a checkpoint file name or None to load the latest checkpoint

        Examples
        --------
        >>> tf_model = ResNet34(load=True)

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
