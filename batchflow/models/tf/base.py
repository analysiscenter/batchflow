# pylint: disable=undefined-variable, no-name-in-module, import-error
""" Contains base class for tensorflow models """
import os
import glob
import re
import threading

import dill
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

from ... import Config
from ..utils import unpack_fn_from_config
from ..base import BaseModel
from .layers import Mip, Upsample, ConvBlock, Crop
from .losses import softmax_cross_entropy, dice
from .nn import piecewise_constant, cyclic_learning_rate


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'bce': tf.losses.sigmoid_cross_entropy,
    'ce': softmax_cross_entropy,
    'crossentropy': softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'l1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss,
    'dice': dice,
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': piecewise_constant,
    'poly': tf.train.polynomial_decay,
    'cyclic': cyclic_learning_rate,
}


class TFModel(BaseModel):
    r""" Base class for all TensorFlow models.
    Allows to easily define complex neural networks via configuration dictionaries.

    Parameters
    ----------
    inputs : dict
        Mapping from placeholder names (e.g. ``images``, ``labels``, ``masks``) to arguments of their initialization.
        Allows to create placeholders of needed format (shape, dtype, data format) with specified name
        and apply some typical transformations (like one-hot-encoding), if needed.

        If value is a string, then it must point to another key from the input-dict, effectively making an alias.
        By default, ``targets`` is aliased to ``labels`` or ``masks``, if present.
        If value is a tuple, then it must contain all of the arguments below in the same order. ``None`` is reserved
        for using default value.
        If value is a dictionary, then it can omit some of the parameters (default values will be at use).

            dtype : str or tf.DType
                Data type. Default is 'float32'.

            shape : int, None, sequence of ints or Nones
                Tensor shape with channels and without batch size. Default is None.

            classes : int, array-like or None
                If int, then number of classes.
                If array-like, then labels of classes (can be strings or anything else).
                If None, then tensor has no classes. Default is None.

            data_format : {'channels_first', 'channels_last', 'f', 'l'}
                The ordering of the dimensions in the inputs. Can be shortened to ``df`` for brevity.
                Default is 'channels_last'.

            transform : str, callable or None
                If str, then one of predefined transform is used.
                If callable, then it is immediately applied to tensor.
                If None, no transform is applied. Default is None.

                Predefined transforms are:
                    - ``ohe`` - one-hot encoding.
                    - ``mip @ d`` - max intensity projection :class:`~.tf.layers.Mip` with depth ``d`` (should be int).
                    - ``downsample @ d`` - downsampling with a factor ``d`` (should be int).

            name : str
                Name for the transformed tensor.

    loss : str, tuple, dict
        Loss function, might be defined in multiple formats.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'mse'``, ``'ce'``, ``'l1'``, ``'cos'``, ``'hinge'``,
              ``'huber'``, ``'logloss'``, ``'dice'``)
            - a function name from `tf.losses <https://www.tensorflow.org/api_docs/python/tf/losses>`_
              (e.g. ``'absolute_difference'`` or ``'sparse_softmax_cross_entropy'``)
            - callable

        It is possible to compute loss not only with network output and ground truth, but with
        any named tensors in model by passing their names as ``predictions`` and ``targets`` parameters.

        If loss is a callable, then either ``add_loss`` should be set to True, or the callable should
        manually add the result to a loss collection.

        ``loss_collection`` parameter can be used to specify collection of losses.

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': {'name': 'sigmoid_cross_entropy', 'label_smoothing': 1e-6}}``
        - ``{'loss': (tf.losses.huber_loss, {'reduction': tf.losses.Reduction.MEAN})}``
        - ``{'loss': {'name': 'dice', 'predictions': 'body_output', 'targets': 'body_targets'}``
        - ``{'loss': external_loss_fn_with_add_loss_inside}``
        - ``{'loss': {'name': external_loss_fn_without_add_loss, 'add_loss': True}}``
        - ``{'loss': {'name': external_loss_fn, 'add_loss': True, 'loss_collection': tf.GraphKeys.LOSSES}}``

    optimizer : str, tuple, dict
        Optimizer, might be defined in multiple formats.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'Adam'``, ``'Adagrad'``, any optimizer from
              `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_ with or without word `Optimizer`)
            - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
              (e.g. ``'FtlrOptimizer'``)
            - callable

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('Ftlr', {'learning_rate_power': 0})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}``
        - ``{'optimizer': some_optimizer_fn}``

    decay : str, tuple, dict
        Learning rate decay algorithm, might be defined in multiple formats.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'exp'``, ``'invtime'``, ``'naturalexp'``, ``'const'``, ``'poly'``, ``'cyclic'``)
            - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
              (e.g. ``'exponential_decay'``)
            - callable

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('polynomial_decay', {'decay_steps': 10000})}``
        - ``{'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}``

    scope : str or sequence of str
        Subset of variables to optimize during training.
        Value ``''`` is reserved for optimizing all trainable variables.
        Putting ``-`` sign before name stands for complement: optimize everything but the passed scope.

        Examples:

        - ``{'scope': ''}``
        - ``{'scope': 'body/custom_layer'}``
        - ``{'scope': '-body/custom_layer'}``
        - ``{'scope': ['body/custom_layer_1', 'head/custom_layer_2']}``

    train_steps : dict
        Configuration of different training procedures.
        Must be a mapping from string names to dictionary with train parameters like
        loss, decay, scope, optimizer. Those keys support syntax defined above.

        If any of loss, decay, scope, optimizer is defined directly in config, it serves as the default
        value for every train step.

        Optimizer and decay, created at one train step, can be re-used in another. To do so, one can
        pass 'use' key with value corresponding to the name of train step from which you want to borrow optimizer.
        Note that in this case you are still free to change loss-function or scope.

        In order to use particular train step during train, one must pass `train_mode` argument to
        :meth:`.TFModel.train` method.

        Examples:

        Create multiple training procedures:
            - one to optimize the whole network to minimize cross-entropy loss with Adam
            - one to optimize weights only in body to minimize Dice-coefficient loss with RMSProp
            - one to optimize weights in initial block and head to minimize cross-entropy loss with re-used
              optimizer from body

        .. code-block:: python

            {'train_steps': {'whole_network': {'loss': 'ce', 'optimizer': 'Adam', 'scope': ''},
                             'only_body': {'loss': 'dice', 'optimizer': 'RMSProp', 'scope': 'body'}},
                             'ib_and_head': {'loss': 'ce', 'use': 'body', 'scope': ['initial_block', 'head']}}

    session : dict
        Parameters for session configuration. `allow_soft_placement` is always True.
        See `Tensorflow Session parameters <https://www.tensorflow.org/api_docs/python/tf/Session>`_.

    device : str or sequence of str
        Device name(s), e.g. ``'/device:GPU:0'`` (TensorFlow-like format), ``['gpu:1', 'CPU:0']``.
        Regular expressions are also allowed, e.g. ``'GPU:*'``.
        Default behaviour is to use the first available GPU (or CPU if no GPUs are detected).
        See `tf.device <https://www.tensorflow.org/api_docs/python/tf/device>`_ for details.

        Batch size must be divisible by number of devices.

    microbatch : int
        Size of chunks to split every batch into. Allows to process given data sequentially, accumulating gradients
        from microbatches and applying them once in the end. Batch size must be divisible by microbatch size.
        Can be changed later via `microbatch` argument of :meth:`.TFModel.train`.

    initial_block : dict
        Parameters for the input block, usually :class:`~.tf.layers.ConvBlock` parameters.

        The only required parameter here is ``initial_block/inputs`` which should contain a name or
        a list of names from ``inputs`` which tensors will be passed to ``initial_block`` as ``inputs``.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': {'inputs': 'features'}}``
        - ``{'initial_block': {'inputs': 'images', 'layout': 'nac', 'filters':64, 'kernel_size': 7, 'strides': 2}}``

    body : dict
        Parameters for the base network layers, usually :class:`~.tf.layers.ConvBlock` parameters.

    head : dict
        Parameters for the head layers, usually :class:`~.tf.layers.ConvBlock` parameters.

    predictions : str or callable
        An operation applied to the head output to make the predictions tensor which is used in the loss function.
        See see :meth:`.TFModel.output` for details.

    output : dict or list
        Auxiliary operations to apply to the network output. See see :meth:`.TFModel.output` for details.

    common : dict
        Parameters to pass to every part of the network (e.g. initial block, body, head),
        usually :class:`~.tf.layers.ConvBlock` parameters.


    **In order to create your own model, it is recommended to:**

    * Take a look at :class:`.BaseModel`: ``build`` and ``load`` methods inherited from it.

    * Take a look at :class:`~.tf.layers.ConvBlock` since it is a widely used as a building block,
      capable of chaining various operations (convolutions, batch normalizations, etc).

    * Define model defaults (e.g. number of filters, dropout rates, etc) by overriding
      :meth:`.TFModel.default_config`. Those parameters are updated with external configuration dictionary.

    * Define config post-processing by overriding :meth:`~.TFModel.build_config`.
      Its main use is to infer parameters that can't be known in advance (e.g. number of classes, shape of inputs).

    * Override :meth:`~.TFModel.initial_block`, :meth:`~.TFModel.body` and :meth:`~.TFModel.head`, if needed.
      You can either use usual tf-functions, or predefined layers like :class:`~tf.layers.ASPP`.
      Conveniently, 'initial_block' is used to make pre-processing (e.g. reshaping or agressive pooling) of inputs,
      'body' contains the meat of the network flow,
      and 'head' makes sure that the output is compatible with targets.

    * To use layers that behave differently at train/test times, ``is_training`` tensor is predefined and passed
      to model parts (initial block, body, head) as keyword argument.
      You can also get it via :meth:`.TFModel.get_from_attr`.

    * To use layers or decays that behave differently depending on number of iterations done, ``global_step`` tensor is
      predefined and passed to model parts (initial block, body, head) as keyword argument.
      You can also get it via :meth:`.TFModel.get_from_attr`.


    **In order to use existing model, it is recommended to define following keys in configuration dictionary:**

    * ``inputs``: defines input data together with parameters like shape, dtype, number of classes.

    * ``loss``, ``optimizer``, ``decay``, ``scope``.

    * ``initial_block`` sub-dictionary must contain ``inputs`` key with names of tensors to use as network inputs.

    * ``initial_block``, ``body``, ``head``: used to define behaviour of respective part of the network.
      Default behaviour is to support all of the :class:`~.tf.layers.ConvBlock` options.
      For complex models, take a look at default config of the chosen model to learn
      which parameters should be configured.

    """
    def __init__(self, *args, **kwargs):
        self.full_config = Config()
        self.session = kwargs.get('session', None)
        self.graph = tf.Graph() if self.session is None else self.session.graph
        self._graph_context = None
        self._train_lock = threading.Lock()

        # Parameters of batch processing: splitting batches into parts and/or using multiple devices to process data
        self.microbatch = None
        self.devices = []
        self.leading_device = None
        self.device_to_scope = {}
        self.scope_to_device = {}
        self.multi_device = False

        # Private storage for often used tensors
        self._attrs = dict()

        # Save/load things
        self._saver = None
        self.preserve = ['_attrs', 'microbatch',
                         'devices', 'leading_device', 'device_to_scope', 'scope_to_device', 'multi_device']

        super().__init__(*args, **kwargs)

    def store_to_attr(self, attr, graph_item, device=None):
        """ Store `graph_item` to private container."""
        if device is None:
            self._attrs[attr] = graph_item
        else:
            if self._attrs.get(attr) is None:
                self._attrs[attr] = {device: graph_item}
            else:
                self._attrs[attr].update({device: graph_item})

    def get_from_attr(self, attr, device=None, default=None):
        """ Get item from private container or directly from model graph."""
        device = device or self._get_current_device() or self.leading_device
        if attr in self._attrs:
            if isinstance(self._attrs[attr], dict):
                if device in self._attrs[attr]:
                    return self._attrs[attr][device]
            return self._attrs[attr]
        if default is not None:
            return default
        return self._check_tensor(attr, device)

    def _check_tensor(self, name, device=None):
        prefix = self.__class__.__name__ + '/'
        if device is not None:
            if device in self.device_to_scope.keys():
                prefix += self.device_to_scope[device]
            else:
                prefix += device

        pattern = '^' + prefix + '.*' + name + '.*'
        valid = [item for item in self.graph.get_operations() if re.match(pattern, item.name)]
        if len(valid) > 1:
            valid = [item for item in valid if re.match('.*_output$', item.name)]
            if len(valid) != 1:
                raise KeyError("Too many tensors match the '%s' name in  %s model" % (name, type(self).__name__))

        if len(valid) == 1:
            return valid[0].values()[0]
        raise KeyError("Model %s does not have '%s' tensor" % (type(self).__name__, name))

    def build(self, *args, **kwargs):
        """ Build the model. """
        # Get list of all available devices, infer leading device and number of devices
        self.devices = self._get_devices()
        if len(self.devices) > 1:
            self.multi_device = len(self.devices)
        self.leading_device = self.devices[0]

        self.device_to_scope = {item: item[1:].replace(':', '_') for item in self.devices}
        self.scope_to_device = {v: k for k, v in self.device_to_scope.items()}

        # Create model graph. First of all, `is_training` and `global_step` tensors are defined;
        # then, for each device, model architecture is created (with inputs placeholders and all);
        # finally, individual train steps with desired loss, optimizer, decay and scope are created
        with self.graph.as_default():
            with tf.variable_scope(self.__class__.__name__):
                with tf.variable_scope('globals'), tf.device(self.leading_device):
                    is_training = tf.placeholder(tf.bool, name='is_training')
                    self.store_to_attr('is_training', is_training)

                    global_step = tf.Variable(0, trainable=False, name='global_step')
                    self.store_to_attr('global_step', global_step)

                for device in self.devices:
                    with tf.device(device):
                        with tf.variable_scope(self.device_to_scope[device]):
                            self.full_config = self.combine_configs()
                            self._make_inputs(config=self.full_config['inputs'],
                                              data_format=self.full_config.get('common/data_format', 'channels_last'))
                            config = self.build_config()

                            self._build(config)

                if self.session is None:
                    self.create_session(config)

                self._make_train_steps(config)
                self.reset()

    def create_session(self, config=None):
        """ Create TF session """
        config = config or self.full_config
        session_config = config.get('session', default={})
        session_config = {**session_config, **{'allow_soft_placement': True}}
        self.session = tf.Session(config=tf.ConfigProto(**session_config))

    def reset(self):
        """ Reset the trained model to allow a new training from scratch """
        with self.session.graph.as_default(), tf.device(self.leading_device):
            self.session.run(tf.global_variables_initializer())

    def _get_devices(self):
        available_devices = device_lib.list_local_devices()

        # Remove internal `XLA` devices, see `using JIT compilation <https://www.tensorflow.org/xla/jit>`_.
        usable_devices = [device.name for device in available_devices
                          if 'XLA' not in device.name]

        if self.config.get('device'):
            devices = self.config.get('device')
            devices = devices if isinstance(devices, list) else [devices]
            devices = [device for name in devices for device in usable_devices
                       if re.search(name.upper(), device.upper()) is not None]
            devices = [device for i, device in enumerate(devices)
                       if device not in devices[:i]]
        else:
            cpu_devices = [device for device in usable_devices
                           if 'CPU' in device]
            gpu_devices = [device for device in usable_devices
                           if 'GPU' in device]
            if gpu_devices:
                devices = [gpu_devices[0]]
            else:
                devices = [cpu_devices[0]]
        return devices

    def _get_current_device(self):
        scope = tf.get_variable_scope().name
        if '/' in scope:
            device_scope = scope.split('/')[1]
            if device_scope in self.scope_to_device:
                return self.scope_to_device[device_scope]
        return None

    def _make_inputs(self, names=None, config=None, data_format='channels_last'):
        # pylint:disable=too-many-statements
        with tf.variable_scope('inputs'):
            device = self._get_current_device()

            names = names or []
            missing_names = set(names) - set(config.keys())
            if len(missing_names) > 0:
                raise KeyError("Inputs should contain {} names".format(missing_names))

            placeholder_names = set(config.keys())
            tensor_names = set(x.get('name') for x in config.values() if isinstance(x, dict) and x.get('name'))
            wrong_names = placeholder_names & tensor_names
            if len(wrong_names) > 0:
                raise ValueError('Inputs contain duplicate names:', wrong_names)

            # add default aliases
            if 'labels' in config and 'targets' not in config:
                config['targets'] = 'labels'
            elif 'masks' in config and 'targets' not in config:
                config['targets'] = 'masks'
            # if targets is defined in the input dict, these implicit aliases will be overwritten.

            param_names = ('dtype', 'shape', 'classes', 'data_format', 'transform', 'name')
            defaults = dict(data_format=data_format)

            placeholders = dict()
            tensors = dict()
            _inputs = dict()
            for input_name, input_config in config.items():
                if isinstance(input_config, str):
                    continue
                if isinstance(input_config, (tuple, list)):
                    input_config = list(input_config) + [None for _ in param_names]
                    input_config = input_config[:len(param_names)]
                    input_config = dict(zip(param_names, input_config))
                    input_config = dict((k, v) for k, v in input_config.items() if v is not None)
                input_config = {**defaults, **input_config}

                reshape = None
                shape = input_config.get('shape')
                if isinstance(shape, int):
                    shape = (shape,)
                if shape:
                    input_config['shape'] = shape
                    shape = [None] + list(shape)

                _inputs[input_name] = dict(config=input_config)
                self.store_to_attr('_inputs', _inputs)

                if self.has_classes(input_name):
                    dtype = input_config.get('dtype', tf.int64)
                    shape = shape or (None,)
                else:
                    dtype = input_config.get('dtype', 'float')
                tensor = tf.placeholder(dtype, shape, input_name)
                placeholders[input_name] = tensor
                self.store_to_attr(input_name, tensor, device)

                if 'df' in input_config and 'data_format' not in input_config:
                    input_config['data_format'] = input_config['df']
                if input_config.get('data_format') == 'l':
                    input_config['data_format'] = 'channels_last'
                elif input_config.get('data_format') == 'f':
                    input_config['data_format'] = 'channels_first'

                _inputs[input_name] = dict(config=input_config)
                self.store_to_attr('_inputs', _inputs)
                tensor = self._make_transform(input_name, tensor, input_config)

                if isinstance(reshape, (list, tuple)):
                    tensor = tf.reshape(tensor, [-1] + list(reshape))

                name = input_config.get('name')
                if name is not None:
                    tensor = tf.identity(tensor, name=name)
                    self.store_to_attr(name, tensor, device)

                tensors[input_name] = tensor

                _inputs[input_name] = dict(config=input_config, placeholder=placeholders[input_name], tensor=tensor)
                if name is not None:
                    _inputs[name] = _inputs[input_name]
                self.store_to_attr('_inputs', _inputs)

            # check for aliases
            for input_name, input_config in config.items():
                if isinstance(input_config, str) and input_name not in _inputs:
                    _inputs[input_name] = _inputs[input_config]
                    tensors[input_name] = tensors[input_config]
                    placeholders[input_name] = placeholders[input_config]
                    tensor = tf.identity(tensors[input_name], name=input_name)
                    self.store_to_attr(input_name, tensors[input_name], device)

            self.store_to_attr('_inputs', _inputs)
            self.store_to_attr('inputs', tensors)
        return placeholders, tensors

    def _make_transform(self, input_name, tensor, config):
        if config is not None:
            transforms = {
                'ohe': self._make_ohe,
                'mip': self._make_mip,
                'downsample': self._make_downsample
            }

            transform_names = config.get('transform')
            if not isinstance(transform_names, list):
                transform_names = [transform_names]
            for transform_name in transform_names:
                if isinstance(transform_name, str):
                    kwargs = dict()
                    if transform_name.startswith('mip'):
                        parts = transform_name.split('@')
                        transform_name = 'mip'
                        kwargs['depth'] = int(parts[1])
                    elif transform_name.startswith('downsample'):
                        parts = transform_name.split('@')
                        transform_name = 'downsample'
                        kwargs['factor'] = int(parts[1])
                    tensor = transforms[transform_name](input_name, tensor, config, **kwargs)
                elif callable(transform_name):
                    tensor = transform_name(tensor)
                elif transform_name:
                    raise ValueError("Unknown transform {}".format(transform_name))
        return tensor

    def _make_ohe(self, input_name, tensor, config):
        if config.get('shape') is None and config.get('classes') is None:
            raise ValueError("shape and classes cannot be both None for input " +
                             "'{}' with one-hot-encoding transform".format(input_name))

        num_classes = self.num_classes(input_name)
        axis = -1 if self.data_format(input_name) == 'channels_last' else 1
        tensor = tf.one_hot(tensor, depth=num_classes, axis=axis)
        return tensor

    def _make_downsample(self, input_name, tensor, config, factor):
        """ Perform downsampling with the factor given. """
        _ = input_name, config
        size = self.shape(tensor, False)
        if None in size[1:]:
            size = self.shape(tensor, True)
        size = size / factor
        size = tf.cast(size, tf.int32)
        tensor = tf.expand_dims(tensor, -1)
        tensor = tf.image.resize_nearest_neighbor(tensor, size)
        tensor = tf.squeeze(tensor, [-1])
        return tensor

    def _make_mip(self, input_name, tensor, config, depth):
        # mip has to know shape
        if config.get('shape') is None:
            raise ValueError('mip transform requires shape specified in the inputs config')
        if depth is None:
            raise ValueError("mip should be specified as mip @ depth, e.g. 'mip @ 3'")
        tensor = Mip(depth=depth, data_format=self.data_format(input_name))(tensor)
        return tensor

    def to_classes(self, tensor, input_name, name=None):
        """ Convert tensor with labels to classes of ``input_name`` """
        if tensor.dtype in [tf.float16, tf.float32, tf.float64]:
            tensor = tf.argmax(tensor, axis=-1, name=name)
        if self.has_classes(input_name):
            self.store_to_attr('_to_classes', input_name, tensor)
        return tensor

    def _make_train_steps(self, config=None, init=True):
        config = config or self.full_config
        self.microbatch = config.get('microbatch')

        # Wrap parameters from config root as `train_steps`
        if config.get('train_steps') is None:
            config.update({'train_steps': {'': {key: config.get(key) for key in
                                                ('optimizer', 'decay', 'loss', 'scope')}}})
            total = lambda _: tf.losses.get_total_loss(name='_TOTAL_LOSS')
        else:
            total = lambda loss: loss

        # First pass through the config: pass values from higher level, create (and store) all of the optimizers
        optimizers = {}
        for key, subconfig in config['train_steps'].items():
            subconfig.update({key: subconfig.get(key) or config.get(key)
                              for key in ('optimizer', 'decay', 'loss', 'scope')})
            if subconfig.get('optimizer') is not None:
                if optimizers.get(key) is None:
                    optimizers[key] = self._make_optimizer(subconfig)

        # Second pass through the config: create loss, get scope variables, minimize via chosen optimizer
        train_steps = {}
        for key, subconfig in config['train_steps'].items():
            # Create losses for every device, then combine them into one via summation
            device_grads, device_losses, ops = [], [], {}

            for device in self.devices:
                with tf.device(device):
                    with tf.variable_scope(self.device_to_scope[device]), tf.variable_scope(key):
                        loss_ = total(self._make_loss(subconfig, device))
                        loss_ = tf.identity(loss_, name='_DEVICE_LOSS')
                        device_losses.append(loss_)

                        optimizer = optimizers.get(subconfig.get('use')) or optimizers.get(key)

                        # It is important to control dependencies in order to work with layers like batch-normalization
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            scope_collection = self._make_scope(subconfig, device)

                            # Simplest operation for training, created always
                            minimize_op = optimizer.minimize(loss_,
                                                             global_step=self.get_from_attr('global_step'),
                                                             var_list=scope_collection)
                            ops['minimize'] = minimize_op

                            # In order to use microbatches, we need to zero-out some storage, then populate it
                            # with computed gradients, and, finally, apply them to the weights at once
                            if self.microbatch:
                                if not self.multi_device:
                                    zero_op, update_op, apply_op = self._make_microbatch_ops(loss_, optimizer,
                                                                                             var_list=scope_collection)
                                    ops.update({'zero_grads': zero_op,
                                                'update_grads': update_op,
                                                'apply_grads': apply_op})

                            # To use multiple devices, we must compute gradients for every device,
                            # combine them on leading device, and apply updates to the weights on every device
                            if self.multi_device:
                                grad_and_vars = optimizer.compute_gradients(loss_,
                                                                            var_list=scope_collection)
                                device_grads.append(grad_and_vars)

            # Store average loss in the attribute, make operation to apply average gradient to the weights
            with tf.device(self.leading_device):
                loss_name = 'loss' if len(key) == 0 else 'loss_' + key
                loss = tf.reduce_mean(tf.stack(device_losses))
                loss = tf.identity(loss, name=loss_name)
                self.store_to_attr(loss_name, loss)

                if self.multi_device:
                    if not self.microbatch:
                        ops['multi_minimize'] = self._make_multi_op(device_grads, optimizer)

                    else:
                        zero_op, update_op, apply_op = self._make_microbatch_multi_ops(device_grads, optimizer)
                        ops.update({'multi_zero_grads': zero_op,
                                    'multi_update_grads': update_op,
                                    'multi_apply_grads': apply_op})

                # We need to explicitly initialize variable for every optimizer in order to not
                # interfere with capability to reuse optimizers for different train_steps
                if init:
                    self.session.run(tf.variables_initializer(optimizer.variables()))

            # Store all the created operations
            train_steps.update({key: ops})

        self.store_to_attr('train_steps', train_steps)

    def _make_loss(self, config, device):
        loss, args = unpack_fn_from_config('loss', config)
        add_loss = args.pop('add_loss', False)
        loss_collection = args.pop('loss_collection', tf.GraphKeys.LOSSES)

        # Make loss callable
        if loss is None or callable(loss):
            pass
        elif isinstance(loss, str) and hasattr(tf.losses, loss):
            loss = getattr(tf.losses, loss)
        elif isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        else:
            raise ValueError("Unknown loss", loss)

        # Use existing loss from graph or make a new one
        if loss is None:
            if len(tf.losses.get_losses()) == 0:
                raise ValueError("Loss is not defined in the model %s" % self)
            tensor_loss = tf.losses.get_losses(loss_collection=loss_collection)
        else:
            # Fetch all the needed tensors
            inputs = args.pop('inputs', None)
            if inputs is not None:
                if isinstance(inputs, (tuple, list)):
                    tensors = [self.get_from_attr(name, device) for name in inputs]
                elif isinstance(inputs, (dict, Config)):
                    tensors = {name: self.get_from_attr(value, device) for name, value in inputs.items()}
            else:
                predictions_name = args.pop('predictions', 'predictions')
                targets_name = args.pop('targets', 'targets')
                predictions = self.get_from_attr(predictions_name, device)
                targets = self.get_from_attr(targets_name, device)
                tensors = [targets, predictions]

            if isinstance(tensors, list):
                tensor_loss = loss(*tensors, **args)
            elif isinstance(tensors, dict):
                tensor_loss = loss(**tensors, **args)

            if add_loss:
                tf.losses.add_loss(tensor_loss, loss_collection)
        return tensor_loss

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name):
            pass
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name):
            optimizer_name = getattr(tf.train, optimizer_name)
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name + 'Optimizer'):
            optimizer_name = getattr(tf.train, optimizer_name + 'Optimizer')
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            optimizer_args['learning_rate'] = decay_name(**decay_args,
                                                         global_step=self.get_from_attr('global_step'))

        if optimizer_name:
            optimizer = optimizer_name(**optimizer_args)
        else:
            optimizer = None

        return optimizer

    def _make_decay(self, config):
        decay_name, decay_args = unpack_fn_from_config('decay', config)

        if decay_name is None or callable(decay_name):
            pass
        elif isinstance(decay_name, str) and hasattr(tf.train, decay_name):
            decay_name = getattr(tf.train, decay_name)
        elif isinstance(decay_name, str):
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def _make_scope(self, config, device):
        scopes = config.get('scope')
        scopes = [scopes] if isinstance(scopes, str) else scopes
        if not isinstance(scopes, (list, tuple)):
            raise ValueError("'Scope' key should be either string or sequence of strings.")

        total = []
        for scope in scopes:
            model_prefix = self.__class__.__name__ + '/'
            device_prefix = model_prefix + self.device_to_scope[device] + '/'

            if (len(scope) > 0) and (scope[0] in ['-', '_', '^']):
                scope_prefix = device_prefix + scope[1:]
            else:
                scope_prefix = device_prefix + scope

            scope_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope_prefix)
            if (len(scope) > 0) and (scope[0] in ['-', '_', '^']):
                scope_collection = [item for item in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, device_prefix)
                                    if item not in scope_collection]
            total.extend(scope_collection)
        return total

    def _make_microbatch_ops(self, loss, optimizer, var_list):
        with tf.variable_scope('microbatch'):
            # Container to store intermediate values of gradients
            count = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='count')
            grad_accum = [tf.Variable(np.empty(var.shape, dtype=np.float32), trainable=False)
                          for var in var_list]

            # Zero-out operation
            with tf.variable_scope('zero_grads'):
                zero_grad_ops = [var.assign(tf.zeros(var.shape)) for var in grad_accum]
                zero_count_op = count.assign(tf.zeros(shape=(), dtype=tf.float32))
                zero_op = zero_grad_ops + [zero_count_op]
                zero_op = tf.group(zero_op, name='zero_grads_op')

            # Compute gradients and add it to the values in the storage
            with tf.variable_scope('update_grads'):
                grad_and_vars = optimizer.compute_gradients(loss, var_list)
                update_grad_ops = [grad_accum[i].assign_add(g) for i, (g, _) in enumerate(grad_and_vars)
                                   if g is not None]
                update_count_op = count.assign_add(tf.constant(1.0, dtype=tf.float32))
                update_op = update_grad_ops + [update_count_op]
                update_op = tf.group(update_op, name='update_grads_op')

            # Apply gradients from the storage to the actual weights
            with tf.variable_scope('apply_grads'):
                grad_and_vars = [(grad_accum[i] / count, v) for i, (_, v) in enumerate(grad_and_vars)]
                apply_op = optimizer.apply_gradients(grad_and_vars,
                                                     global_step=self.get_from_attr('global_step'))
                apply_op = tf.group(apply_op, name='apply_grads_op')
        return zero_op, update_op, apply_op

    def _make_multi_op(self, gradients, optimizer):
        operations = []
        # Each iteration of this loop works with 'copies' of the same variable on different devices
        for grad_and_vars in zip(*gradients):
            # Average gradients from different devices
            expanded = [tf.expand_dims(g, 0) for g, _ in grad_and_vars if g is not None]
            concatted = tf.concat(expanded, axis=0)
            averaged = tf.reduce_mean(concatted, axis=0)

            # Apply gradient on the leading device, then distribute to the others
            leading_device_variable = grad_and_vars[0][1]
            apply_op = optimizer.apply_gradients([(averaged, leading_device_variable)],
                                                 global_step=self.get_from_attr('global_step'))

            distribute_weights = [v.assign(leading_device_variable) for _, v in grad_and_vars[1:]]
            op = tf.group([apply_op] + distribute_weights, name='apply_weights_op')
            operations.append(op)

        # Combine update operations for every variable into single one
        op = tf.group(operations, name='multi_minimize_op')
        return op

    def _make_microbatch_multi_ops(self, gradients, optimizer):
        global_step = self.get_from_attr('global_step')
        zero_ops = []
        update_ops = []
        apply_ops = []

        with tf.variable_scope('microbatch_multi'):
            count = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='count')
            zero_count_op = count.assign(tf.zeros(shape=(), dtype=tf.float32))
            zero_ops.append(zero_count_op)

            update_count_op = count.assign_add(tf.constant(1.0, dtype=tf.float32))
            update_ops.append(update_count_op)

            for grad_and_vars in zip(*gradients):
                # Leading device variable
                var = grad_and_vars[0][1]
                grad_accum = tf.Variable(np.empty(var.shape, dtype=np.float32), trainable=False)
                zero_grad_op = grad_accum.assign(tf.zeros(var.shape))
                zero_ops.append(zero_grad_op)

                # Average gradients from different devices
                expanded = [tf.expand_dims(g, 0) for g, _ in grad_and_vars if g is not None]
                concatted = tf.concat(expanded, axis=0)
                averaged = tf.reduce_mean(concatted, axis=0)
                update_grad_op = grad_accum.assign_add(averaged)
                update_ops.append(update_grad_op)

                apply_grad_op_ = optimizer.apply_gradients([(grad_accum / count, var)],
                                                           global_step=global_step)
                distribute_weights = [v.assign(var) for _, v in grad_and_vars[1:]]
                apply_grad_op = tf.group([apply_grad_op_] + distribute_weights, name='apply_weights_op')
                apply_ops.append(apply_grad_op)

            zero_op = tf.group(zero_ops, name='multi_zero_grads_op')
            update_op = tf.group(update_ops, name='multi_update_grads_op')
            apply_op = tf.group(apply_ops, name='multi_apply_grads_op')

        return zero_op, update_op, apply_op

    def get_number_of_trainable_vars(self):
        """ Return the number of trainable variable in the model graph """
        arr = np.asarray([np.prod(v.get_shape().as_list()) for v in self.graph.get_collection('trainable_variables')])
        return np.sum(arr)

    def get_tensor_config(self, tensor, **kwargs):
        """ Return tensor configuration.

        Parameters
        ----------
        tensor : str or tf.Tensor
            If str, then name of tensor.

        Returns
        -------
        dict
            tensor config (see :meth:`.TFModel._make_inputs`)

        Raises
        ------
        ValueError shape in tensor configuration isn't int, tuple or list
        """
        inputs = self.get_from_attr('_inputs')

        if tensor in inputs:
            config = inputs[tensor]['config']
            shape = config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                kwargs['shape'] = shape
        elif isinstance(tensor, str):
            try:
                tensor = self.get_from_attr(tensor)
            except KeyError:
                config = {}
            else:
                shape = tensor.get_shape().as_list()[1:]
                data_format = self.full_config.get('common/data_format') or 'channels_last'
                config = dict(dtype=tensor.dtype, shape=shape,
                              name=tensor.name, data_format=data_format)
        else:
            config = {}

        config = {**config, **kwargs}
        return config

    def eval(self, mode):
        """ Change model learning phase. Important to use to control behaviour of layers, that
        perform different operations on train/inference (dropout, batch-norm).

        Parameters
        ----------
        mode : bool or int
            If evaluates to True, then all the layers are set to use `train` behaviour.
            If evaluates to False, then all the layers are set to use `test` behaviour.
        """
        if isinstance(mode, bool):
            mode = int(mode)
        tf.keras.backend.set_learning_phase(mode)

    def _map_name(self, name, device=None):
        if isinstance(name, str):
            return self.get_from_attr(name, device)
        return name

    def _fill_feed_dict(self, feed_dict=None, device=None, is_training=True):
        feed_dict = feed_dict or {}
        _feed_dict = {}
        for placeholder, value in feed_dict.items():
            if self.has_classes(placeholder):
                classes = self.get_tensor_config(placeholder).get('classes')
                if not isinstance(classes, int):
                    classes = self.classes(placeholder)
                    get_indices = np.vectorize(lambda c, arr=classes: np.where(c == arr)[0])
                    value = get_indices(value)
            placeholder = self._map_name(placeholder, device)
            value = self._map_name(value, device)
            _feed_dict.update({placeholder: value})
        if self.get_from_attr('is_training') not in _feed_dict:
            _feed_dict.update({self.get_from_attr('is_training'): is_training})
        return _feed_dict

    def _fill_fetches(self, fetches=None, default=None):
        fetches = fetches or default
        if isinstance(fetches, str):
            _fetches = self._map_name(fetches)
        elif isinstance(fetches, (tuple, list)):
            _fetches = []
            for fetch in fetches:
                _fetches.append(self._map_name(fetch))
        elif isinstance(fetches, dict):
            _fetches = dict()
            for key, fetch in fetches.items():
                _fetches.update({key: self._map_name(fetch)})
        else:
            _fetches = fetches
        return _fetches

    def _recast_output(self, out, ix=None, fetches=None):
        if isinstance(out, np.ndarray):
            fetch = fetches[ix] if ix is not None else fetches
            if isinstance(fetch, str):
                fetch = self.graph.get_tensor_by_name(fetch)
            _to_classes = self.get_from_attr('_to_classes', default={})
            if fetch in _to_classes:
                return self.classes(_to_classes[fetch])[out]
        return out

    def _fill_output(self, output, fetches):
        if isinstance(output, (tuple, list)):
            _output = []
            for i, o in enumerate(output):
                _output.append(self._recast_output(o, i, fetches))
            output = type(output)(_output)
        elif isinstance(output, dict):
            _output = type(output)()
            for k, v in output.items():
                _output.update({k: self._recast_output(v, k, fetches)})
        else:
            output = self._recast_output(output, fetches=fetches)

        return output

    def train(self, fetches=None, feed_dict=None, use_lock=True, train_mode='', microbatch=None, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        fetches : tuple, list
            Sequence of `tf.Operation` and/or `tf.Tensor` to calculate.
        feed_dict : dict
            Input data, where key is a placeholder name and value is a numpy value.
        use_lock : bool
            If True, the whole train step is locked, thus allowing for multithreading.
        train_mode : str or sequence of str
            Name(s) of train step(s) to optimize. Regular expressions are allowed.
            If multiple train steps are selected (either via passing a sequence or by using regular expression),
            then all of them are optimized sequentially.
        microbatch : int
            Size of chunks to split every batch into. Note that if this option was not specified
            in the model configuration, the first invocation of this method would create additional operations.

        Returns
        -------
        tuple, list
            Calculated values of tensors in `fetches` in the same order.

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly as named arguments.

        Examples
        --------
        .. code-block:: python

            model.train(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as:

        .. code-block:: python

            model.train(fetches='loss', images=B('images'), labels=B('labels'))

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
        """
        with self.graph.as_default():
            train_steps = self.get_from_attr('train_steps')
            # Use microbatch size from either args or config, and if
            # necessary ops for microbatch-training are absent, create them
            if microbatch is not False:
                microbatch = microbatch or self.microbatch
            self.microbatch = microbatch

            if (microbatch) and (len(list(train_steps.values())[0]) == 1):
                self._make_train_steps(self.full_config, init=False)

            if microbatch is True: # if config option is set to True, but train option left unspectified,
                microbatch = False # it is faster to pretend that there is no microbatching

            # `feed_dict` processing: updating it with all kwargs,
            # optionally splitting it for microbatch train, resulting in list of feed_dicts,
            # updating every of them with `_fill_feed_dict` so tensorflow can work with it
            feed_dict = feed_dict or {}
            feed_dict = {**feed_dict, **kwargs}

            # `fetches` and `train_mode` processing
            if fetches is None:
                _fetches = tuple()
            else:
                names = [fetches] if isinstance(fetches, str) else fetches
                _fetches = self._fill_fetches(names, default=None)

            if not isinstance(train_mode, (tuple, list)):
                train_mode = [train_mode]

            # Acquire lock so only one `train` is active at a time
            if use_lock:
                self._train_lock.acquire()

            if train_steps:
                for mode in train_mode:
                    if mode in train_steps.keys():
                        train_fetches = [train_steps[mode]]
                    else:
                        train_fetches = [train_step for name, train_step in train_steps.items()
                                         if re.search(mode, name) is not None]

                    if not microbatch:
                        if not self.multi_device:
                            output = self._vanilla_train(train_fetches, _fetches, feed_dict)
                        else:
                            output = self._multi_train(train_fetches, _fetches, feed_dict)
                    else:
                        feed_dicts = self._split_feed_dict(feed_dict, size=microbatch)

                        if not self.multi_device:
                            outputs = self._microbatch_train(train_fetches, _fetches, feed_dicts)
                        else:
                            outputs = self._microbatch_multi_train(train_fetches, _fetches, feed_dicts)

                        outputs = [[item[i] for item in outputs] for i, _ in enumerate(names)]
                        output = [np.mean(outputs[i]) if 'loss' in name else outputs[i][-1]
                                  for i, name in enumerate(names)]

                    output = output[0] if isinstance(fetches, str) else output
            else:
                output = None

            if use_lock:
                self._train_lock.release()
            return self._fill_output(output, _fetches)

    def _split_feed_dict(self, feed_dict, num_parts=None, size=None):
        splitted = {}
        for key, value in feed_dict.items():
            if hasattr(value, '__len__'):
                if num_parts is None:
                    num_parts = len(value) // size
                if len(value) % num_parts != 0:
                    raise ValueError('Batch size must be divisible by {}, but is {}'.format(num_parts, len(value)))
                splitted[key] = np.array_split(value, num_parts)

        splitted_ = [{key: value[i] for key, value in splitted.items()}
                     for i in range(num_parts)]
        return splitted_

    def _vanilla_train(self, train_fetches, fetches, feed_dict):
        # Get list of train operations to run
        all_fetches = [ops['minimize'] for ops in train_fetches]
        if fetches is not None:
            all_fetches += [fetches]

        # Fill feed_dict with placeholders
        _fd = self._fill_feed_dict(feed_dict, is_training=True)
        *_, output = self.session.run(all_fetches, feed_dict=_fd)

        return output

    def _multi_train(self, train_fetches, _fetches, feed_dict):
        # Get list of train operations to run
        all_fetches = [ops['multi_minimize'] for ops in train_fetches]
        if _fetches is not None:
            all_fetches += [_fetches]

        # Split batch into even parts for every device, then run complex operation
        # that computes gradients on every device, combines them on the leading one,
        # and finally sends updates back to devices
        _feed_dicts = self._split_feed_dict(feed_dict, num_parts=self.multi_device)

        _fd = {}
        for part, device in zip(_feed_dicts, self.devices):
            _fd = {**_fd, **self._fill_feed_dict(part, device)}
        *_, output = self.session.run(all_fetches, feed_dict=_fd)

        return output

    def _microbatch_train(self, train_fetches, _fetches, feed_dicts):
        _feed_dicts = [self._fill_feed_dict(part, is_training=True) for part in feed_dicts]

        outputs = []
        for ops in train_fetches:
            # Get train operations to run
            zero_op, update_op, apply_op = ops['zero_grads'], \
                                           ops['update_grads'], \
                                           ops['apply_grads']
            all_fetches = [update_op]
            if _fetches is not None:
                all_fetches += [_fetches]

            # For every train step, zero out gradient accumulators,then update them with gradients,
            # computed on each of `feed_dicts`, and finally apply accumulated values to weights
            self.session.run(zero_op, feed_dict=_feed_dicts[0])
            for _fd in _feed_dicts:
                _, _output = self.session.run(all_fetches, feed_dict=_fd)
                outputs += [_output]
            self.session.run(apply_op, feed_dict=_feed_dicts[-1])
        return outputs

    def _microbatch_multi_train(self, train_fetches, _fetches, feed_dicts):
        outputs = []
        for ops in train_fetches:
            # Get train operations to run
            zero_op, update_op, apply_op = ops['multi_zero_grads'], \
                                           ops['multi_update_grads'], \
                                           ops['multi_apply_grads']
            all_fetches = [update_op]
            if _fetches is not None:
                all_fetches += [_fetches]

            # For every microbatch run complex operation that computes gradients on every device,
            # combines them on the leading one, and stores into accumulator. When the last
            # microbatch is processed, accumulated value is applied to the weights on leading device,
            # and finally distributed to other devices
            for i, feed_dict in enumerate(feed_dicts):
                _feed_dicts = self._split_feed_dict(feed_dict, num_parts=self.multi_device)
                _fd = {}
                for part, device in zip(_feed_dicts, self.devices):
                    _fd = {**_fd, **self._fill_feed_dict(part, device)}

                if i == 0:
                    self.session.run(zero_op, feed_dict=_fd)

                _, _output = self.session.run(all_fetches, feed_dict=_fd)
                outputs += [_output]
            self.session.run(apply_op, feed_dict=_fd)
        return outputs

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get predictions on the data provided

        Parameters
        ----------
        fetches : tuple, list
            Sequence of `tf.Operation` and/or `tf.Tensor` to calculate.
        feed_dict : dict
            Input data, where key is a placeholder name and value is a numpy value.

        Returns
        -------
        tuple, list
            Calculated values of tensors in `fetches` in the same order.

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly.

        Examples
        --------
        .. code-block:: python

            model.predict(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as:

        .. code-block:: python

            model.predict(fetches='loss', images=B('images'), labels=B('labels'))

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
        """
        with self.graph.as_default():
            feed_dict = {} if feed_dict is None else feed_dict
            feed_dict = {**feed_dict, **kwargs}
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=False)
            _fetches = self._fill_fetches(fetches, default='predictions')
            output = self.session.run(_fetches, _feed_dict)
        return self._fill_output(output, _fetches)

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model and most of important attributes.

        Parameters
        ----------
        path : str
            Path to a directory where all model files will be stored.

        Examples
        --------
        .. code-block:: python

            tf_model = ResNet34()
            tf_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34
        """
        with self.graph.as_default():
            if not os.path.exists(path):
                os.makedirs(path)
            if self._saver is None:
                self._saver = tf.train.Saver()
            self._saver.save(self.session, os.path.join(path, 'model'), *args,
                             global_step=self.get_from_attr('global_step'), **kwargs)

        preserved = dict()
        for attribute_name in self.preserve:
            attribute = getattr(self, attribute_name)
            preserved[attribute_name] = self._to_names(attribute)
        with open(os.path.join(path, 'attributes.dill'), 'wb') as f:
            dill.dump(preserved, f)

    def _to_names(self, graph_item):
        # Base cases
        if isinstance(graph_item, tf.Tensor):
            return ('Tensor', graph_item.name)
        if isinstance(graph_item, tf.Operation):
            return ('Operation', graph_item.name)
        if isinstance(graph_item, tf.Variable):
            return ('Variable', graph_item.op.name)
        if isinstance(graph_item, (bool, str, int, float)) or graph_item is None:
            return graph_item

        # Handle different containers
        if isinstance(graph_item, (list, tuple, np.ndarray)):
            return type(graph_item)([self._to_names(item) for item in graph_item])
        if isinstance(graph_item, (dict, Config)):
            return type(graph_item)({key: self._to_names(graph_item[key]) for key in graph_item.keys()})
        raise ValueError('Unrecognized type of value.')

    def load(self, path, graph=None, checkpoint=None, *args, **kwargs):
        """ Load a TensorFlow model and most important attributes from files

        Parameters
        ----------
        path : str
            Directory where a model is stored.
        graph : str
            Filename for a metagraph file.
        checkpoint : str or None
            If str, then checkpoint file name.
            If None, then load the latest checkpoint.

        Examples
        --------
        .. code-block:: python

            resnet = ResNet34(load=dict(path='/path/to/models/resnet34'))
            tf_model.load(path='/path/to/models/resnet34')
        """
        _ = args, kwargs
        self.graph = tf.Graph()

        with self.graph.as_default():
            if graph is None:
                graph_files = glob.glob(os.path.join(path, '*.meta'))
                graph_files = [os.path.splitext(os.path.basename(graph))[0] for graph in graph_files]
                all_steps = []
                for _graph in graph_files:
                    try:
                        step = int(_graph.split('-')[-1])
                    except ValueError:
                        pass
                    else:
                        all_steps.append(step)
                graph = '-'.join(['model', str(max(all_steps))]) + '.meta'

            graph_path = os.path.join(path, graph)
            saver = tf.train.import_meta_graph(graph_path)

            if checkpoint is None:
                checkpoint_path = tf.train.latest_checkpoint(path)
            else:
                checkpoint_path = os.path.join(path, checkpoint)

            self.create_session()
            saver.restore(self.session, checkpoint_path)

        with open(os.path.join(path, 'attributes.dill'), 'rb') as dill_file:
            restored = dill.load(dill_file)

        for attribute_name, value in restored.items():
            setattr(self, attribute_name, self._to_graph_items(value))
        self.preserve = list(restored.keys())

    def _to_graph_items(self, name):
        # Base cases
        if isinstance(name, (bool, str, int, float)) or name is None:
            return name

        # Handle different containers
        if isinstance(name, (list, tuple, np.ndarray)):
            if len(name) == 2:
                type_, name_ = name
                if type_ == 'Variable':
                    with self.graph.as_default():
                        return tf.global_variables(name_)[0]
                if type_ == 'Tensor':
                    return self.graph.get_tensor_by_name(name_)
                if type_ == 'Operation':
                    return self.graph.get_operation_by_name(name_)
            return type(name)([self._to_graph_items(item) for item in name])

        if isinstance(name, (dict, Config)):
            return type(name)({key: self._to_graph_items(name[key]) for key in name.keys()})
        raise ValueError('Unrecognized type of value.')


    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        """ Transform inputs. Usually used for initial preprocessing, e.g. reshaping, downsampling etc.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        name : str
            Scope name.

        Notes
        -----
        For other parameters see :class:`~.tf.layers.ConvBlock`.

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('initial_block', **kwargs)
        if kwargs.get('layout'):
            return ConvBlock(name=name, **kwargs)(inputs)
        return inputs

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers which produce a network embedding.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        name : str
            Scope name.

        Notes
        -----
        For other parameters see :class:`~.tf.layers.ConvBlock`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        .. code-block:: python

            MyModel.body(inputs, layout='ca ca ca', filters=[128, 256, 512], kernel_size=3)
        """
        kwargs = cls.fill_params('body', **kwargs)
        if kwargs.get('layout'):
            return ConvBlock(name=name, **kwargs)(inputs)
        return inputs

    @classmethod
    def make_encoder(cls, inputs, name='encoder', **kwargs):
        """ Build the body and return the last tensors of each spatial resolution.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        name : str
            Scope name.
        kwargs : dict
            Body params.

        Notes
        -----
        In order to use custom class as encoder, body of the network must create ``group-i/output`` tensors.
        An example of this can be seen in :class:`~.Xception`.
        """
        config = cls.fill_params('body', **kwargs)
        order = config.get('order')
        config = config[order[0]] if order else config

        steps = None
        for loc in ['num_stages', 'num_blocks', 'num_layers']:
            _steps = cls.get(loc, config=config)
            _steps = len(_steps) if hasattr(_steps, '__len__') else _steps
            steps = steps or _steps

        with tf.variable_scope(name):
            x = cls.add_block('body', config=cls.fill_params('body', **kwargs),
                              inputs=inputs, defaults=kwargs)

            scope = tf.get_default_graph().get_name_scope()
            template_name = '/'.join([scope, 'body{}',
                                      'group-{}', 'output:0'])

            encoder_tensors = [inputs]
            for i in range(steps):
                tensor_name = template_name.format('/{}'.format(order[0]) if order else '', i)
                x = tf.get_default_graph().get_tensor_by_name(tensor_name)
                encoder_tensors.append(x)
        return encoder_tensors

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ The last network layers which produce predictions. Usually used to make network output
        compatible with the `targets` tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        name : str
            Scope name.

        Notes
        -----
        For other parameters see :class:`~.tf.layers.ConvBlock`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

        .. code-block:: python

            MyModel.head(network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits:

        .. code-block:: python

            MyModel.head(network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """
        kwargs = cls.fill_params('head', **kwargs)
        if kwargs.get('layout'):
            return ConvBlock(name=name, **kwargs)(inputs)
        return inputs


    def output(self, inputs, predictions=None, ops=None, **kwargs):
        """ Add output operations to the model graph, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : tf.Tensor or a sequence of tf.Tensors
            input tensors. If list, all resulting outputs will have postfix _{i} where i is an index of input.

        predictions : str or callable
            Operation to apply to the network output to obtain tensor which is used in loss computation.

            If str, then one of predefined operations:
                - 'sigmoid' - ``sigmoid(inputs)``
                - 'proba' - ``softmax(inputs)``
                - 'labels' - ``argmax(inputs)``
                - 'softplus' - ``softplus(inputs)``

            If callable, then user-defined operation.

        ops : sequence, dict or OrderedDict
            Auxiliary operations to apply.

            If sequence, then operations to apply. Transformed tensors are stored with the same name, as operation
            If dict, then mapping from prefixes to operations. Transformed tensors are stored with
            the prefixed name of the operation.

            For multi-output models ensure that an ordered dict is used (e.g. :class:`~collections.OrderedDict`).

        Raises
        ------
        ValueError if the number of inputs does not equal to the number of prefixes
        TypeError if inputs is not a Tensor or a sequence of Tensors

        Examples
        --------
        .. code-block:: python

            config = {
                'output': ['proba', 'labels']
            }

        However, if one of the placeholders also has a name 'labels', then it will be lost as the model
        will rewrite the name 'labels' with an output. In this case dict might be more convenient:

        .. code-block:: python

            config = {
                'output': {'predicted': ['proba', 'labels']}
            }

        Now the output will be stored under names 'predicted_proba' and 'predicted_labels'.
        """
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}

        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, tf.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            self._add_output_op(tensor, predictions, 'predictions', 'predictions_' + str(i), **kwargs)

            for prefix in ops.keys():
                if prefix:
                    ctx = tf.variable_scope(prefix)
                    ctx.__enter__()
                else:
                    ctx = None

                attr_prefix = prefix + '_' if prefix else ''
                attr_postfix = '_' + str(i) if len(inputs) > 1 else ''

                for oper in ops[prefix]:
                    if callable(oper):
                        name = oper.__name__
                    else:
                        name = oper

                    if oper is None:
                        attr_name = attr_prefix[:-1] + (name or '') + attr_postfix
                    else:
                        attr_name = attr_prefix + name + attr_postfix

                    self._add_output_op(tensor, oper, oper, attr_name, **kwargs)

                if ctx:
                    ctx.__exit__(None, None, None)

        _predictions = [self.get_from_attr('predictions_' + str(i)) for i in range(len(inputs))]
        self._add_predictions(_predictions, 'predictions', 'predictions', **kwargs)

    def _add_output_op(self, inputs, oper, name, attr_name, **kwargs):
        device = self._get_current_device()

        if oper is None:
            self._add_output_identity(inputs, name, attr_name, device, **kwargs)
        elif oper == 'softplus':
            self._add_output_softplus(inputs, name, attr_name, device, **kwargs)
        elif oper == 'sigmoid':
            self._add_output_sigmoid(inputs, name, attr_name, device, **kwargs)
        elif oper == 'proba':
            self._add_output_proba(inputs, name, attr_name, device, **kwargs)
        elif oper == 'labels':
            self._add_output_labels(inputs, name, attr_name, device, **kwargs)
        elif callable(oper):
            self._add_output_callable(inputs, oper, None, attr_name, device, **kwargs)

    def _add_output_identity(self, inputs, name, attr_name, device, **kwargs):
        _ = kwargs
        x = tf.identity(inputs, name=name)
        self.store_to_attr(attr_name, x, device)
        return x

    def _add_output_softplus(self, inputs, name, attr_name, device, **kwargs):
        _ = kwargs
        proba = tf.nn.softplus(inputs, name=name)
        self.store_to_attr(attr_name, proba, device)

    def _add_output_sigmoid(self, inputs, name, attr_name, device, **kwargs):
        _ = kwargs
        proba = tf.sigmoid(inputs, name=name)
        self.store_to_attr(attr_name, proba, device)

    def _add_output_proba(self, inputs, name, attr_name, device, **kwargs):
        axis = self.channels_axis(kwargs['data_format'])
        proba = tf.nn.softmax(inputs, name=name, axis=axis)
        self.store_to_attr(attr_name, proba, device)

    def _add_output_labels(self, inputs, name, attr_name, device, **kwargs):
        class_axis = self.channels_axis(kwargs.get('data_format'))
        predicted_classes = tf.argmax(inputs, axis=class_axis, name=name)
        self.store_to_attr(attr_name, predicted_classes, device)

    def _add_output_callable(self, inputs, oper, name, attr_name, device, **kwargs):
        _ = kwargs
        x = oper(inputs)
        name = name or oper.__name__
        self.store_to_attr(attr_name, x, device)
        return x

    def _add_predictions(self, predictions, name, attr_name, **kwargs):
        _ = kwargs
        device = self._get_current_device()
        if len(predictions) > 1:
            tf.stack(predictions, name=name)
        else:
            predictions = predictions[0]
            tf.identity(predictions, name=name)
        self.store_to_attr(attr_name, predictions, device)

    @classmethod
    def default_config(cls):
        """ Define model defaults.

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in :meth:`~.TFModel.build_config` or when calling :meth:`.Pipeline.init_model`.

        Examples
        --------
        .. code-block:: python

            @classmethod
            def default_config(cls):
                config = TFModel.default_config()
                config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                               pool_size=3, pool_strides=2)
                config['body/filters'] = 32
                config['head'] = dict(layout='cnadV', dropout_rate=.2)
                return config
        """
        config = Config()
        config['inputs'] = {}
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['predictions'] = None
        config['output'] = None
        config['optimizer'] = ('Adam', dict())
        config['decay'] = (None, dict())
        config['scope'] = ''
        config['common'] = {'batch_norm': {'momentum': .1}}

        return config

    @classmethod
    def fill_params(cls, _name, **kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = Config(config.get(_name))
        _config = _config + kwargs # Update _config with kwargs (addition order is important)
        config = {**config['common'], **_config}
        return config

    def combine_configs(self):
        config = self.default_config() + self.config
        return config

    def build_config(self, config=None):
        """ Define model's architecture configuration.

        * Don't forget to call ``super().build_config(names)`` in the beginning.

        * Define parameters for :meth:`.TFModel.initial_block`, :meth:`.TFModel.body`, :meth:`.TFModel.head`,
          which depend on inputs.

        * Don't forget to return ``config`` at the end.

        Examples
        --------
        .. code-block:: python

            def build_config(self, config=None):
                config = super().build_config(config)
                config['head/num_classes'] = self.num_classes('targets')
                return config
        """
        config = config or self.full_config
        inputs = config.get('initial_block/inputs')

        if isinstance(inputs, str):
            if not config.get('common/data_format'):
                config['common/data_format'] = self.data_format(inputs)
            config['initial_block/inputs'] = self.get_from_attr('inputs')[inputs]
        elif isinstance(inputs, list):
            # If inputs use different data formats, you need to manually control this parameter
            # in model parts (initial_block, body, head).
            if not config.get('common/data_format'):
                config['common/data_format'] = self.data_format(inputs[0])
            config['initial_block/inputs'] = [self.get_from_attr('inputs')[name]
                                              for name in inputs]
        else:
            raise ValueError('initial_block/inputs should be specified with a name or a list of names.')

        config['head/targets'] = self.get_from_attr('targets')
        return config

    def _add_block(self, name, config, inputs, defaults=None):
        if defaults is None:
            defaults = {'is_training': self.get_from_attr('is_training'),
                        'global_step': self.get_from_attr('global_step'),
                        **config['common']}

        config = config[name]
        order = config.get('order') or [None]

        tensor = inputs
        for item in order:
            block = config if item is None else config[item]
            args = {'name': '{}/{}'.format(name, item) if item else '{}'.format(name),
                    **defaults}

            if callable(block):
                tensor = block(tensor, **args)
            elif isinstance(block, dict):
                args = {**args, **block}
                block_class = block.get('block_class') or self
                tensor = getattr(block_class, name)(inputs=tensor, **args)
            else:
                raise TypeError('NN blocks can be configured as a function, a dict with parameters or \
                                 sequence of these, instead got {} in {}'.format(type(block), name))
        return tensor

    @classmethod
    def add_block(cls, name, config, inputs, defaults=None):
        """ Add all model parts of the same type. """
        if name not in config:
            config = {name: config}
        defaults = defaults or {}
        tensor = cls._add_block(cls, name, config, inputs, defaults)
        return tensor

    def _build(self, config=None):
        config = config or self.full_config
        inputs = config.pop('initial_block/inputs')
        x = self._add_block('initial_block', config, inputs=inputs)
        x = self._add_block('body', config, inputs=x)
        output = self._add_block('head', config, inputs=x)
        self.output(output, predictions=config['predictions'], ops=config['output'], **config['common'])

    def data_format(self, tensor, **kwargs):
        """ Return the tensor data format (channels_last or channels_first).

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        data_format : str
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('data_format')

    def has_classes(self, tensor):
        """ Check if a tensor has classes defined in the config. """
        config = self.get_tensor_config(tensor)
        has = config.get('classes') is not None
        return has

    def classes(self, tensor):
        """ Return the classes. """
        config = self.get_tensor_config(tensor)
        classes = config.get('classes')
        if isinstance(classes, int):
            return np.arange(classes)
        return np.asarray(classes)

    def num_classes(self, tensor):
        """ Return the number of classes. """
        if self.has_classes(tensor):
            classes = self.classes(tensor)
            return classes if isinstance(classes, int) else len(classes)
        return self.get_num_channels(tensor)

    def get_num_channels(self, tensor, **kwargs):
        """ Return the number of channels in the tensor.

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of channels : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        shape = (None,) + config.get('shape')
        channels_axis = self.channels_axis(tensor, **kwargs)
        return shape[channels_axis] if shape else None

    @classmethod
    def num_channels(cls, tensor, data_format='channels_last'):
        """ Return number of channels in the input tensor.

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        shape = tensor.get_shape().as_list()
        axis = cls.channels_axis(data_format)
        return shape[axis]

    def get_shape(self, tensor, **kwargs):
        """ Return the tensor shape without batch dimension.

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('shape')

    @classmethod
    def shape(cls, tensor, dynamic=False):
        """ Return shape of the input tensor without batch size.

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            If True, returns tensor which represents shape. If False, returns list of ints and/or Nones.

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        return shape[1:]

    def get_spatial_dim(self, tensor, **kwargs):
        """ Return the tensor spatial dimensionality (without batch and channels dimensions).

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of spatial dimensions : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return len(config.get('shape')) - 1

    @classmethod
    def spatial_dim(cls, tensor):
        """ Return spatial dim of the input tensor (without channels and batch dimension).

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        dim : int
        """
        return len(tensor.get_shape().as_list()) - 2

    def get_spatial_shape(self, tensor, **kwargs):
        """ Return the tensor spatial shape (without batch and channels dimensions).

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        spatial shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        data_format = config.get('data_format')
        shape = config.get('shape')[:-1] if data_format == 'channels_last' else config.get('shape')[1:]
        return shape

    @classmethod
    def spatial_shape(cls, tensor, data_format='channels_last', dynamic=False):
        """ Return the tensor spatial shape (without batch and channels dimensions).

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            If True, returns tensor which represents shape. If False, returns list of ints and/or Nones.

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        axis = slice(1, -1) if data_format == "channels_last" else slice(2, None)
        return shape[axis]

    def get_batch_size(self, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor.

        Parameters
        ----------
        tensor : str or tf.Tensor
            If str, then name of pre-stored tensor.

        Returns
        -------
        batch size : int or None
        """
        if isinstance(tensor, tf.Tensor):
            pass
        elif isinstance(tensor, str):
            tensor = self.get_from_attr(tensor)
        else:
            raise TypeError("Tensor can be tf.Tensor or string, but given %s" % type(tensor))

        return tensor.get_shape().as_list()[0]

    @classmethod
    def batch_size(cls, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor.

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        batch size : int or None
        """
        return tensor.get_shape().as_list()[0]

    @classmethod
    def channels_axis(cls, data_format='channels_last'):
        """ Return the integer channels axis based on string data format. """
        return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1


    @classmethod
    def crop(cls, inputs, resize_to, data_format='channels_last'):
        """ Crop input tensor to a shape of a given image.
        If resize_to does not have a fully defined shape (resize_to.get_shape() has at least one None),
        the returned tf.Tensor will be of unknown shape except the number of channels.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        resize_to : tf.Tensor
            Tensor which shape the inputs should be resized to.
        data_format : str {'channels_last', 'channels_first'}
            Data format.
        """
        return Crop(resize_to=resize_to, data_format=data_format)(inputs)

    @classmethod
    def se_block(cls, inputs, ratio, name='se', **kwargs):
        """ Squeeze and excitation block.

        Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

        Parameters
        ----------
        ratio : int
            Squeeze ratio for the number of filters.
        """
        x = ConvBlock(**{**kwargs, 'layout': 'S', 'attention_mode': 'se', 'ratio': ratio, 'name': name})(inputs)
        return x

    @classmethod
    def scse_block(cls, inputs, ratio=2, name='scse', **kwargs):
        """ Concurrent spatial and channel squeeze and excitation.

        Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
        in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

        Parameters
        ----------
        ratio : int, optional
            Squeeze ratio for the number of filters in spatial squeeze
            and channel excitation block. Default is 2.
        """
        x = ConvBlock(**{**kwargs, 'layout': 'S', 'attention_mode': 'scse', 'ratio': ratio, 'name': name})(inputs)
        return x

    @classmethod
    def upsample(cls, inputs, factor=None, resize_to=None, layout='b', name='upsample', **kwargs):
        """ Upsample input tensor.

        Parameters
        ----------
        inputs : tf.Tensor or tuple of two tf.Tensor
            Tensor to resize.
        factor : int
            Upsamping scale.
        resize_to : tf.Tensor
            Tensor which shape is used to resize the output.
        layout : str
            Resizing technique, a sequence of:
            - R - use residual connection with bilinear additive upsampling (must be the first symbol)
            - b - bilinear resize
            - B - bilinear additive upsampling
            - N - nearest neighbor resize
            - t - transposed convolution
            - X - subpixel convolution

        Returns
        -------
        tf.Tensor
        """
        x = Upsample(factor=factor, layout=layout, name=name, **kwargs)(inputs)
        if resize_to is not None:
            x = cls.crop(x, resize_to, kwargs['data_format'])
        return x
