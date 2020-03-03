""" Eager version of TorchModel. """
import os
import re
import warnings
import threading
import inspect
from collections import OrderedDict
from functools import partial
from pprint import pprint

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .utils import unpack_fn_from_config, get_shape
from .layers import ConvBlock
from .losses import CrossEntropyLoss
from ... import Config



LOSSES = {
    'mse': nn.MSELoss,
    'bce': nn.BCEWithLogitsLoss,
    'ce': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'absolutedifference': nn.L1Loss,
    'l1': nn.L1Loss,
    'cosine': nn.CosineSimilarity,
    'cos': nn.CosineSimilarity,
    'hinge': nn.HingeEmbeddingLoss,
    'huber': nn.SmoothL1Loss,
    'logloss': CrossEntropyLoss,
}

DECAYS = {
    'exp': torch.optim.lr_scheduler.ExponentialLR,
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'step': torch.optim.lr_scheduler.StepLR,
    'multistep': torch.optim.lr_scheduler.MultiStepLR,
    'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
}

DECAYS_DEFAULTS = {
    torch.optim.lr_scheduler.ExponentialLR : dict(gamma=0.96),
    torch.optim.lr_scheduler.LambdaLR : dict(lr_lambda=lambda epoch: 0.96**epoch),
    torch.optim.lr_scheduler.StepLR: dict(step_size=30),
    torch.optim.lr_scheduler.MultiStepLR: dict(milestones=[30, 80]),
    torch.optim.lr_scheduler.CosineAnnealingLR: dict(T_max=None)
}



class TorchModel:
    r""" Base class for eager Torch models.

    Parameters
    ----------
    config : dict, :class:`~Config`
        Configuration of model creation. Below are the valid keys.

    inputs : dict, optional
        Mapping from placeholder names (e.g. ``images``, ``labels``, ``masks``) to arguments of their initialization.
        Allows to create placeholders of needed shape and data format and initialize model before
        first pass of actual batch data (thus explicitly imposing shapes).

        Value must be a dictionary with parameters. If some parameters are omitted, then defaults will be at use.
        Valid keys are:

            dtype : str or torch.dtype
                Data type. Default is 'float32'.

            shape : int, None, sequence of ints or Nones
                Tensor shape with channels and without batch size. Default is None.

            classes : int, array-like or None
                If int, then number of classes.
                If None, then tensor has no classes. Default is None.

    placeholder_batch_size : int
        If `inputs` is specified with all the required shapes, then it serves as size of batch dimension during
        placeholder (usually np.ndarrays with zeros) creation. Default value is 2.

    loss : str, tuple, dict, list
        Loss function, might be defined in multiple formats.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.
        If list, then sequence of losses in previous formats.

        Name must be one of:
            - short name (e.g. ``'mse'``, ``'ce'``, ``'l1'``, ``'cos'``, ``'hinge'``,
              ``'huber'``, ``'logloss'``, ``'dice'``)
            - a class name from `torch losses <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
              (e.g. ``'PoissonNLL'`` or ``'TripletMargin'``)
            - callable

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': ('KLDiv', {'reduction': 'none'})``
        - ``{'loss': {'name': MyCustomLoss, 'epsilon': 1e-6}}``
        - ``{'loss': my_custom_loss_fn}``
        - ``{'loss': ['dice', 'bce']}``

    optimizer : str, tuple, dict
        Optimizer, might be defined in multiple formats.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'Adam'``, ``'Adagrad'``, any optimizer from
              `torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_)
            - a class with ``Optimizer`` interface
            - a callable which takes model parameters and optional args

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('SparseAdam', {'lr': 0.01})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': {'name': MyCustomOptimizer, momentum=0.95}}``

    decay : str, tuple, dict
        Learning rate decay algorithm, might be defined in multiple formats.
        All decays require to have ``n_iters`` as a key in a configuration
        dictionary that contains the number of iterations in one epoch.

        If str, then short ``name``.
        If tuple, then ``(name, *args)``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:

        - short name (``'exp'``, ``'invtime'``, ``'naturalexp'``, ``'const'``, ``'poly'``)
        - a class name from `torch.optim.lr_scheduler
          <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
          (e.g. ``'LambdaLR'``) except ``'ReduceLROnPlateau'``.
        - a class with ``_LRScheduler`` interface
        - a callable which takes optimizer and optional args

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('StepLR', {'steps_size': 10000})}``
        - ``{'decay': {'name': MyCustomDecay, 'decay_rate': .5}``

    n_iters : int
        Frequency of making step of learning rate decay.

    train_steps : dict
        Configuration of different training procedures.
        Must be a mapping from string names to dictionary with train parameters like
        loss, optimizer, decay, n_iters. Those keys support syntax defined above.

        If any of loss, optimizer, decay, n_iters is defined directly in config, it serves as the default
        value for every train step.

        Optimizer and decay, created at one train step, can be re-used in another. To do so, one can
        pass 'use' key with value corresponding to the name of train step from which you want to borrow optimizer.
        Note that in this case you are still free to change loss-function or scope.

        In order to use particular train step during train, one must pass `train_mode` argument to
        :meth:`.TorchModel.train` method.

        Examples:

        Create multiple training procedures:
            - one to optimize weights to minimize cross-entropy with Adam
            - one to optimize weights to minimize Dice-coefficient loss with RMSProp
            - one to optimize weights to minimize cross-entropy loss with re-used optimizer from previous

        .. code-block:: python

            {'train_steps': {'adam_ce': {'loss': 'ce', 'optimizer': 'Adam'},
                             'rmsprop_dice': {'loss': 'dice', 'optimizer': 'RMSProp'}},
                             'rmsprop_ce': {'loss': 'ce', 'use': 'rmsprop_dice'}}

    device : str, torch.device or sequence
        If str, a device name (e.g. 'cpu' or 'gpu:0'). Regular expressions are also allowed (e.g. 'gpu:*').
        If torch.device, then device to be used.
        If sequence, then each entry must be in one of previous formats, and batch data is paralleled across them.
        Default behaviour is to use one (and only one) device of the best available type (priority to GPU over CPU).

    benchmark : bool
        Whether to optimize network's forward pass after the first batch. Can speed up training if shapes of inputs
        are constant.

    profile : bool
        Whether to collect stats of model training timings.
        If True, then stats can be accessed via `profile_info` attribute or :meth:`.show_profile_info` method.

    sync_frequency : int
        How often to apply accumulated gradients to the weights. Default value is to apply them after each batch.

    microbatch : int, bool or None
        Also known as virtual batch. If int, then size of chunks to split every batch into.
        Allows to process given data sequentially, accumulating gradients from microbatches and applying them
        once in the end. Can be changed later in the `train` method. Batch size must be divisible by microbatch size.
        If True, then every batch is split into individual items (same as microbatch equals 1).
        If False or None, then feature is not used. Default is not to use microbatching.

    order : sequence
        Defines sequence of network blocks in the architecture. Default is initial_block -> body -> head.
        Each element of the sequence must be either a string, a tuple or a dict.
        If string, then it is used as name of method to use, as config key to use, as name in model repr.
        For example, ``'initial_block'`` stands for using ``self.initial_block`` with config[`initial_block`]
        as parameters, and model representation would show this part of network as `initial_block`.
        If tuple, then it must have three elements: (block_name, config_name, method).
        If dict, then it must contain three keys: `block_name`, `config_name`, `method`.
        In cases of tuple and dict, `method` can also be callable.

    initial_block : dict
        User-defined module or parameters for the input block, usually
        :class:`~.eager_torch.layers.ConvBlock` parameters.

        If ``initial_block/inputs`` is specified with a name or list of names,
        then it should contain names from ``inputs`` with info about shapes of tensors to be passed to `initial_block`.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': dict(inputs='features')}``
        - ``{'initial_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``
        - ``{'initial_block': MyCustomModule(some_param=1, another_param=2)}``

    body : dict or nn.Module
        User-defined module or parameters for the base network layers,
        usually :class:`~.eager_torch.layers.ConvBlock` parameters.

    head : dict or nn.Module
        User-defined module or parameters for the head layers,
        usually :class:`~.eager_torch.layers.ConvBlock` parameters.

    predictions : str or callable
        An operation applied to the head output to make the predictions tensor which is used in the loss function.
        See :meth:`.TorchModel.output` for details.

    output : dict or list
        Auxiliary operations to apply to network predictions. See :meth:`.TorchModel.output` for details.

    common : dict
        Default parameters for all blocks (see :class:`~.eager_torch.layers.ConvBlock`).


    **In order to create your own model, it is recommended to:**

    * Take a look at :class:`~.eager_torch.layers.ConvBlock` since it is widely used as a building
      block almost everywhere.

    * Define model defaults (e.g. number of filters, dropout rates, etc) by overriding
      :meth:`.TorchModel.default_config`. Those parameters are then updated with external configuration dictionary.

    * Define config post-processing by overriding :meth:`~.TorchModel.build_config`.
      Its main use is to infer parameters that can't be known in advance (e.g. number of classes, shape of inputs).

    * Override :meth:`~.TorchModel.initial_block`, :meth:`~.TorchModel.body` and :meth:`~.TorchModel.head`, if needed.
      You can either use usual `Torch layers <https://pytorch.org/docs/stable/nn.html>`_,
      or predefined layers like :class:`~eager_torch.layers.PyramidPooling`.
      Conveniently, 'initial_block' is used to make pre-processing (e.g. reshaping or agressive pooling) of inputs,
      'body' contains the meat of the network flow, and 'head' makes sure that the output is compatible with targets.


    **In order to use existing model, it is recommended to:**

    * If ``inputs`` key defines shapes for all tensors in ``initial_block/inputs``, then model is created off of
      placeholders (tensors with all zeros); otherwise, the first batch data is used to create model.

    * ``loss``, ``optimizer``, ``decay`` keys.

    * ``initial_block`` sub-dictionary with ``inputs`` key with names of tensors to use as network inputs.

    * ``initial_block``, ``body``, ``head`` keys are used to define behaviour of respective part of the network.
      Default behaviour is to support all of the :class:`~.eager_torch.layers.ConvBlock` options.
      For complex models, take a look at default config of the chosen model to learn
      which parameters should be configured.
    """
    def __init__(self, config=None):
        self.full_config = Config(config)
        self.config = Config(config)
        self.train_lock = threading.Lock()

        self.input_names = None
        self.input_shapes = None
        self.target_shape = None
        self.classes = None
        self.model = None
        self.device = None
        self.devices = []
        self.train_steps = None

        self.sync_counter = 0
        self.microbatch = None

        self.iter_info = {}
        self.profilers = []
        self.profile_info = None
        self.preserve = ['full_config', 'input_shapes', 'target_shape', 'classes',
                         'model',
                         'train_steps', 'sync_counter', 'microbatch']

        load = self.config.get('load')
        build = self.config.get('build', default=load is None)
        if load:
            self.load(**load)
        if build:
            self.build()

    def reset(self):
        """ Allows to recreate model from scratch. """
        self.model = None
        self.iter_info = {}


    def build(self):
        """ Build the model. """
        self.full_config = self.combine_configs()
        self._get_devices()
        self._get_placeholder_shapes()
        self.full_config = self.build_config()

        # If the inputs are set in config with their shapes we can build right away
        if self.input_shapes:
            self._build()

    @classmethod
    def default_config(cls):
        """ Define model defaults.

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in :meth:`~.TorchModel.build_config` or when calling :meth:`.Pipeline.init_model`.

        Examples
        --------
        .. code-block:: python

            @classmethod
            def default_config(cls):
                config = TorchModel.default_config()
                config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                               pool_size=3, pool_strides=2)
                config['body/filters'] = 32
                config['head'] = dict(layout='cnadV', dropout_rate=.2)
                return config
        """
        config = Config()
        config['inputs'] = {}
        config['placeholder_batch_size'] = 2

        config['device'] = None
        config['benchmark'] = True
        config['profile'] = False
        config['microbatch'] = None
        config['sync_frequency'] = 1

        config['train_steps'] = None
        config['loss'] = None
        config['optimizer'] = 'Adam'
        config['decay'] = None

        config['order'] = ['initial_block', 'body', 'head']
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['common'] = {}

        config['predictions'] = None
        config['output'] = None
        return config

    def combine_configs(self):
        """ Combine default configuration and the external one. """
        config = self.default_config() + self.config
        return config

    def build_config(self):
        """ Define model's architecture configuration.

        * Don't forget to call ``super().build_config(names)`` in the beginning.

        * Define parameters for :meth:`.TorchModel.initial_block`, :meth:`.TorchModel.body`, :meth:`.TorchModel.head`,
          which depend on inputs.

        * Dont forget to return ``config`` at the end.

        Examples
        --------
        .. code-block:: python

            def build_config(self, names=None):
                config = super().build_config(names)
                config['head/filters'] = self.num_classes('targets')
                return config
        """
        config = self.full_config

        if config.get('inputs'):
            inputs_config = config['inputs']

            # Add default aliases
            if 'targets' not in inputs_config:
                if 'labels' in inputs_config:
                    inputs_config['targets'] = inputs_config['labels']
                elif 'masks' in inputs_config:
                    inputs_config['targets'] = inputs_config['masks']

            # Fetch default data format for all the parts of the network
            inputs = config.get('initial_block/inputs')
            if isinstance(inputs, str):
                data_format = inputs_config.get(inputs, {}).get('data_format')
            elif isinstance(inputs, (tuple, list)):
                data_format = inputs_config.get(inputs[0], {}).get('data_format')
            else:
                data_format = 'channels_first'
            config['common/data_format'] = config.get('common/data_format') or data_format

        config['head/target_shape'] = self.target_shape
        config['head/classes'] = self.classes

        if config.get('head/units') is None:
            config['head/units'] = self.classes
        if config.get('head/filters') is None:
            config['head/filters'] = self.classes
        return config


    def _get_devices(self):
        devices = self.full_config.get('device')
        if devices is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            devices = devices if isinstance(devices, list) else [devices]
            available_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())] + ['cpu']
            for dev in devices:
                if isinstance(dev, torch.device):
                    self.devices.append(dev)
                elif isinstance(dev, str):
                    dev_ = dev.lower()
                    dev_ = dev_.replace('gpu', 'cuda')
                    dev_ = dev_.replace('cpu:0', 'cpu')

                    devices = [torch.device(device) for device in available_devices
                               if re.search(dev_, device.lower()) is not None]
                    self.devices.extend(devices)
                else:
                    raise TypeError('Wrong device type: {}'.format(type(dev)))
            self.devices = [device for i, device in enumerate(self.devices)
                            if device not in self.devices[:i]]
            self.device = self.devices[0]

        torch.backends.cudnn.benchmark = self.full_config.get('benchmark', 'cuda' in self.device.type)

    def _get_placeholder_shapes(self):
        config = self.full_config

        input_names = config.pop('initial_block/inputs', default=None)
        if input_names is not None:
            batch_size = config.get('placeholder_batch_size', 2)
            input_names = input_names if isinstance(input_names, (tuple, list)) else [input_names]

            shapes = []
            for name in input_names:
                cfg = config['inputs'].get(name, {})
                if 'shape' in cfg:
                    shapes.append((batch_size, *cfg['shape']))
                else:
                    shapes.append(None)

            if None not in shapes:
                self.input_shapes = shapes
            self.input_names = input_names

        if config.get('inputs'):
            classes, shapes = [], []
            for name in ['labels', 'masks', 'targets']:
                cfg = config['inputs'].get(name, {})
                if 'classes' in cfg:
                    classes.append(cfg['classes'])
                if 'shape' in cfg:
                    shapes.append(cfg['shape'])
            if len(classes) == 1:
                self.classes = classes[0]
            if len(shapes) == 1:
                self.target_shape = (batch_size, *shapes[0])
                if self.classes is None:
                    self.classes = shapes[0][0]


    def _build(self, inputs=None):
        config = self.full_config
        order = config.get('order')

        inputs = inputs or self._placeholder_data()

        blocks = []
        for item in order:
            if isinstance(item, str):
                block_name = config_name = method = item
            elif isinstance(item, tuple) and len(item) == 3:
                block_name, config_name, method = item
            elif isinstance(item, dict):
                block_name = item['block_name']
                config_name = item.get('config_name', block_name)
                method = item.get('method', config_name)

            inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs
            block = self._make_block(config_name, method, config, inputs)
            if block is not None:
                block.to(self.device)
                inputs = block(inputs)
                blocks.append((block_name, block))

        self.model = nn.Sequential(OrderedDict(blocks))
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, self.devices)
        else:
            self.model.to(self.device)

        self.train_steps = self._make_train_steps(config)

    def _placeholder_data(self):
        data = [np.zeros(shape, dtype=np.float32) for shape in self.input_shapes]
        data = self._fill_param(data)
        return data

    def _make_block(self, name, method, config, inputs):
        if isinstance(config[name], nn.Module):
            block = config[name]
        elif isinstance(config[name], dict):
            config = {**config['common'], **config[name]}
            if 'module' in config:
                module = config['module']
                if isinstance(module, nn.Module):
                    block = module
                else:
                    kwargs = config.get('module_kwargs', {})
                    if 'inputs' in inspect.getfullargspec(module.__init__)[0]:
                        kwargs = {'inputs': inputs, **kwargs}
                    block = module(*config.get('module_args', []), **kwargs)
            else:
                method = getattr(self, method) if isinstance(method, str) else method
                block = method(inputs=inputs, **config)
        else:
            raise ValueError('{} must be configured either as nn.Module or dictionary, got {}'.format(name, config))
        return block


    def _make_train_steps(self, config):
        # Wrap parameters from config root as `train_steps`
        if config.get('train_steps') is None:
            config['train_steps'] = {'': {key: config.get(key) for key in
                                          ('loss', 'optimizer', 'decay', 'n_iters')}}

        # First pass through the config: pass values from higher level, create (and store) all of the optimizers
        optimizers = {}
        for key, subconfig in config['train_steps'].items():
            subconfig.update({key: subconfig.get(key) or config.get(key)
                              for key in ('loss', 'optimizer', 'decay', 'n_iters')})
            if subconfig.get('optimizer') is not None:
                if optimizers.get(key) is None:
                    optimizers[key] = self._make_optimizer(subconfig)

        # Second pass through the config: create loss, get scope variables, minimize via chosen optimizer
        train_steps = {}
        for key, subconfig in config['train_steps'].items():
            loss = self._make_loss(subconfig)
            optimizer, decay = optimizers.get(subconfig.get('use')) or optimizers.get(key)
            step = {
                'loss': loss,
                'optimizer': optimizer,
                'decay': decay,
                'n_iters': subconfig.get('n_iters'),
            }
            train_steps.update({key: step})

        return train_steps

    def _make_loss(self, config):
        res = unpack_fn_from_config('loss', config)
        res = res if isinstance(res, list) else [res]

        losses = []
        for loss, args in res:
            loss_fn = None
            if isinstance(loss, str):
                if hasattr(nn, loss):
                    loss = getattr(nn, loss)
                elif hasattr(nn, loss + "Loss"):
                    loss = getattr(nn, loss + "Loss")
                else:
                    loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
            elif isinstance(loss, type):
                pass
            elif isinstance(loss, nn.Module):
                loss_fn = loss
            elif callable(loss):
                loss_fn = partial(loss, **args)
            else:
                raise ValueError("Loss is not defined in the model %s" % self.__class__.__name__)
            loss_fn = loss_fn or loss(**args)
            if isinstance(loss_fn, nn.Module):
                loss_fn.to(device=self.device)
            losses.append(loss_fn)
        return losses

    def _make_optimizer(self, config):
        optimizer, optimizer_args = unpack_fn_from_config('optimizer', config)

        if callable(optimizer) or isinstance(optimizer, type):
            pass
        elif isinstance(optimizer, str) and hasattr(torch.optim, optimizer):
            optimizer = getattr(torch.optim, optimizer)
        else:
            raise ValueError("Unknown optimizer", optimizer)

        if optimizer:
            optimizer = optimizer(self.model.parameters(), **optimizer_args)
        else:
            raise ValueError("Optimizer is not defined", optimizer)

        decay, decay_args = self._make_decay(config)
        if decay is not None:
            decay = decay(optimizer, **decay_args)
        return optimizer, decay

    def _make_decay(self, config):
        decay, decay_args = unpack_fn_from_config('decay', config)
        n_iters = config.get('n_iters')

        if decay is None:
            return decay, decay_args
        if 'n_iters' not in config:
            raise ValueError("Missing required key ``'n_iters'`` in the cofiguration dict.")

        if callable(decay) or isinstance(decay, type):
            pass
        elif isinstance(decay, str) and hasattr(torch.optim.lr_scheduler, decay):
            decay = getattr(torch.optim.lr_scheduler, decay)
        elif decay in DECAYS:
            decay = DECAYS.get(decay)
        else:
            raise ValueError("Unknown learning rate decay method", decay)

        if decay in DECAYS_DEFAULTS:
            decay_dict = DECAYS_DEFAULTS.get(decay).copy()
            if decay == DECAYS['cos']:
                decay_dict.update(T_max=n_iters)
            decay_dict.update(decay_args)
            decay_args = decay_dict.copy()
        return decay, decay_args


    @classmethod
    def get_defaults(cls, name, kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = Config(config.get(name))
        _config = _config + (kwargs or {})
        config = {**config['common'], **_config}
        return config

    @classmethod
    def initial_block(cls, inputs, **kwargs):
        """ Transform inputs. Usually used for initial preprocessing, e.g. reshaping, downsampling etc.

        Notes
        -----
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        kwargs = cls.get_defaults('initial_block', kwargs)
        if kwargs.get('layout') or kwargs.get('base_block'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Base layers which produce a network embedding.

        Notes
        -----
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        kwargs = cls.get_defaults('body', kwargs)
        if kwargs.get('layout') or kwargs.get('base_block'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def head(cls, inputs, target_shape, classes, **kwargs):
        """ The last network layers which produce predictions. Usually used to make network output
        compatible with the `targets` tensor.

        Notes
        -----
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        _ = target_shape, classes
        kwargs = cls.get_defaults('head', kwargs)
        if kwargs.get('layout') or kwargs.get('base_block'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    def information(self, config=True, devices=True, train_steps=True, model=False, misc=True):
        """ Show information about model configuration, used devices, train steps, architecture and more. """
        template = '\n##### {}:'

        if config:
            print(template.format('Config'))
            pprint(self.full_config.config)

        if devices:
            print(template.format('Devices'))
            print('Leading device is {}'.format(self.device, ))
            if self.devices:
                _ = [print('Device {} is {}'.format(i, d)) for i, d in enumerate(self.devices)]

        if train_steps:
            print(template.format('Train steps'))
            pprint(self.train_steps)

        if model:
            print(template.format('Model'))
            print(self.model)

        if misc:
            print(template.format('Additional info'))
            if self.input_shapes:
                _ = [print('Input {} has shape {}'.format(i, s)) for i, s in enumerate(self.input_shapes)]
            if self.target_shape:
                print('Target has shape {}'.format(self.target_shape))

            if self.model:
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print('\nTotal number of parameters in model: {}'.format(num_params))

            iters = {key: value.get('iter', 0) for key, value in self.train_steps.items()}
            print('\nTotal number of passed training iterations: {}'.format(sum(list(iters.values()))))
            if len(iters) > 1:
                print('Number of training iterations for individual train steps:')
                pprint(iters)

            print(template.format('Last iteration params'))
            pprint(self.iter_info)

    @property
    def info(self):
        """ Show information about model configuration, used devices, train steps and more. """
        self.information()

    def show_profile_info(self, per_iter=False, sortby=None, limit=10, parse=False):
        """ Show stored profiling information with varying levels of details. """
        if (self.profile_info is None) or parse:
            self._parse_profilers()

        if self.device.type == 'cpu':
            columns = ['ncalls', 'CPU_tottime', 'CPU_cumtime', 'CPU_tottime_avg']
            if sortby is None:
                sortby = ('CPU_tottime', 'sum') if per_iter is False else 'CPU_tottime'
        else:
            columns = ['ncalls', 'CUDA_cumtime', 'CUDA_cumtime_avg']
            if sortby is None:
                sortby = ('CUDA_cumtime', 'sum') if per_iter is False else 'CUDA_cumtime'

        if per_iter is False:
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.reset_index().groupby(['name']).agg(aggs)
                      .sort_values(sortby, ascending=False)[:limit])
        else:
            result = (self.profile_info.reset_index().set_index(['iter', 'name'])[columns]
                      .sort_values(['iter', sortby], ascending=[True, False])
                      .groupby(level=0).apply(lambda df: df[:limit]).droplevel(0))
        return result

    def _parse_profilers(self):
        us_in_s = 1000.0 * 1000.0

        indices, values = [], []
        for i, profiler in enumerate(self.profilers):
            for evt in profiler.function_events.key_averages():
                indices.append((i, evt.key))
                row_dict = {
                    'ncalls': evt.count,
                    'CPU_tottime': evt.self_cpu_time_total / us_in_s,
                    'CPU_cumtime': evt.cpu_time_total / us_in_s,
                    'CUDA_cumtime': evt.cuda_time_total / us_in_s,
                }
                values.append(row_dict)
        multiindex = pd.MultiIndex.from_tuples(indices, names=['iter', 'name'])

        self.profile_info = pd.DataFrame(values, index=multiindex,
                                         columns=['ncalls', 'CPU_tottime', 'CPU_cumtime', 'CUDA_cumtime'])
        self.profile_info['CPU_tottime_avg'] = self.profile_info['CPU_tottime'] / self.profile_info['ncalls']
        self.profile_info['CUDA_cumtime_avg'] = self.profile_info['CUDA_cumtime'] / self.profile_info['ncalls']


    def set_debug_mode(self, mode=True):
        """ Changes representation of model to a more or less detailed.
        By default, model representation reduces the description of the most complex modules.
        """
        if self.model is None:
            raise ValueError('Model is not initialized yet. ')
        self.model.apply(lambda module: setattr(module, 'debug', mode))


    def save_graph(self, log_dir=None, **kwargs):
        """ Save model graph for later visualization via tensorboard.

        Parameters
        ----------
        logdir : str
            Save directory location. Default is `runs/CURRENT_DATETIME_HOSTNAME`, which changes after each run.
            Use hierarchical folder structure to compare between runs easily,
            e.g. ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them from within tensorboard.

        Examples
        --------
        To easily check model graph inside Jupyter Notebook, run::

        model.save_graph()
        %load_ext tensorboard
        %tensorboard --logdir runs/

        Or, using command line::
        tensorboard --logdir=runs
        """
        # Import here to avoid unnecessary tensorflow imports inside tensorboard
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir, **kwargs)
        writer.add_graph(self.model, self._placeholder_data())
        writer.close()


    def _fill_value(self, value):
        if value.dtype not in [np.float32, 'float32']:
            value = value.astype(np.float32)

        value = torch.from_numpy(value)
        if self.device:
            value = value.to(self.device)
        return value

    def _fill_param(self, inputs):
        if isinstance(inputs, (tuple, list)):
            inputs = [self._fill_value(item) for item in inputs]
        else:
            inputs = self._fill_value(inputs)
        return inputs

    def _fill_input(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError('Use either positional or keyword arguments in `train` call.')

        if kwargs:
            for name in ['labels', 'masks', 'targets']:
                if name in kwargs:
                    targets = kwargs.pop(name)

            args = [kwargs.get(name) for name in (self.input_names or list(kwargs.keys()))]
            args.append(targets)
        return tuple([self._fill_param(arg) for arg in args])

    def _fill_output(self, fetches, outputs):
        fetches = fetches if fetches is not None else []
        _fetches = [fetches] if isinstance(fetches, str) else fetches

        output = []
        for f in _fetches:
            if f in outputs:
                v = outputs[f]
                if isinstance(v, (torch.Tensor, torch.autograd.Variable)):
                    v = v.detach().cpu().numpy()
                output.append(v)
            else:
                raise KeyError('Unknown value to fetch', f)

        output = output[0] if isinstance(fetches, str) else type(fetches)(output)
        return output


    def train(self, *args, feed_dict=None, fetches=None, use_lock=False, train_mode='',
              accumulate_grads=True, sync_frequency=True, microbatch=True, profile=False, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        args
            Arguments to be passed directly into the model.
        feed_dict : dict
            If ``initial_block/inputs`` are set, then this argument allows to pass data inside,
            with keys being names and values being actual data.
        fetches : tuple, list
            Sequence of tensor names to calculate and return.
        use_lock : bool
            If True, the whole train step is locked, thus allowing for multithreading.
        train_mode : str or sequence of str
            Name(s) of train step(s) to optimize. Regular expressions are allowed.
            If multiple train steps are selected (either via passing a sequence or by using regular expression),
            then all of them are optimized sequentially.
        accumulate_grads : bool
            If True, then gradients from different train modes are accumulated and applied once at the end.
            If False, then gradients are applied for each of the train modes separately.
        sync_frequency : int, bool or None
            If int, then how often to apply accumulated gradients to the weights.
            If True, then value from config is used (default value is to apply gradients after each batch of data).
            If False or None, then gradients are applied after each batch of data.
        microbatch : int, bool or None
            If int, then size of chunks to split every batch into. Allows to process given data sequentially,
            accumulating gradients from microbatches and applying them once in the end.
            If True, then value from config is used (default value is not to use microbatching).
            If False or None, then microbatching is not used.
        profile : bool
            Whether to collect stats of model training timings.
            If True, then stats can be accessed via `profile_info` attribute or :meth:`.show_profile_info` method.
        kwargs : dict
            Additional named arguments directly passed to `feed_dict`.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same order.

        Examples
        --------
        .. code-block:: python

            model.train(B('images'), B('labels'), fetches='loss')
        """
        config = self.full_config
        *inputs, targets = self._fill_input(*args, **{**(feed_dict or {}), **kwargs})

        if sync_frequency is True:
            sync_frequency = config['sync_frequency']
        elif sync_frequency is False or sync_frequency is None:
            sync_frequency = 1
        train_mode = train_mode if isinstance(train_mode, (tuple, list)) else [train_mode]

        if microbatch:
            if microbatch is True:
                microbatch = config.get('microbatch')
            else:
                microbatch = microbatch or config.get('microbatch')

        if microbatch:
            microbatch = 1 if microbatch is True else microbatch
            steps = len(targets) // microbatch
            splitted_inputs = [[item[i:i + microbatch] for item in inputs] for i in range(0, len(targets), microbatch)]
            splitted_targets = [targets[i:i + microbatch] for i in range(0, len(targets), microbatch)]
        else:
            steps = 1
            splitted_inputs = [inputs]
            splitted_targets = [targets]


        if self.model is None:
            if isinstance(splitted_inputs[0], (list, tuple)):
                self.input_shapes = [get_shape(item) for item in splitted_inputs[0]]
            else:
                self.input_shapes = get_shape(splitted_inputs[0])

            self.target_shape = get_shape(splitted_targets[0])
            if self.classes is None:
                if len(self.target_shape) > 1: # segmentation
                    self.classes = self.target_shape[1]

            self.build_config()
            self._build(splitted_inputs[0])

        self.model.train()

        profile = profile or config.profile
        if profile:
            profiler = torch.autograd.profiler.profile(use_cuda='cpu' not in self.device.type)
            profiler.__enter__()

        if use_lock:
            self.train_lock.acquire()

        outputs = []
        for i in range(steps):
            _inputs = splitted_inputs[i]
            _targets = splitted_targets[i]

            output = self._train(*_inputs, _targets, fetches=fetches, train_mode=train_mode,
                                 accumulate_grads=accumulate_grads, sync_frequency=sync_frequency*steps)

            outputs.append(output)

        if use_lock:
            self.train_lock.release()

        outputs = [outputs] if isinstance(fetches, str) else outputs
        output = []
        for i, _ in enumerate(outputs[0]):
            lst = [np.asarray(item[i]) for item in outputs]
            output.append(np.concatenate(lst, axis=0) if lst[0].size != 1 else np.mean(lst))
        output = output[0] if isinstance(fetches, str) else output

        if profile:
            profiler.__exit__(None, None, None)
            self.profilers.append(profiler)

        self.iter_info.update({'microbatch': microbatch,
                               'sync_frequency': sync_frequency,
                               'steps': steps,
                               'train_mode': train_mode,
                               'actual_model_inputs_shape': [get_shape(item) for item in _inputs],
                               'actual_model_outputs_shape': get_shape(_targets),
                               })
        return output

    def _train(self, *args, fetches=None, train_mode='', accumulate_grads=True, sync_frequency=True):
        *inputs, targets = args
        inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs

        output_container = {}

        if not accumulate_grads:
            predictions = self.model(inputs)

        for mode in train_mode:
            if mode in self.train_steps.keys():
                train_fetches = [(mode, self.train_steps[mode])]
            else:
                train_fetches = [(name, train_step) for name, train_step in self.train_steps.items()
                                 if re.search(mode, name) is not None]

            mode_loss = 0
            for name, step in train_fetches:
                loss_fn, optimizer, decay = step['loss'], step['optimizer'], step['decay']

                if 'initialized' not in step:
                    optimizer.zero_grad()
                    step['initialized'] = True

                if accumulate_grads:
                    predictions = self.model(inputs)
                loss = sum([loss_fn_(predictions, targets) for loss_fn_ in loss_fn]) / len(loss_fn)
                mode_loss += loss
                loss.backward()
                step['iter'] = step.get('iter', 0) + 1

                if self.sync_counter >= sync_frequency:
                    self.sync_counter = 1
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    self.sync_counter += 1

                if decay:
                    if step.get('current_iter', 0) >= step['n_iters']:
                        decay.step()
                        step['current_iter'] = 0
                    step['current_iter'] = step.get('current_iter', 0) + 1

                output_container['loss' + '_'*int(len(name) > 0) + name] = loss
            output_container['loss' + '_'*int(len(mode) > 0) + mode] = mode_loss
        output_container['predictions'] = predictions

        config = self.full_config
        additional_outputs = self.output(inputs=predictions, predictions=config['predictions'],
                                         ops=config['output'])
        output_container = {**output_container, **additional_outputs}
        output = self._fill_output(fetches, output_container)
        return output


    def predict(self, *args, targets=None, feed_dict=None, train_mode='', fetches=None, **kwargs):
        """ Get predictions on the data provided.

        Parameters
        ----------
        args : sequence
            Arguments to be passed directly into the model.
        feed_dict : dict
            If ``initial_block/inputs`` are set, then this argument allows to pass data inside,
            with keys being names and values being actual data.
        targets : ndarray, optional
            Targets to calculate loss.
        fetches : tuple, list
            Sequence of tensors to fetch from the model.
        train_mode : str
            Exact name of train step to use to calculate loss.
        kwargs : dict
            Additional named arguments directly passed to `feed_dict`.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same order.

        Examples
        --------
        .. code-block:: python

            model.predict(B('images'), targets=B('labels'), fetches='loss')
        """
        feed_dict = {**(feed_dict or {}), **kwargs}
        if len(feed_dict) == 1:
            _, value = feed_dict.popitem()
            args = (*args, value)
        if feed_dict:
            if targets is not None and 'targets' in feed_dict.keys():
                warnings.warn("`targets` already present in `feed_dict`, so those passed as keyword arg won't be used")
            *inputs, targets = self._fill_input(*args, **feed_dict)
        else:
            inputs = self._fill_input(*args)
            if targets is not None:
                targets = self._fill_input(targets)[0]
        inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs

        self.model.eval()

        with torch.no_grad():
            output_container = {}
            predictions = self.model(inputs)

            if targets is not None:
                if train_mode in self.train_steps.keys():
                    step = self.train_steps[train_mode]
                else:
                    raise ValueError('`train_mode` must reference exact `train_step`.')

                loss_fn = step['loss']
                loss = sum([loss(predictions, targets) for loss in loss_fn]) / len(loss_fn)
                output_container['loss' + '_'*int(len(train_mode) > 0) + train_mode] = loss
            output_container['predictions'] = predictions

        config = self.full_config
        additional_outputs = self.output(inputs=predictions, predictions=config['predictions'],
                                         ops=config['output'])
        output_container = {**output_container, **additional_outputs}
        output = self._fill_output(fetches, output_container)
        return output


    def output(self, inputs, predictions=None, ops=None):
        """ Add output operations to the model, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : torch.Tensor or a sequence of torch.Tensors
            Input tensors.

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

        outputs = {}
        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(tensor)))

            prefix = [*ops.keys()][i]
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '')
            for oper in ops[prefix]:
                name, output = self._add_output_op(tensor, oper, oper, attr_prefix)
                outputs[name] = output
        return outputs

    def _add_output_op(self, inputs, oper, name, attr_prefix):
        if oper is None:
            output = inputs
        elif oper == 'softplus':
            output = torch.nn.functional.softplus(inputs)
        elif oper == 'sigmoid':
            output = torch.nn.functional.sigmoid(inputs)
        elif oper == 'proba':
            output = torch.nn.functional.softmax(inputs, dim=1)
        elif oper == 'labels':
            output = inputs.argmax(dim=1)
        elif callable(oper):
            output = oper(inputs)
            name = oper.__name__
        return attr_prefix + name, output


    def save(self, path, *args, **kwargs):
        """ Save torch model.

        Parameters
        ----------
        path : str
            Path to a file where the model data will be stored.

        Examples
        --------
        .. code-block:: python

            torch_model = ResNet34()

        Now save the model

        .. code-block:: python

            torch_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34.
        """
        _ = args
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save({item: getattr(self, item) for item in self.preserve}, path, pickle_module=dill, **kwargs)

    def load(self, path, *args, eval=False, **kwargs):
        """ Load a torch model from files.

        Parameters
        ----------
        path : str
            File path where a model is stored.

        eval : bool
            Whether to switch the model to eval mode.

        Examples
        --------
        .. code-block:: python

            resnet = ResNet34(load=dict(path='/path/to/models/resnet34'))

            torch_model.load(path='/path/to/models/resnet34')

            TorchModel(config={'device': 'gpu:2', 'load/path': '/path/to/models/resnet34'})

        **How to move the model to device**

        The model will be moved to device specified in the model config by key `device`.
        """
        _ = args
        self._get_devices()

        if self.device:
            checkpoint = torch.load(path, map_location=self.device, pickle_module=dill, **kwargs)
        else:
            checkpoint = torch.load(path, pickle_module=dill, **kwargs)

        for item in self.preserve:
            setattr(self, item, checkpoint.get(item))

        if self.device:
            self.model.to(self.device)

        if eval:
            self.model.eval()
