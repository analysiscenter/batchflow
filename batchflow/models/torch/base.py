""" Eager version of TorchModel. """
import os
import re
import warnings
import threading
import inspect
from collections import OrderedDict
from functools import partial

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .visualization import VisualizationMixin
from .utils import unpack_fn_from_config, get_shape
from .layers import ConvBlock
from .losses import CrossEntropyLoss, BinaryLovaszLoss, LovaszLoss, SSIM, MSSIM
from .losses import binary as binary_losses, multiclass as multiclass_losses
from ..base import BaseModel
from ... import Config



LOSSES = {
    'l1': nn.L1Loss,
    'huber': nn.SmoothL1Loss,
    'absolutedifference': nn.L1Loss,
    'mse': nn.MSELoss,
    'cos': nn.CosineSimilarity,
    'cosine': nn.CosineSimilarity,
    'hinge': nn.HingeEmbeddingLoss,
    'ssim': SSIM,
    'mssim': MSSIM,

    'bce': nn.BCEWithLogitsLoss,
    'bdice': binary_losses.Dice,
    'btversky': binary_losses.Tversky,
    'blovasz': BinaryLovaszLoss,

    'ce': CrossEntropyLoss,
    'crossentropy': CrossEntropyLoss,
    'logloss': CrossEntropyLoss,
    'dice': multiclass_losses.Dice,
    'lovasz': LovaszLoss
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



class TorchModel(BaseModel, VisualizationMixin):
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

    loss : str, dict
        Loss function, might be defined in multiple formats.

        If str, then short ``name``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'mse'``, ``'ce'``, ``'l1'``, ``'cos'``, ``'hinge'``,
              ``'huber'``, ``'logloss'``, ``'dice'``)
            - a class name from `torch losses <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
              (e.g. ``'PoissonNLL'`` or ``'TripletMargin'``)
            - an instance of `:class:torch.nn.Module`
            - callable

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': {'name': 'KLDiv', 'reduction': 'none'}}``
        - ``{'loss': {'name': MyCustomLoss, 'epsilon': 1e-6}}``
        - ``{'loss': my_custom_loss_fn}``

    optimizer : str, dict
        Optimizer, might be defined in multiple formats.

        If str, then short ``name``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'Adam'``, ``'Adagrad'``, any optimizer from
              `torch.optim <https://pytorch.org/docs/stable/optim.html#algorithms>`_)
            - a class with ``Optimizer`` interface
            - a callable which takes model parameters and optional args

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': {'name': 'SparseAdam', 'lr': 0.01}}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}}``
        - ``{'optimizer': {'name': MyCustomOptimizer, 'momentum': 0.95}}``

    decay : dict, list of dicts
        The learning rate decay algorithm might be defined in multiple formats.
        All decays require to have 'frequency' as a key in a configuration dictionary.
        Parameter 'frequency' sets how often do decay step: at every `'frequency'`
        iteration. Each decay might have optional parameters 'first_iter' and 'last_iter'
        that defines the closed range of iterations where decay is at work.
        If you want to use a learning rate warmup and decay together,
        you should use a list of decays (see examples).

        If dict, then ``{'name': name, **kwargs}``.
        If list, then each item is a dict of format described above.

        Name must be one of:

        - a class name from `torch.optim.lr_scheduler
          <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
          (e.g. ``'LambdaLR'``) except ``'ReduceLROnPlateau'``.
        - short name (``'exp'`` - ExponentialLR, ``'lambda'`` - LambdaLR, ``'step'`` - StepLR,
                      ``'multistep'`` - MultiStepLR, ``'cos'`` - CosineAnnealingLR)
        - a class with ``_LRScheduler`` interface
        - a callable which takes optimizer and optional args

        Examples:

        - ``{'decay': {'name: 'exp', 'frequency': 5, 'first_iter': 6, 'last_iter': 20}}``
        - ``{'decay': {'name': 'StepLR', 'steps_size': 10000, 'frequency': 5}}``
        - ``{'decay': {'name': MyCustomDecay, 'decay_rate': .5, 'frequency': 15, 'first_iter': 400}``
        - ``{'decay': [{'name': 'exp', 'gamma': 1, 'frequency': 1, 'last_iter': 900},
                       {'name': 'exp', 'gamma': 0.96, 'frequency': 2, 'first_iter': 901}]``

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

    amp : bool
        Whether to use automated mixed precision during model training and inference. Default is True.
        The output type of predictions remains float32.

    sync_frequency : int
        How often to apply accumulated gradients to the weights. Default value is to apply them after each batch.

    microbatch : int, bool or None
        Also known as virtual batch. If int, then size of chunks to split every batch into.
        Allows to process given data sequentially, accumulating gradients from microbatches and applying them
        once in the end. Can be changed later in the `train` method. Batch size must be divisible by microbatch size.
        If True, then every batch is split into individual items (same as microbatch equals 1).
        If False or None, then feature is not used. Default is not to use microbatching.

    sam_rho : float
        Foret P. et al. "`Sharpness-Aware Minimization for Efficiently Improving Generalization
        <https://arxiv.org/abs/2010.01412>`_".
        If evaluates to False, then SAM is not used.
        If float, then controls the size of neighborhood (check the paper for details).
    sam_individual_norm : bool
        If True, then each gradient is scaled according to its own L2 norm.
        If False, then one common gradient norm is computed and used as a scaler for all gradients.

    callbacks : sequence of `:class:callbacks.BaseCallback`
        Callbacks to call at the end of each training iteration.

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
        :class:`~.torch.layers.ConvBlock` parameters.

        If ``initial_block/inputs`` is specified with a name or list of names,
        then it should contain names from ``inputs`` with info about shapes of tensors to be passed to `initial_block`.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': dict(inputs='features')}``
        - ``{'initial_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``
        - ``{'initial_block': MyCustomModule(some_param=1, another_param=2)}``

    body : dict or nn.Module
        User-defined module or parameters for the base network layers,
        usually :class:`~.torch.layers.ConvBlock` parameters.

    head : dict or nn.Module
        User-defined module or parameters for the head layers,
        usually :class:`~.torch.layers.ConvBlock` parameters.

    predictions : str or callable
        An operation applied to the head output to make the predictions tensor which is used in the loss function.
        See :meth:`.TorchModel.output` for details.

    output : dict or list
        Auxiliary operations to apply to network predictions. See :meth:`.TorchModel.output` for details.

    common : dict
        Default parameters for all blocks (see :class:`~.torch.layers.ConvBlock`).


    **In order to create your own model, it is recommended to:**

    * Take a look at :class:`~.torch.layers.ConvBlock` since it is widely used as a building
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
      Default behaviour is to support all of the :class:`~.torch.layers.ConvBlock` options.
      For complex models, take a look at default config of the chosen model to learn
      which parameters should be configured.
    """
    PRESERVE = [
        'full_config', 'config', 'model',
        'input_names', 'input_shapes', 'target_shape', 'classes',
        'loss', 'optimizer', 'decay', 'decay_step',
        'sync_counter', 'microbatch',
        'iteration', 'iter_info', 'lr_list', 'syncs', 'decay_iters',
        '_loss_list', 'loss_list',
    ]

    def __init__(self, config=None):
        self.full_config = Config(config)
        self.model_lock = threading.Lock()

        # Shapes of inputs and outputs
        self.input_names = None
        self.input_shapes = None
        self.target_shape = None
        self.classes = None

        # Pytorch model
        self.model = None

        # Leading device and list of all devices used
        self.device = None
        self.devices = []

        # Train procedure and ifrastructure
        self.loss = None
        self.optimizer = None
        self.decay = None
        self.decay_step = None

        self.amp = True
        self.scaler = None

        self.callbacks = []

        # Memory amortization: accumulate gradients to update weights later
        self.sync_frequency = 1
        self.sync_counter = 0
        self.microbatch = None

        # Sharpness-aware minimization
        self.sam_rho = 0.0
        self.sam_individual_norm = True

        # Store info about passed train iterations
        self.iteration = 0
        self.iter_info = {}
        self.lr_list = []
        self.syncs = []
        self.decay_iters = []
        self._loss_list = []
        self.loss_list = []

        # Profile kernels used
        self.profile = False
        self.profilers = []
        self.profile_info = None
        super().__init__(config)

    def reset(self):
        """ Allows to recreate model from scratch. """
        self.model = None
        self.iter_info = {}


    def build(self):
        """ Build the model. """
        # Create config from default and external one
        self.full_config = self.combine_configs()
        self._get_devices()
        self._get_placeholder_shapes()
        self.full_config = self.build_config()

        # Store some of the config values
        self.microbatch = self.full_config.get('microbatch', None)
        self.sync_frequency = self.full_config.get('sync_frequency', 1)
        self.amp = self.full_config.get('amp', True)

        self.sam_rho = self.full_config.get('sam_rho', 0.0)
        self.sam_individual_norm = self.full_config.get('sam_individual_norm', False)
        self.profile = self.full_config.get('profile', False)

        self.callbacks = [callback.set_model(self) for callback in self.full_config.get('callbacks', [])]

        # If the inputs are set in config with their shapes we can build right away
        if self.input_shapes:
            self._build()


    # Create config of model creation: combine the external and default ones
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

        config['loss'] = None
        config['optimizer'] = 'Adam'
        config['decay'] = None
        config['amp'] = True

        config['sam_rho'] = 0.0
        config['sam_individual_norm'] = True

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

    def unpack(self, name):
        """ Get params from config. """
        unpacked = unpack_fn_from_config(name, self.full_config)
        if isinstance(unpacked, list):
            return {name: unpacked}
        key, kwargs = unpacked
        return {name: key, **kwargs}


    # Prepare to build the model: determine device(s) and shape(s)
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

    def _to_device(self):
        """ Select whether to put model on a single device or to a number of devices in `DataParallel` mode.

        Notes
        -----
        The method serves for code simplification at build / load stages and shouldn't be applied to prebuilt
        models since it does not change models attributes (like `self.device`) and does not process model-related
        objects (like loss functions or optimizers).
        """
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, self.devices)
        else:
            self.model.to(self.device)

    # Chain multiple building blocks to create model
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
        self._to_device()

        self.make_loss(**self.unpack('loss'))
        self.make_optimizer(**self.unpack('optimizer'))
        self.make_decay(**self.unpack('decay'), optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()


    def _placeholder_data(self):
        data = [np.zeros(shape, dtype=np.float32) for shape in self.input_shapes]
        data = self.transfer_to_device(data)
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


    # Create training procedure(s): loss, optimizer, decay
    def make_loss(self, loss, **kwargs):
        """ Set model loss. Changes the `loss` attribute. """
        loss_fn = None
        # Parse `loss` to actual module
        if isinstance(loss, str):
            # String like 'ce', 'bdice' or 'CrossEntropy'
            if hasattr(nn, loss):
                loss = getattr(nn, loss)
            elif hasattr(nn, loss + "Loss"):
                loss = getattr(nn, loss + "Loss")
            else:
                loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)

        elif isinstance(loss, nn.Module):
            # Already a valid module
            loss_fn = loss
        elif callable(loss):
            # Callable: just pass other arguments in
            loss_fn = partial(loss, **kwargs)
        elif isinstance(loss, type):
            # Class to make module
            pass
        else:
            raise ValueError("Loss is not defined in the model %s" % self.__class__.__name__)

        loss_fn = loss_fn or loss(**kwargs)
        if isinstance(loss_fn, nn.Module):
            loss_fn.to(device=self.device)

        self.loss = loss_fn

    def make_optimizer(self, optimizer, **kwargs):
        """ Set model optimizer. Changes the `optimizer` attribute. """
        # Choose the optimizer
        if callable(optimizer) or isinstance(optimizer, type):
            pass
        elif isinstance(optimizer, str) and hasattr(torch.optim, optimizer):
            optimizer = getattr(torch.optim, optimizer)
        else:
            raise ValueError("Unknown optimizer", optimizer)

        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def make_decay(self, decay, optimizer=None, **kwargs):
        """ Set model decay. Changes the `decay` and `decay_step` attribute. """
        if isinstance(decay, (tuple, list)):
            decays = decay
        else:
            decays = [(decay, kwargs)] if decay else []

        self.decay, self.decay_step = [], []

        for decay_, decay_kwargs in decays:
            if decay_ is None:
                raise ValueError('Missing `name` key in the decay configuration')

            # Parse decay
            if callable(decay_) or isinstance(decay_, type):
                pass
            elif isinstance(decay_, str) and hasattr(torch.optim.lr_scheduler, decay_):
                decay = getattr(torch.optim.lr_scheduler, decay_)
            elif decay_ in DECAYS:
                decay_ = DECAYS.get(decay_)
            else:
                raise ValueError('Unknown learning rate decay method', decay_)

            # Parse step parameters
            step_params = {
                'first_iter': 0,
                'last_iter': np.inf,
                **decay_kwargs
            }
            if 'frequency' not in step_params:
                raise ValueError('Missing `frequency` key in the decay configuration')

            # Set defaults for some of the decays
            if decay_ in DECAYS_DEFAULTS:
                decay_dict = DECAYS_DEFAULTS.get(decay_).copy()
                if decay == DECAYS['cos']:
                    decay_dict.update(T_max=step_params['frequency'])
                decay_kwargs = {**decay_dict, **decay_kwargs}

            # Remove unnecessary keys from kwargs
            for key in ['first_iter', 'last_iter', 'frequency']:
                decay_kwargs.pop(key, None)

            # Create decay or store parameters for later usage
            if optimizer:
                decay_ = decay_(optimizer, **decay_kwargs)
            else:
                decay = (decay_, decay_kwargs)

            self.decay.append(decay_)
            self.decay_step.append(step_params)


    # Use an external model
    def set_model(self, model):
        """ Set the underlying model to a supplied one and update training infrastructure. """
        self.model = model

        self._to_device()

        self.make_loss(**self.unpack('loss'))
        self.make_optimizer(**self.unpack('optimizer'))
        self.make_decay(**self.unpack('decay'), optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

    # Define model structure
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


    # Transfer data to/from device(s)
    def parse_inputs(self, *args, **kwargs):
        """ Convert arguments (either positional or keyword) into inputs and targets of a neural network. """
        if args and kwargs:
            raise ValueError('Use either positional or keyword arguments in `train` call.')

        if kwargs:
            for name in ['labels', 'masks', 'targets']:
                if name in kwargs:
                    targets = kwargs.pop(name)

            args = [kwargs.get(name) for name in (self.input_names or list(kwargs.keys()))]
            args.append(targets)
        return args

    def transfer_to_device(self, data):
        """ Transfer (possibly nested) structure to device and return the same structure. """
        if isinstance(data, (tuple, list)):
            return [self.transfer_to_device(item) for item in data]

        if isinstance(data, np.ndarray):
            if data.dtype not in [np.float32, 'float32']:
                data = data.astype(np.float32)
            data = torch.from_numpy(data).to(self.device)
            return data

        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
            return data

        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            if data.device.id == self.device.index:
                data = torch.utils.dlpack.from_dlpack(data.toDlpack())
                return data
            raise TypeError(f'cupy arrays should reside on the same GPU, as model itself: {self.device}.')

        if data is None:
            return None
        raise TypeError('Passed data should either be a `np.ndarray`, `torch.Tensor` or `cupy.ndarray`. ')

    def transfer_from_device(self, data):
        """ Transfer (possibly nested) structure from device and return the same structure. """
        if isinstance(data, (tuple, list)):
            return [self.transfer_from_device(item) for item in data]

        if isinstance(data, (torch.Tensor, torch.autograd.Variable)):
            return data.detach().cpu().numpy()

        if isinstance(data, (np.ndarray, int, float)):
            return data
        raise TypeError('Passed data should either be a `torch.Tensor` or sequence of them. ')

    def parse_output(self, fetches, outputs):
        """ Retrieve tensors from device in the same structure, as `fetches`. """
        fetches = fetches if fetches is not None else []
        _fetches = [fetches] if isinstance(fetches, str) else fetches

        output = []
        for name in _fetches:
            if name in outputs:
                value = outputs[name]
                value = self.transfer_from_device(value)
                output.append(value)
            else:
                raise KeyError('Unknown value to fetch', name)

        output = output[0] if isinstance(fetches, str) else type(fetches)(output)
        return output


    # Apply model to train/predict on given data
    def train(self, *args, feed_dict=None, fetches=None, use_lock=True, profile=False,
              sync_frequency=True, microbatch=True, sam_rho=None, sam_individual_norm=None, **kwargs):
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
            If True, then model, loss and gradient update operations are locked, thus allowing for multithreading.
        sync_frequency : int, bool or None
            If int, then how often to apply accumulated gradients to the weights.
            If True, then value from config is used (default value is to apply gradients after each batch of data).
            If False or None, then gradients are applied after each batch of data.
        microbatch : int, bool or None
            If int, then size of chunks to split every batch into. Allows to process given data sequentially,
            accumulating gradients from microbatches and applying them once in the end.
            If True, then value from config is used (default value is not to use microbatching).
            If False or None, then microbatching is not used.
        sam_rho : float
            Foret P. et al. "`Sharpness-Aware Minimization for Efficiently Improving Generalization
            <https://arxiv.org/abs/2010.01412>`_".
            If evaluates to False, then SAM is not used.
            If float, then controls the size of neighborhood (check the paper for details).
        sam_individual_norm : bool
            If True, then each gradient is scaled according to its own L2 norm.
            If False, then one common gradient norm is computed and used as a scaler for all gradients.
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
        # Prepare inputs and targets: convert to Torch Tensors and transfer to device
        *inputs, targets = self.parse_inputs(*args, **{**(feed_dict or {}), **kwargs})

        # Lock the entire method; release in any case
        try:
            if use_lock:
                self.model_lock.acquire()

            # Parse arguments
            if sync_frequency is True:
                sync_frequency = self.sync_frequency
            elif sync_frequency is False or sync_frequency is None:
                sync_frequency = 1

            if microbatch:
                if microbatch is True:
                    microbatch = self.microbatch
                else:
                    microbatch = microbatch or self.microbatch

            # Split data into microbatches, if needed
            if microbatch:
                microbatch = 1 if microbatch is True else microbatch
                steps = len(targets) // microbatch
                split_inputs = [[item[i:i + microbatch] for item in inputs] for i in range(0, len(targets), microbatch)]
                split_targets = [targets[i:i + microbatch] for i in range(0, len(targets), microbatch)]
            else:
                steps = 1
                split_inputs = [inputs]
                split_targets = [targets]

            # Prepare parameters for SAM
            if sam_rho is None:
                sam_rho = self.sam_rho
            if sam_individual_norm is None:
                sam_individual_norm = self.sam_individual_norm

            # Create Pytorch model if it is yet to be initialized, based on the actual inputs
            if self.model is None:
                if isinstance(split_inputs[0], (list, tuple)):
                    self.input_shapes = [get_shape(item) for item in split_inputs[0]]
                else:
                    self.input_shapes = get_shape(split_inputs[0])

                self.target_shape = get_shape(split_targets[0])
                if self.classes is None:
                    if len(self.target_shape) > 1: # segmentation
                        self.classes = self.target_shape[1]

                # Can use the first two items to build model: no need for the whole tensor
                self.build_config()
                build_inputs = [item[:2] for item in split_inputs[0]]
                build_inputs = self.transfer_to_device(build_inputs)
                self._build(build_inputs)

            self.model.train()

            # Set up the profiling, if needed
            profile = profile or self.profile
            if profile:
                profiler = torch.autograd.profiler.profile(use_cuda='cpu' not in self.device.type)
                profiler.__enter__()

            # Train on each of the microbatches
            outputs = []
            for i in range(steps):
                _inputs = split_inputs[i]
                _targets = split_targets[i]

                _inputs = self.transfer_to_device(_inputs)
                _targets = self.transfer_to_device(_targets)

                output = self._train(*_inputs, _targets, fetches=fetches, sync_frequency=sync_frequency*steps,
                                     sam_rho=sam_rho, sam_individual_norm=sam_individual_norm)
                outputs.append(output)

            # Store the average value of loss over the entire batch
            self.loss_list.append(np.mean(self._loss_list[-steps:]))

            # Parse `outputs` to a desired structure. `outputs` stores fetches for each microbatch
            # which must be aggregated to get fetches for the whole batch. Scalar values will be
            # aggregated by `mean`, array values will be concatenated by the first (batch) axis.
            if fetches:
                outputs = [[item] for item in outputs] if isinstance(fetches, str) else outputs
                output = []
                for i in range(len(outputs[0])):
                    fetches_values = [item[i] for item in outputs]
                    if fetches_values[0].size != 1:
                        output.append(np.concatenate(fetches_values, axis=0))
                    else:
                        output.append(np.mean(fetches_values))
                if isinstance(fetches, str):
                    output = output[0]
            else:
                output = []

            # Exit the profiling mode
            if profile:
                profiler.__exit__(None, None, None)
                self.profilers.append(profiler)

            # Store info about current iteration
            self.iter_info.update({
                'amp': self.amp,
                'microbatch': microbatch,
                'sync_frequency': sync_frequency,
                'steps': steps,
                'sam': bool(sam_rho), 'sam_rho': sam_rho, 'sam_individual_norm': sam_individual_norm,
                'actual_model_inputs_shape': [get_shape(item) for item in _inputs],
                'actual_model_outputs_shape': get_shape(_targets),
            })

            # Call the callbacks
            for callback in self.callbacks:
                callback.on_iter_end()

        finally:
            if use_lock:
                self.model_lock.release()
        return output

    def _train(self, *args, fetches=None, sync_frequency=True, sam_rho=0.0, sam_individual_norm=True):
        # Parse inputs
        *inputs, targets = args
        inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs

        # Apply model, compute loss and gradients
        with torch.cuda.amp.autocast(enabled=self.amp):
            predictions = self.model(inputs)

        # SAM: store grads from previous microbatches
        if self.iteration >= 1 and bool(sam_rho):
            for p in self.model.parameters():
                if p.grad is not None:
                    self.optimizer.state[p]['previous_grad'] = p.grad.clone().detach()
                    p.grad = None

        with torch.cuda.amp.autocast(enabled=self.amp):
            loss = self.loss(predictions, targets)
            loss_ = loss if sync_frequency == 1 else loss / sync_frequency
        (self.scaler.scale(loss_) if self.amp else loss).backward()

        # SAM: use obtained grads to move to the local maxima
        if self.iteration >= 1 and bool(sam_rho):
            # Fetch gradients
            grads = []
            params_with_grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
                    p.grad = None

            # Move to the local maxima
            if sam_individual_norm:
                epsilons = [grad * sam_rho / (grad.detach().norm(2).to(self.device)) for grad in grads]
            else:
                grad_norm = torch.stack([g.detach().norm(2).to(self.device) for g in grads]).norm(2)
                epsilons = [eps * sam_rho / grad_norm for eps in grads]

            if self.amp:
                scale = self.scaler.get_scale()
                epsilons = [eps / scale for eps in epsilons]
            params_with_grads = [p + eps for p, eps in zip(params_with_grads, epsilons)]

            # Compute new gradients: direction to move to minimize the local maxima
            with torch.cuda.amp.autocast(enabled=self.amp):
                predictions = self.model(inputs)
                loss_inner = self.loss(predictions, targets)
            (self.scaler.scale(loss_inner) if self.amp else loss_inner).backward()

            # Cancel the previous update to model parameters, add stored gradients from previous microbatches
            params_with_grads = [p - eps for p, eps in zip(params_with_grads, epsilons)]

            for p in self.model.parameters():
                previous_grad = self.optimizer.state[p].get('previous_grad')
                if previous_grad is not None:
                    p.grad.add_(previous_grad)

        # Store loss value for every microbatch
        self._loss_list.append(loss.detach().cpu().numpy())

        # Whether to update weights or keep accumulating
        if self.sync_counter == sync_frequency - 1:
            # Store learning rate: once per sync
            # Note: we do it before decay, so it is actual LR used on this iteration
            self.lr_list.append([group['lr'] for group in self.optimizer.param_groups])

            # Update weights and remove grads
            if self.amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Optimization over default `zero_grad`; can be removed after PyTorch >= 1.8
            for p in self.model.parameters():
                p.grad = None
            self.iteration += 1

            # Apply decay to learning rate, if needed
            if self.decay:
                for decay, decay_step in zip(self.decay, self.decay_step):
                    step_cond = (self.iteration - decay_step['first_iter']) % decay_step['frequency'] == 0
                    range_cond = decay_step['first_iter'] <= self.iteration <= decay_step['last_iter']
                    if step_cond and range_cond:
                        decay.step()
                        self.decay_iters.append(self.iteration)

            # Update counters
            self.sync_counter = 0
            self.syncs.append(True)
        else:
            self.sync_counter += 1
            self.syncs.append(False)

        # Store outputs
        output_container = {
            'predictions': predictions,
            'loss': loss,
        }

        config = self.full_config
        additional_outputs = self.output(inputs=predictions,
                                         predictions=config['predictions'],
                                         ops=config['output'])
        output_container = {**output_container, **additional_outputs}
        output = self.parse_output(fetches, output_container)
        return output


    def predict(self, *args, targets=None, feed_dict=None, fetches=None, use_lock=True, **kwargs):
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
        use_lock : bool
            If True, then model and loss computation operations are locked, thus allowing for multithreading.
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
        inputs, targets = self._make_prediction_inputs(*args, targets=targets, feed_dict=feed_dict, **kwargs)

        # Acquire lock, release anyway
        try:
            if use_lock:
                self.model_lock.acquire()

            self.model.eval()

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp):
                output_container = {}
                inputs = self.transfer_to_device(inputs)
                predictions = self.model(inputs)

                if self.amp:
                    if isinstance(predictions, (tuple, list)):
                        predictions = type(predictions)(p.float() for p in predictions)
                    else:
                        predictions = predictions.float()
                output_container['predictions'] = predictions

                if targets is not None:
                    targets = self.transfer_to_device(targets)
                    output_container['loss'] = self.loss(predictions, targets)

            config = self.full_config
            additional_outputs = self.output(inputs=predictions, predictions=config['predictions'],
                                             ops=config['output'])
            output_container = {**output_container, **additional_outputs}
            output = self.parse_output(fetches, output_container)

        finally:
            if use_lock:
                self.model_lock.release()
        return output

    def _make_prediction_inputs(self, *args, targets=None, feed_dict=None, **kwargs):
        """ Parse arguments to create valid inputs for the model.
        Implements the logic of parsing the positional and keyword arguments to the model,
        possibly wrapped into `feed_dict` dictionary, or even combination of the two.

        Used under the hood of :meth:`~.TorchModel.predict` method.

        Examples
        --------
        .. code-block:: python

            model.predict(B('images'), targets=B('labels'))
            model.predict(images=B('images'), targets=B('labels'))
            model.predict(B('images'), targets=B('labels'), masks=B('masks'))
        """
        # Concatenate `kwargs` and `feed_dict`; if not empty, use keywords in `parse_input`
        feed_dict = {**(feed_dict or {}), **kwargs}
        if len(feed_dict) == 1:
            _, value = feed_dict.popitem()
            args = (*args, value)
        if feed_dict:
            if targets is not None and 'targets' in feed_dict.keys():
                warnings.warn("`targets` already present in `feed_dict`, so those passed as keyword arg won't be used")
            *inputs, targets = self.parse_inputs(*args, **feed_dict)

        # Positional arguments only
        else:
            inputs = self.parse_inputs(*args)
            if targets is not None:
                targets = self.parse_inputs(targets)[0]
        inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs
        return inputs, targets

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


    # Preserve model for later usage
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

        if kwargs.get('pickle_module') is None:
            kwargs['pickle_module'] = dill

        torch.save({item: getattr(self, item) for item in self.PRESERVE}, path, **kwargs)

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

        if kwargs.get('pickle_module') is None:
            kwargs['pickle_module'] = dill

        if self.device:
            checkpoint = torch.load(path, map_location=self.device, **kwargs)
        else:
            checkpoint = torch.load(path, **kwargs)

        # `load_config` is a reference to `self.config` used to update `full_config`
        # It is required since `self.config` is overwritten in the cycle below
        load_config = self.config

        for item in self.PRESERVE:
            setattr(self, item, checkpoint.get(item))
        self.full_config = self.full_config + load_config

        self._to_device()

        if eval:
            self.model.eval()


    # Debug and profile the performance
    def set_debug_mode(self, mode=True):
        """ Changes representation of model to a more or less detailed.
        By default, model representation reduces the description of the most complex modules.
        """
        if self.model is None:
            raise ValueError('Model is not initialized yet. ')
        self.model.apply(lambda module: setattr(module, 'debug', mode))

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
