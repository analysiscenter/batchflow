""" Eager version of TorchModel. """
import os
import re
import threading
import inspect
from collections import OrderedDict
from functools import partial

import dill
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .initialization import best_practice_resnet_init
from .visualization import VisualizationMixin
from .utils import unpack_fn_from_config, get_shape
from .layers import ConvBlock
from .losses import CrossEntropyLoss, BinaryLovaszLoss, LovaszLoss, SSIM, MSSIM
from .losses import binary as binary_losses, multiclass as multiclass_losses
from ..base import BaseModel
from ...config import Config



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

    init_weights : callable, 'best_practice_resnet', or None
        Model weights initilaization.
        If None, then default initialization is used.
        If 'best_practice_resnet', then common used non-default initialization is used.
        If callable, then callable applied to each layer.

        Examples:

        - ``{'init_weights': 'best_practice_resnet'}``
        - .. code-block:: python

            def callable_init(module): # example of a callable for init
                if isinstance(module, nn.Linear):
                    nn.kaiming_normal_(module.weight)

            config = {'init_weights': callable_init}

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
        - .. code-block:: python

            {'decay': [{'name': 'exp', 'gamma': 1, 'frequency': 1, 'last_iter': 900},
                       {'name': 'exp', 'gamma': 0.96, 'frequency': 2, 'first_iter': 901}]

    device : str, torch.device or sequence
        If str, a device name (e.g. ``'cpu'`` or ``'gpu:0'``). Regular expressions are also allowed (e.g. ``'gpu:*'``).
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
        'inputs_shapes', 'targets_shapes', 'classes',
        'loss', 'optimizer', 'decay', 'decay_step',
        'sync_counter', 'microbatch',
        'iteration', 'iter_info', 'lr_list', 'syncs', 'decay_iters',
        '_loss_list', 'loss_list',
    ]

    LABELS_ALIASES = ['labels', 'masks', 'targets']

    def __init__(self, config=None):
        self.full_config = Config(config)
        self.model_lock = threading.Lock()

        # Shapes of inputs and targets
        self.inputs_shapes = None
        self.targets_shapes = None
        self.classes = None

        # Pytorch model
        self.model = None

        # Leading device and list of all devices to use
        self.device = None
        self.devices = []

        # Train procedure and ifrastructure
        self.init_weights = None
        self.loss = None
        self.optimizer = None
        self.decay = None
        self.decay_step = None

        self.amp = True
        self.scaler = None

        self.operations = {}
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
        self.last_iteration_info = {}
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
        """ Delete the underlying model and all the infrastructure. Use to create model from scratch. """
        # TODO: do we really need this?
        self.model = None
        self.last_iteration_info = {}


    def build(self):
        """ Initialize the instance: make the config, attributes, and, if possible, PyTorch model. """
        # Create config from default and external one
        self.full_config = self.combine_configs()
        self.parse_devices()
        self.parse_placeholder_shapes()

        self.update_config()
        self.update_attributes()

        # If the inputs are set in config with their shapes we can build right away
        if self.inputs_shapes:
            self.build_model()


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

        config['output'] = None
        return config

    def combine_configs(self):
        """ Combine default configuration and the external one. """
        config = self.default_config() + self.config
        return config

    def update_config(self):
        """ Update config with instance attributes. """
        config = self.full_config

        config['head/targets_shapes'] = self.targets_shapes
        # As `update_config` can be called multiple times, and `head/classes` key can have value `None`,
        # we need to use `or` insetad of `get`
        config['head/classes'] = config.get('head/classes') or self.classes

        if config.get('head/units') is None:
            config['head/units'] = config.get('head/classes')
        if config.get('head/filters') is None:
            config['head/filters'] = config.get('head/classes')

        print('CCC', self.classes)

    def update_attributes(self):
        """ Update instance attributes from config. """
        config = self.full_config

        self.init_weights = config.get('init_weights', None)
        self.microbatch = config.get('microbatch', None)
        self.sync_frequency = config.get('sync_frequency', 1)
        self.amp = config.get('amp', True)

        self.sam_rho = config.get('sam_rho', 0.0)
        self.sam_individual_norm = config.get('sam_individual_norm', False)
        self.profile = config.get('profile', False)

        self.callbacks = [callback.set_model(self) for callback in config.get('callbacks', [])]

        operations = config['output']
        if not isinstance(operations, dict):
            operations = operations or []
            operations = list(operations) if isinstance(operations, (tuple, list)) else [operations]
            operations = {'' : operations}
        self.operations = operations


    # Prepare to build the PyTorch model: determine device(s) and shape(s)
    def parse_devices(self):
        """ !!. """
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

    def parse_placeholder_shapes(self):
        """ Update attributes with shapes from config. """
        config = self.full_config

        batch_size = config.get('placeholder_batch_size', 2)
        inputs_shapes = config.get('inputs_shapes') or config.get('input_shapes')
        targets_shapes = config.get('targets_shapes') or config.get('target_shapes')
        classes = config.get('classes')

        if inputs_shapes:
            inputs_shapes = self._to_nested_list(inputs_shapes)
            self.inputs_shapes = [(batch_size, *shape) for shape in inputs_shapes]

        if targets_shapes:
            targets_shapes = self._to_nested_list(targets_shapes)
            self.targets_shapes = [(batch_size, *shape) for shape in targets_shapes]

            if not classes:
                self.classes = [item[0] for item in targets_shapes]

        if classes:
            classes = list(classes) if isinstance(classes, (tuple, list)) else [classes]
            self.classes = classes

    @staticmethod
    def _to_nested_list(sequence):
        if not isinstance(sequence[0], (tuple, list)):
            return [list(sequence)]
        return [list(item) for item in sequence]


    def make_placeholder_data(self):
        """ !!. """
        data = [np.zeros(shape, dtype=np.float32) for shape in self.inputs_shapes]
        data = self.transfer_to_device(data)
        return data


    # Create training infrastructure: loss, optimizer, decay
    def unpack(self, name):
        """ Get params from config. """
        unpacked = unpack_fn_from_config(name, self.full_config)
        if isinstance(unpacked, list):
            return {name: unpacked}
        key, kwargs = unpacked
        return {name: key, **kwargs}


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
        self.initialize_weights()


        self.model_to_device()

        self.make_loss(**self.unpack('loss'))
        self.make_optimizer(**self.unpack('optimizer'))
        self.make_decay(**self.unpack('decay'), optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

    # Chain multiple building blocks to create model
    def build_model(self, inputs=None):
        """ !!. """
        inputs = inputs or self.make_placeholder_data()
        inputs = inputs[0] if len(inputs) == 1 else inputs

        blocks = OrderedDict()
        for item in self.full_config.get('order'):
            # Get the `block_name`, which is used as the name in the Sequential,
            #         `config_name`, which is used to retrieve parameters from config,
            #     and `method`, which is either a callable or name of the method to get from the current instance
            if isinstance(item, str):
                block_name = config_name = method = item
            elif isinstance(item, tuple) and len(item) == 3:
                block_name, config_name, method = item
            elif isinstance(item, dict):
                block_name = item['block_name']
                config_name = item.get('config_name', block_name)
                method = item.get('method', config_name)

            # Make block, from the `inputs`, transfer it to device
            # Important: apply to the `inputs` before showing to the next block, so the shapes/etc are updated
            block = self.make_block(config_name, method, inputs)

            if block is not None:
                block.to(self.device)
                inputs = block(inputs)
                blocks[block_name] = block

        # Use the OrderedDict in Sequential to give readable names to stages
        self.model = nn.Sequential(blocks)
        self.initialize_weights()
        self.model_to_device()

        # Make the infrastructure after the model is created, so that they are on the same device
        self.make_loss(**self.unpack('loss'))
        self.make_optimizer(**self.unpack('optimizer'))
        self.make_decay(**self.unpack('decay'), optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

    def make_block(self, name, method, inputs):
        """ !!. """
        config = self.full_config
        block = config[name]

        if isinstance(block, nn.Module):
            # Already initialized module
            pass

        elif isinstance(block, dict):
            block_params = {**config['common'], **block}

            if 'module' in block_params:
                # A custom module
                module = block_params['module']

                if isinstance(module, nn.Module):
                    # Already initialized module
                    block = module
                else:
                    # Initialize module with parameters from config. Add `inputs`, if needed
                    kwargs = block_params.get('module_kwargs', {})
                    if 'inputs' in inspect.getfullargspec(module.__init__)[0]:
                        kwargs['inputs'] = inputs
                    block = module(**kwargs)
            else:
                # A string to get the module from the instance or callable that returns nn.Module
                method = getattr(self, method) if isinstance(method, str) else method
                block = method(inputs=inputs, **block_params)
        else:
            raise ValueError(f'`{name}` must be configured either as nn.Module or dictionary, got {block_params}')
        return block


    # Pre-defined building blocks
    @classmethod
    def get_block_defaults(cls, name, kwargs):
        """ Make block parameters from class default config and kwargs. """
        class_config = cls.default_config()
        return class_config['common'] + Config(class_config.get(name)) + (kwargs or {})

    @classmethod
    def block(cls, inputs, name, **kwargs):
        """ Model building block: either a :class:`~.torch.layers.ConvBlock` or a `base_block`. """
        kwargs = cls.get_block_defaults(name, kwargs)
        if kwargs.get('layout') or kwargs.get('base_block'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

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
        return cls.block(inputs, name='initial_block', **kwargs)

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
        return cls.block(inputs, name='body', **kwargs)

    @classmethod
    def head(cls, inputs, **kwargs):
        """ The last network layers which produce predictions. Usually used to make network output
        compatible with the `targets` tensor.

        Notes
        -----
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        return cls.block(inputs, name='head', **kwargs)


    # Model weights initialization
    def initialize_weights(self):
        """ Initialize model weights with a callable or use default."""
        if self.model and (self.init_weights is not None):
            # Parse model weights initilaization
            if isinstance(self.init_weights, str):
                # We have only one variant of predefined init function, so we check that init is str for a typo case
                # The common used non-default weights initialization:
                self.init_weights = best_practice_resnet_init

            # Weights and biases initialization
            self.model.apply(self.init_weights)


    # Transfer to/from device(s)
    def transfer_to_device(self, data):
        """ Transfer (possibly nested) structure to device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return {key : self.transfer_to_device(value) for key, value in data.items()}

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
        raise TypeError('Passed data should either be a `np.ndarray`, `torch.Tensor` or `cupy.ndarray`.')

    def transfer_from_device(self, data):
        """ Transfer (possibly nested) structure from device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return {key : self.transfer_from_device(value) for key, value in data.items()}

        if isinstance(data, (tuple, list)):
            return [self.transfer_from_device(item) for item in data]

        if isinstance(data, (torch.Tensor, torch.autograd.Variable)):
            cpu_tensor = data.detach().cpu().numpy()
            if self.amp and cpu_tensor.dtype != np.float32:
                cpu_tensor = cpu_tensor.astype(np.float32)
            return cpu_tensor

        if isinstance(data, (np.ndarray, int, float)):
            return data
        raise TypeError('Passed data should either be a `torch.Tensor` or a container of them. ')

    def model_to_device(self):
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


    # Apply model to train/predict on given data
    def train(self, inputs, targets, outputs=None, lock=True, profile=False,
              sync_frequency=True, microbatch=True, microbatch_drop_last=True,
              sam_rho=None, sam_individual_norm=None):
        """ Train the model with the data provided

        Parameters
        ----------
        !!
        lock : bool
            If True, then model, loss and gradient update operations are locked, thus allowing for multithreading.
        sync_frequency : int, bool or None
            If int, then how often to apply accumulated gradients to the weights.
            If True, then value from config is used. Default value is 1,
            which means to apply gradients after each batch of data.
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
        # Lock the entire method; release in any case
        try:
            if lock:
                self.model_lock.acquire() #pylint: disable=consider-using-with
            self.last_iteration_info = {}

            # Parse inputs and targets: always a list
            inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
            targets = list(targets) if isinstance(targets, (tuple, list)) else [targets]

            # Parse outputs: always a list
            single_output = isinstance(outputs, str)
            outputs = [outputs] if single_output else (outputs or [])

            # Parse train parameters
            if sync_frequency is True:
                sync_frequency = self.sync_frequency
            elif sync_frequency is False or sync_frequency is None:
                sync_frequency = 1

            if microbatch:
                if microbatch is True:
                    microbatch = self.microbatch
                else:
                    microbatch = microbatch or self.microbatch

            # Prepare parameters for SAM
            if sam_rho is None:
                sam_rho = self.sam_rho
            if sam_individual_norm is None:
                sam_individual_norm = self.sam_individual_norm

            # Compute batch_size and make sure it is the same for all inputs and targets
            batch_size = len(inputs[0])
            for i, item in enumerate(inputs):
                if len(item) != batch_size:
                    raise ValueError('All of `inputs` should have the same length, as the first one!'
                                     f'Input at position `{i}` has size {len(item)}!={batch_size}')
            for i, item in enumerate(targets):
                if len(item) != batch_size:
                    raise ValueError('All of `targets` should have the same length, as the first of `inputs`!'
                                     f'Target at position `{i}` has size {len(item)}!={batch_size}')

            # Split data into microbatches, if needed
            if microbatch:
                chunked_inputs = [[item[i:i + microbatch] for item in inputs]
                                  for i in range(0, batch_size, microbatch)]
                chunked_targets = [[item[i:i + microbatch] for item in targets]
                                   for i in range(0, batch_size, microbatch)]

                if microbatch_drop_last and batch_size % microbatch != 0:
                    chunked_inputs = chunked_inputs[:-1]
                    chunked_targets = chunked_targets[:-1]
            else:
                chunked_inputs = [inputs]
                chunked_targets = [targets]

            steps = len(chunked_inputs)
            inputs_shapes = [get_shape(item) for item in chunked_inputs[-1]]
            targets_shapes = [get_shape(item) for item in chunked_targets[-1]]
            self.last_iteration_info.update({'inputs_shapes': inputs_shapes,
                                             'targets_shapes': targets_shapes})

            # Create PyTorch model if it is yet to be initialized, based on the actual inputs
            if self.model is None:
                # Update config with shapes
                self.inputs_shapes = inputs_shapes
                self.targets_shapes = targets_shapes
                if not self.classes:
                    self.classes = [shape[1] for shape in targets_shapes]

                self.update_config()

                # Can use the first two items to build model: no need for the whole tensor
                build_inputs = [item[:2] for item in chunked_inputs[0]]
                build_inputs = self.transfer_to_device(build_inputs)
                self.build_model(build_inputs)

            self.model.train()

            # Set up the profiling, if needed
            profile = profile or self.profile
            if profile:
                profiler = torch.autograd.profiler.profile(use_cuda='cpu' not in self.device.type)
                profiler.__enter__()

            # Train on each of the microbatches
            chunked_outputs = []
            for chunk_inputs, chunk_targets in zip(chunked_inputs, chunked_targets):
                chunk_inputs = self.transfer_to_device(chunk_inputs)
                chunk_targets = self.transfer_to_device(chunk_targets)

                # Return is a dictionary with desired outputs and some meta information
                chunk_outputs = self._train(inputs=chunk_inputs, targets=chunk_targets, outputs=outputs,
                                            sync_frequency=sync_frequency*steps,
                                            sam_rho=sam_rho, sam_individual_norm=sam_individual_norm)
                chunked_outputs.append(chunk_outputs)

            # Exit the profiling
            if profile:
                profiler.__exit__(None, None, None)
                self.profilers.append(profiler)

            # Store the average value of loss over microbatches
            self.loss_list.append(np.mean(self._loss_list[-steps:]))

            # Store info about current iteration
            self.last_iteration_info.update({
                'amp': self.amp,
                'microbatch': microbatch,
                'sync_frequency': sync_frequency,
                'steps': steps,
                'sam': bool(sam_rho), 'sam_rho': sam_rho,
                'sam_individual_norm': sam_individual_norm,
                'outputs': outputs,
            })

            # Call the callbacks
            for callback in self.callbacks:
                callback.on_iter_end()

            # Parse `chunked_outputs` to a desired structure. `chunked_outputs` stores `outputs` for each microbatch
            # which must be aggregated to get `outputs` for the whole batch.
            # Scalar values are aggregated by `mean`, array values are concatenated along the first (batch) axis.
            result = []

            for output_name in outputs:
                # All tensors for current `output_name`
                chunked_output = [chunk_outputs[output_name] for chunk_outputs in chunked_outputs]

                if chunked_output[0].size != 1:
                    result.append(np.concatenate(chunked_output, axis=0))
                else:
                    result.append(np.mean(chunked_output))
            if single_output:
                result = result[0]

        finally:
            if lock:
                self.model_lock.release()
        return result

    def _train(self, inputs, targets, outputs, sync_frequency, sam_rho, sam_individual_norm):
        # Parse inputs
        inputs = inputs[0] if len(inputs) == 1 else inputs
        targets = targets[0] if len(targets) == 1 else targets

        # Compute predictions; store shapes for introspection
        with torch.cuda.amp.autocast(enabled=self.amp):
            predictions = self.model(inputs)
        predictions_ = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        self.last_iteration_info['predictions_shapes'] = [get_shape(item) for item in predictions_]

        # SAM: store grads from previous microbatches
        if self.iteration >= 1 and bool(sam_rho):
            self._train_sam_store_gradients()

        # Compute loss and gradients; store loss value for every microbatch
        with torch.cuda.amp.autocast(enabled=self.amp):
            loss = self.loss(predictions, targets)
            loss_ = loss / sync_frequency
        (self.scaler.scale(loss_) if self.amp else loss_).backward()
        self._loss_list.append(self.transfer_from_device(loss))

        # SAM: use obtained grads to move to the local maxima
        if self.iteration >= 1 and bool(sam_rho):
            self._train_sam_update_gradients(inputs=inputs, targets=targets, sync_frequency=sync_frequency,
                                             sam_rho=sam_rho, sam_individual_norm=sam_individual_norm)

        # Whether to update weights or keep accumulating
        if self.sync_counter == sync_frequency - 1:
            # Store learning rate: we do it before decay, so it is actual LR used on this iteration
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

        # Make all possible outputs
        additional_outputs = self.make_outputs(predictions=predictions)
        output_container = {
            **additional_outputs,
            'predictions': predictions,
            'loss': loss,
        }

        # Transfer only the requested outputs to CPU
        requested_outputs = {key : output_container[key] for key in outputs}
        return self.transfer_from_device(requested_outputs)

    def _train_sam_store_gradients(self):
        """ Store gradients from previous microbatches. """
        for p in self.model.parameters():
            if p.grad is not None:
                self.optimizer.state[p]['previous_grad'] = p.grad.clone().detach()
                p.grad = None

    def _train_sam_update_gradients(self, inputs, targets, sync_frequency, sam_rho, sam_individual_norm):
        """ Update gradients to move to the local maxima. """
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
            predictions_inner = self.model(inputs)
            loss_inner = self.loss(predictions_inner, targets) / sync_frequency
        (self.scaler.scale(loss_inner) if self.amp else loss_inner).backward()

        # Cancel the previous update to model parameters, add stored gradients from previous microbatches
        params_with_grads = [p - eps for p, eps in zip(params_with_grads, epsilons)]

        for p in self.model.parameters():
            previous_grad = self.optimizer.state[p].get('previous_grad')
            if previous_grad is not None:
                p.grad.add_(previous_grad)


    def predict(self, inputs, targets=None, outputs=None, lock=True):
        """ Get predictions on the data provided.

        Parameters
        ----------
        !!
        targets : ndarray, optional
            Targets to calculate loss.
        fetches : tuple, list
            Sequence of tensors to fetch from the model.
        lock : bool
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

        # Acquire lock; release in any case
        try:
            if lock:
                self.model_lock.acquire() #pylint: disable=consider-using-with

            # Parse outputs: always a list
            single_output = isinstance(outputs, str)
            outputs = [outputs] if single_output else (outputs or [])

            self.model.eval()
            output_container = {}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp):
                inputs = self.transfer_to_device(inputs)
                predictions = self.model(inputs)
                output_container['predictions'] = predictions

                if targets is not None:
                    targets = self.transfer_to_device(targets)
                    loss = self.loss(predictions, targets)
                    output_container['loss'] = loss

            # Make all possible outputs
            additional_outputs = self.make_outputs(predictions=predictions)
            output_container.update(additional_outputs)

            # Transfer only the requested outputs to CPU
            requested_outputs = [output_container[key] for key in outputs]
            result = self.transfer_from_device(requested_outputs)
            if single_output:
                result = result[0]

        finally:
            if lock:
                self.model_lock.release()
        return result



    def make_outputs(self, predictions):
        """ !!. """
        predictions = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]

        outputs = {}
        # Iterate over tensors in predictions and the corresponding output operations
        for i, (tensor, (output_prefix, output_operations)) in enumerate(zip(predictions, self.operations.items())):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'Network outputs are expected to be tensors, got {type(tensor)} instead!')

            output_prefix = output_prefix + '_' if output_prefix else ''

            # For each operation, add multiple aliases
            for j, operation in enumerate(output_operations):
                output_tensor, operation_name = self.apply_output_operation(tensor, operation)
                if operation_name:
                    outputs[output_prefix + operation_name] = output_tensor # i.e. `first_sigmoid`, `sigmoid`

                outputs.update({
                    output_prefix + str(j) : output_tensor, # i.e. `first_0`, `0`
                    f'predictions_{i}_{j}' : output_tensor, # i.e. `predictions_0_0`
                })

            # For each tensor, add default alias
            outputs[f'predictions_{i}'] = tensor
        return outputs

    @staticmethod
    def apply_output_operation(tensor, operation):
        """ !!. """
        with torch.no_grad():
            if operation is None:
                result = tensor
                name = ''
            elif operation == 'softplus':
                result = torch.nn.functional.softplus(tensor)
                name = operation
            elif operation == 'sigmoid':
                result = torch.sigmoid(tensor)
                name = operation
            elif operation == 'proba':
                result = torch.nn.functional.softmax(tensor, dim=1)
                name = operation
            elif operation == 'labels':
                result = tensor.argmax(dim=1)
                name = operation
            elif callable(operation):
                result = operation(tensor)
                name = operation.__name__
        return result, name


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
        self.parse_devices()

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

        self.model_to_device()

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
