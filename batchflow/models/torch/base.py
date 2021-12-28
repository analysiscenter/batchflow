""" Eager version of TorchModel. """
import os
import re
import inspect
from threading import Lock
from functools import partial
from itertools import zip_longest
from collections import OrderedDict

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
from .mixins import OptimalBatchSizeMixin, LayerHook, ExtractionMixin, VisualizationMixin
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


class TorchModel(BaseModel, ExtractionMixin, OptimalBatchSizeMixin, VisualizationMixin):
    """ Base class for Torch models.

    Implements two main logics:
        - the first is to build PyTorch model as a sequence of configurable nn.Modules
        - the second is to make infrastructure for model training, e.g. loss, optimizer and decay,
        and provide methods for the model training and inference.
    In the `examples` section you can find a drop-in template for your model.

    All of the parameters for both logics are defined in the config, supplied at initialization.
    The detailed description can be seen at `parameters` section; here, we describe the overall structure of keys:
        - global `cuda` parameters:
            - `device` sets the desired accelerator to use. Default is to use the single best available (GPU over CPU).
            - `benchmark` defines the `cuda` behavior: trade some GPU memory to get minor (~15%) acceleration.
            Default is True.

        - PyTorch model configuration.
            - `order` defines the sequence of blocks to build the model from. Default is initial_block -> body -> head.
            Separation of the NN into multiple blocks is just for convenience, so we can split
            the preprocessing, main body of the model, and postprocessing into individual parts.
            In the simplest case, each element is a string that points to other key in the config,
            which is used to create a :class:`~.torch.layers.ConvBlock`.
            Check the detailed description for more complex cases.
            - `initial_block`, `body`, `head` are parameters for this respective parts of the neural network.
            Defaults are empty layouts, meaning no operations.
            - `common` parameters are passed to each of the neural network parts. Default is empty.
            - `output` defines additional operations, applied to the output after loss computation.
            By default, we have `predictions`, `predictions_{i}` and `predictions_{i}_{j}` aliases.
            Note that these do not interfere with loss computation and are here only for convenience.
            - `init_weights` allows to initialize weights.

        - shapes info. If fully provided, used to initialize the model. If no shapes are given in the config,
        the model is created at the time of the first `train` call by looking at the actual batch data and shapes.
        Keys are `inputs_shapes`, `targets_shapes`, `classes`, and `placeholder_batch_size`.
        By default, no shapes are set in the config.

        - train and inference common parameters:
            - `amp` turns on/off automatic mixed precision, which allows to perform some of the operations in `float16`.
            Default is True.
            - `microbatch_size` allows to split the training/inference batches in chunks (microbatches) and process
            them sequentially. During train, we apply gradients only after all microbatches from the batch are used.
            Default is to not use microbatching.

        - train only parameters:
            - `sync_frequency` to apply gradients only once in a `sync_frequency` calls to `train` method.
            Default is to apply gradients after each `train` iteration.
            - `callbacks` to apply operations at the end of each iteration. Default is no callbacks.
            - `sam_rho`, `sam_individual_norm` to use sharpness-aware minimization. Default is to not use SAM at all.
            - `profile` to get detailed report of model performance. Default is False.

        - infrastructure for training:
            - `loss`. No default value, so this key is required.
            - `optimizer`. Default is `Adam`.
            - `decay`. Default is to not use learning rate decay.


    We recommend looking at :class:`~.torch.layers.ConvBlock` to learn about parameters for model building blocks,
    and at :class:`~.EncoderDecoder` which allows more sophisticated logic of block chaining.


    Parameters
    ----------
    config : dict, :class:`~Config`
        Configuration of model creation. Below are the valid keys.

    # Global parameters
    device : str, torch.device or sequence
        Device to use for model, training and inference.
        If str, a device name (e.g. ``'cpu'`` or ``'gpu:0'``). Regular expressions are also allowed (e.g. ``'gpu:*'``).
        If torch.device, then device to be used.
        If sequence, then each entry must be in one of previous formats, and batch data is paralleled across them.
        Default behaviour is to use one (and only one) device of the best available type (priority to GPU over CPU).

    benchmark : bool
        Whether to optimize network's forward pass during the first batch.
        Leverages the memory-speed trade-off: the network may use more GPU memory to compute predictions faster.
        Speeds up the forward pass by ~15% if shapes of inputs are constant.
        Make sure not to use different shapes of inputs.


    # Model building configuration
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
        User-defined module or parameters for the input block, usually :class:`~.torch.layers.ConvBlock` parameters.

        Examples:

        - ``{'initial_block': dict(layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``
        - ``{'initial_block': MyCustomModule(some_param=1, another_param=2)}``

    body : dict or nn.Module
        User-defined module or parameters for the base network layers,
        usually :class:`~.torch.layers.ConvBlock` parameters.

    head : dict or nn.Module
        User-defined module or parameters for the prediction layers,
        usually :class:`~.torch.layers.ConvBlock` parameters.

    common : dict
        Default parameters for all blocks (see :class:`~.torch.layers.ConvBlock`).

    output : str, list or dict
        Auxiliary operations to apply to the network predictions.
        If dict, then should have the same length and order as network predictions.
        Each key defines this prediction name, each value should be a str/list of operations to apply to this tensor.
        For example, ``{'my_prediction' : ['sigmoid', my_callable, 'softmax]}``.
        Generated outputs are available as `my_prediction_{j}`, `my_prediction_sigmoid`,
        and also by alias `predictions_{i}_{j}`, where `i` is the tensor ordinal and `j` is operation ordinal.

        If list or str, then default prefix `''` is used.
        See :meth:`.TorchModel.output` for more details.

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


    # Shapes: optional
    inputs_shapes : sequence
        Shapes of the input tensors without the batch size.
        Must be a tuple (one input) or sequence of tuples (multiple inputs) with shapes.

    targets_shapes : sequence
        Shapes of the target tensors without the batch size.
        Must be a tuple (one target) or sequence of tuples (multiple targets) with shapes.
        Available as `targets_shapes` parameter in the `head` block.

    classes : int or sequence of ints
        Number of desired classes in the output tensor. Available as `classes` parameter in the `head` block.

    placeholder_batch_size : int
        If `inputs` is specified with all the required shapes, then it serves as size of batch dimension during
        placeholder (usually np.ndarrays with zeros) creation. Default value is 2.


    # Train and inference behavior
    amp : bool
        Whether to use automated mixed precision during model training and inference. Default is True.
        The output type of predictions remains float32. Can be changed in `train` and `predict` arguments.

    microbatch_size : int, bool or None
        Also known as virtual batch. Allows to process given data sequentially,
        accumulating gradients from microbatches and applying them once in the end.
        If int, then size of chunks to split every batch into.
        If False or None, then this feature is not used. Default is not to use microbatching.
        Can be changed in `train` and `predict` arguments.


    # Additional train modifications
    sync_frequency : int
        How often to apply accumulated gradients to the weights. Default value is to apply them after each batch.
        Can be changed in `train` and `predict` arguments.

    callbacks : sequence of `:class:callbacks.BaseCallback`
        Callbacks to call at the end of each training iteration.

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


    # Infrastructure
    loss : str, dict
        Loss function, might be defined in multiple formats.

        If str, then short ``name``.
        If dict, then ``{'name': name, **kwargs}``.

        Name must be one of:
            - short name (e.g. ``'mse'``, ``'ce'``, ``'l1'``, ``'cos'``, ``'hinge'``,
              ``'huber'``, ``'logloss'``, ``'dice'``)
            - a class name from `torch losses <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
              (e.g. ``'PoissonNLL'`` or ``'TripletMargin'``)
            - an instance or constructor of `:class:torch.nn.Module`
            - callable

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': {'name': 'KLDiv', 'reduction': 'none'}}``
        - ``{'loss': {'name': MyCustomLoss, 'epsilon': 1e-6}}``
        - ``{'loss': my_custom_loss_fn}``
        - ``{'loss': my_custom_loss_class}``

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


    Examples
    --------
    segmentation_config = {
        # Model layout
        'initial_block': {'layout': 'cna cna cnap',                    # string layout: c=conv, n=BN, a=act, p=pool
                          filters: [INT, INT, INT],                    # individual filters for each convolution
                          'kernel_size': 3},                           # common kernel_size for all convolutions

        'body': {'base_block': ResBlock,                               # in ConvBlock, we can use any nn.Module as base
                 'filters': INT, 'kernel_size': INT,
                 'downsample': False, 'attention': 'scse'},            # additional parameters of ResBlock module

        'head': {'layout' : 'cna', 'filters': 1},                      # postprocessing
        'output': 'sigmoid',                                           # can get `sigmoid` output in the `predict`

        # Train configuration
        'loss': 'bdice',                                               # binary dice coefficient as loss function
        'optimizer': {'name': 'Adam', 'lr': 0.01,},
        'decay': {'name': 'exp', 'gamma': 0.9, 'frequency': 100},
        'microbatch_size': 16,                                         # size of microbatches at training
    }
    """
    PRESERVE = [
        'full_config', 'config', 'model',
        'inputs_shapes', 'targets_shapes', 'classes',
        'loss', 'optimizer', 'decay', 'decay_step',
        'sync_counter', 'microbatch_size',
        'iteration', 'last_train_info', 'last_predict_info',
        'lr_list', 'syncs', 'decay_iters',
        '_loss_list', 'loss_list',
    ]

    def __init__(self, config=None):
        self.full_config = Config(config)
        self.model_lock = Lock()

        # Shapes of inputs and targets
        self.placeholder_batch_size = 2
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
        self.microbatch_size = None

        # Sharpness-aware minimization
        self.sam_rho = 0.0
        self.sam_individual_norm = True

        # Store info about passed train/predict iterations
        self.iteration = 0
        self.last_train_info = {}
        self.last_predict_info = {}
        self.lr_list = []
        self.syncs = []
        self.decay_iters = []
        self._loss_list = []
        self.loss_list = []

        # Profile kernels used
        self.profile = False
        self.profilers = []
        self.profile_info = None

        # Store the config for later usage
        self.external_config = Config(config)

        #
        load = self.external_config.get('load')
        if load:
            self.load(**load)
        else:
            self.initialize()


    def initialize(self):
        """ Initialize the instance: make the config, attributes, and, if possible, PyTorch model. """
        # Create config from default and external one
        self.config = self.combine_configs()

        # First, extract all necessary info from config into the instance attributes.
        # Then, update config with some of parsed values -- mainly for convenience.
        self.parse_attributes()
        self.update_config()

        # If the inputs are set in config with their shapes we can build right away
        if self.inputs_shapes:
            self.build_model()

    def reset(self):
        """ Delete the underlying model and all the infrastructure. Use to create model from scratch. """
        # TODO: do we really need this?
        self.model = None
        self.last_train_info = {}


    # Create config of model creation: combine the external and default ones
    @classmethod
    def default_config(cls):
        """ Define model defaults.

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        Don't forget to use the default config from parent class.
        """
        config = Config({
            # Devices and memory control
            'amp': True,
            'device': None,
            'benchmark': True,
            'microbatch_size': False,
            'sync_frequency': 1,
            'profile': False,

            # Model building
            'order': ['initial_block', 'body', 'head'],
            'initial_block': {},
            'body': {},
            'head': {},
            'common': {},

            # Additional operations to apply to model predictions
            'output': None,

            # Shapes
            'placeholder_batch_size': 2,

            # Training infrastructure
            'loss': None,
            'optimizer': 'Adam',
            'decay': None,

            # SAM: sharpness-aware minimization
            'sam_rho': 0.0,
            'sam_individual_norm': True,
        })
        return config

    def combine_configs(self):
        """ Combine default configuration and the external one. """
        config = self.default_config() + self.external_config
        return config

    def update_config(self):
        """ Update config with instance attributes. """
        config = self.config

        config['head/targets_shapes'] = self.targets_shapes
        # As `update_config` can be called multiple times, and `head/classes` key can have value `None`,
        # we need to use `or` insetad of `get`
        config['head/classes'] = config.get('head/classes') or self.classes

        if config.get('head/units') is None:
            config['head/units'] = config.get('head/classes')
        if config.get('head/filters') is None:
            config['head/filters'] = config.get('head/classes')


    # Parse config keys into instance attributes
    def parse_attributes(self):
        """ Parse instance attributes from config. """
        config = self.config

        self.init_weights = config.get('init_weights', None)
        self.microbatch_size = config.get('microbatch_size', config.get('microbatch', False))
        self.sync_frequency = config.get('sync_frequency', 1)
        self.amp = config.get('amp', True)

        self.sam_rho = config.get('sam_rho', 0.0)
        self.sam_individual_norm = config.get('sam_individual_norm', False)
        self.profile = config.get('profile', False)

        self.callbacks = [callback.set_model(self) for callback in config.get('callbacks', [])]

        # Parse operations, that should be applied to model predictions, into a dictionary
        operations = config['output']
        if not isinstance(operations, dict):
            operations = operations or []
            operations = list(operations) if isinstance(operations, (tuple, list)) else [operations]
            operations = {'' : operations}
        self.operations = operations

        self._parse_devices()
        self._parse_placeholder_shapes()

    def _parse_devices(self):
        """ Extract `devices` and `benchmark` from config.
        If the config value is not set, use the best available accelerator.
        """
        devices = self.config.get('device')

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

        torch.backends.cudnn.benchmark = self.config.get('benchmark', 'cuda' in self.device.type)

    def _parse_placeholder_shapes(self):
        """ Extract `inputs_shapes`, `targets_shapes`, `classes` from config. """
        config = self.config

        batch_size = config.get('placeholder_batch_size', 2)
        inputs_shapes = config.get('inputs_shapes') or config.get('input_shapes')
        targets_shapes = config.get('targets_shapes') or config.get('target_shapes')
        classes = config.get('classes')

        self.placeholder_batch_size = batch_size

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

    def make_placeholder_data(self, batch_size=None):
        """ Create a sequence of tensor, based on the parsed `inputs_shapes`. """
        batch_size = batch_size or self.placeholder_batch_size

        data = [np.zeros((batch_size, *shape[1:]), dtype=np.float32)
                for shape in self.inputs_shapes]
        return data


    # Create training infrastructure: loss, optimizer, decay
    def make_infrastructure(self):
        """ Create loss, optimizer and decay, required for training the model. """
        self.make_loss(**self._unpack('loss'))
        self.make_optimizer(**self._unpack('optimizer'))
        self.make_decay(**self._unpack('decay'), optimizer=self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()

    def _unpack(self, name):
        """ Get params from config. """
        # TODO: move all code here to make it more explicit
        unpacked = unpack_fn_from_config(name, self.config)
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
        elif isinstance(loss, type):
            # Class to make module
            pass
        elif callable(loss):
            # Callable: just pass other arguments in
            loss_fn = partial(loss, **kwargs)
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


    # Set pre-initialized model or chain multiple building blocks to create model
    def set_model(self, model):
        """ Set the underlying PyTorch model to a supplied one and update training infrastructure. """
        self.model = model
        self.initialize_weights()
        self.model_to_device()

        self.make_infrastructure()


    def build_model(self, inputs=None):
        """ Create the instance of PyTorch model by chaining multiple blocks sequentially.
        After it, create training infrastructure (loss, optimizer, decay).

        The order is defined by `order` key in the config, which is [`initial_block`, `body`, `head`] by default.
        Each item in `order` should describe the block name, the config name and method to create. It can be a:
            - string, then we use it as name, config key and method name
            - tuple of three elements, which are name, config key and method name or callable
            - dictionary with three items, which are `block_name`, `config_name` and `method`.

            The `block_name` is used as the identifier in resulting model, i.e. `model.body`, `model.head`.
            The `config_name` is used to retrieve block creation parameters from config.
            The `method` is either a callable or name of the method to get from the current instance.
            Either method or callable should return an instance of nn.Module and accept block parameters.
        """
        inputs = inputs or self.make_placeholder_data()
        inputs = inputs[0] if len(inputs) == 1 else inputs
        inputs = self.transfer_to_device(inputs)

        blocks = OrderedDict()
        for item in self.config.get('order'):
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
            # Important: apply to the `inputs` before passing them to the next block, so the shapes/etc are updated
            block = self.make_block(config_name, method, inputs)

            if block is not None:
                block.to(self.device)
                inputs = block(inputs)
                blocks[block_name] = block

        # Use the OrderedDict in Sequential to give readable names to stages
        self.model = nn.Sequential(blocks)
        self.initialize_weights()
        self.model_to_device()

        self.make_infrastructure()

    def make_block(self, name, method, inputs):
        """ Create the block with `method` by retrieving its parameters from config by `name`. """
        config = self.config
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
                    kwargs = {**block, **block_params.get('module_kwargs', {})}
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
        """ Model building block. """
        kwargs = cls.get_block_defaults(name, kwargs)
        if kwargs.get('layout') or kwargs.get('base_block'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def initial_block(cls, inputs, **kwargs):
        """ Transform inputs. Usually used for initial preprocessing, e.g. reshaping, downsampling etc.
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        return cls.block(inputs, name='initial_block', **kwargs)

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Base layers which produce a network embedding.
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        return cls.block(inputs, name='body', **kwargs)

    @classmethod
    def head(cls, inputs, **kwargs):
        """ Produce predictions. Usually used to make network output compatible with the `targets` tensor.
        For parameters see :class:`~.torch.layers.ConvBlock`.

        Returns
        -------
        torch.nn.Module or None
        """
        return cls.block(inputs, name='head', **kwargs)


    # Model weights initialization
    def initialize_weights(self):
        """ Initialize model weights with a pre-defined or supplied callable. """
        if self.model and (self.init_weights is not None):
            # Parse model weights initilaization
            if isinstance(self.init_weights, str):
                # We have only one variant of predefined init function, so we check that init is str for a typo case
                # The common used non-default weights initialization:
                self.init_weights = best_practice_resnet_init

            # Actual weights initialization
            self.model.apply(self.init_weights)


    # Transfer to/from device(s)
    def transfer_to_device(self, data):
        """ Transfer (possibly nested) data structure to device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return type(data)({key : self.transfer_to_device(value) for key, value in data.items()})

        if isinstance(data, (tuple, list)):
            return type(data)(self.transfer_to_device(item) for item in data)

        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
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
        raise TypeError('Passed data should either be a `np.ndarray`, `torch.Tensor`, `cupy.ndarray`, '
                        f'or a container of them, got{type(data)}.')

    def transfer_from_device(self, data):
        """ Transfer (possibly nested) data structure from device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return type(data)({key : self.transfer_from_device(value) for key, value in data.items()})

        if isinstance(data, (tuple, list)):
            return type(data)(self.transfer_from_device(item) for item in data)

        if isinstance(data, (torch.Tensor, torch.autograd.Variable)):
            cpu_tensor = data.detach().cpu().numpy()
            if self.amp and cpu_tensor.dtype != np.float32:
                cpu_tensor = cpu_tensor.astype(np.float32)
            return cpu_tensor

        if isinstance(data, (np.ndarray, int, float)):
            return data
        raise TypeError('Passed data should either be a `np.ndarray`, `torch.Tensor`'
                        f' or a container of them, got {type(data)}.')

    def model_to_device(self):
        """ Put model on device(s). If needed, apply DataParallel wrapper. """
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, self.devices)
        else:
            self.model.to(self.device)


    # Apply model to train/predict on given data
    def train(self, inputs, targets, outputs=None, lock=True, profile=False,
              sync_frequency=True, microbatch_size=None, microbatch_drop_last=True,
              sam_rho=None, sam_individual_norm=None):
        """ Train the model with the data provided

        Parameters
        ----------
        inputs : np.ndarray or sequence of them
            Model inputs. If there is a single input, then it is passed to model directly; otherwise, we pass a list.
            If the microbatching is used, individual elements are split along the first axis.
        targets : np.ndarray or sequence of them
            Model targets to calculate loss with.
            If there is a single target, then it is passed to loss computation directly; otherwise, we pass a list.
            If the microbatching is used, individual elements are split along the first axis.
        outputs : str or sequence of them
            Desired outputs of the method.
            Each string defines a tensor to get and should be one of pre-defined or set in `outputs` key in the config.
            Pre-defined tensors are `predictions`, `loss`, and `predictions_{i}` for multi-output models.
        lock : bool
            If True, then model, loss and gradient update operations are locked, thus allowing for multithreading.
        sync_frequency : int, bool or None
            If int, then how often to apply accumulated gradients to the weights.
            If True, then value from config is used.
            Default value is 1, which means to apply gradients after each batch of data.
            If False or None, then gradients are applied after each batch of data.
        microbatch_size : int, bool or None
            If int, then size of chunks to split every batch into. Allows to process given data sequentially,
            accumulating gradients from microbatches and applying them once in the end.
            If None, then value from config is used (default value is not to use microbatching).
            If False, then microbatching is not used.
        microbatch_drop_last : bool
            Whether to drop microbatches, that are smaller than the microbatch size. Default is True.
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

        Returns
        -------
        Calculated values of requested tensors from `outputs` in the same order.

        Examples
        --------
        .. code-block:: python

            model.train(B('images'), B('labels'), fetches='loss')
        """
        # Lock the entire method; release in any case
        try:
            if lock:
                self.model_lock.acquire()
            self.last_train_info = {}

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

            # Prepare parameters for SAM
            if sam_rho is None:
                sam_rho = self.sam_rho
            if sam_individual_norm is None:
                sam_individual_norm = self.sam_individual_norm

            # Split the data into `microbatch_size` size chunks
            (chunked_inputs, chunked_targets,
             batch_size, microbatch_size) = self.split_into_microbatches(inputs, targets,
                                                                         microbatch_size, microbatch_drop_last)

            steps = len(chunked_inputs)
            inputs_shapes = [get_shape(item) for item in chunked_inputs[-1]]
            targets_shapes = [get_shape(item) for item in chunked_targets[-1]]
            self.last_train_info.update({'inputs_shapes': inputs_shapes,
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
                # Compute forward and backward passes of the model, apply gradients, evaluate requested outputs
                chunk_outputs = self._train(inputs=chunk_inputs, targets=chunk_targets, outputs=outputs[:],
                                            sync_frequency=sync_frequency*steps,
                                            sam_rho=sam_rho, sam_individual_norm=sam_individual_norm)
                chunked_outputs.append(chunk_outputs)

            # Exit the profiling
            if profile:
                profiler.__exit__(None, None, None)
                self.profilers.append(profiler)

            # Call the callbacks
            for callback in self.callbacks:
                callback.on_iter_end()

            # Aggregate the outputs from microbatches
            result = self.aggregate_microbatches(outputs, chunked_outputs, single_output)

            # Store the average value of loss over microbatches
            self.loss_list.append(np.mean(self._loss_list[-steps:]))

            # Store info about current train iteration
            self.last_train_info.update({
                'amp': self.amp,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
                'sync_frequency': sync_frequency,
                'steps': steps,
                'sam': bool(sam_rho), 'sam_rho': sam_rho,
                'sam_individual_norm': sam_individual_norm,
                'outputs': outputs,
            })

        finally:
            if lock:
                self.model_lock.release()
        return result

    def _train(self, inputs, targets, outputs, sync_frequency, sam_rho, sam_individual_norm):
        # Parse inputs
        inputs = inputs[0] if len(inputs) == 1 else inputs
        targets = targets[0] if len(targets) == 1 else targets
        inputs = self.transfer_to_device(inputs)
        targets = self.transfer_to_device(targets)

        # Convert layer ids into LayerHooks
        outputs = self.prepare_outputs(outputs)

        # Compute predictions; store shapes for introspection
        with torch.cuda.amp.autocast(enabled=self.amp):
            predictions = self.model(inputs)

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
        additional_outputs = self.compute_outputs(predictions=predictions)
        output_container = {
            **additional_outputs,
            'predictions': predictions,
            'loss': loss,
        }

        # Log inner info
        predictions_ = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        self.last_train_info['predictions_shapes'] = [get_shape(item) for item in predictions_]
        self.last_train_info['available_outputs'] = list(output_container.keys())

        # Retrieve requested outputs
        requested_outputs = self.extract_outputs(outputs, output_container)

        # Transfer only the requested outputs to CPU
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


    def predict(self, inputs, targets=None, outputs=None, lock=True, microbatch_size=False):
        """ Get predictions on the data provided.

        Parameters
        ----------
        inputs : np.ndarray or sequence of them
            Model inputs. Passed directly to model.
        targets : np.ndarray or sequence of them
            Optional model targets to calculate loss with. Passed directly to model.
        outputs : str or sequence of them
            Desired outputs of the method.
            Each string defines a tensor to get and should be one of:
                - pre-defined tensors, which are `predictions`, `loss`, and `predictions_{i}` for multi-output models.
                - values described in the `outputs` key in the config
                - layer id, which describes how to access the layer through a series of `getattr` and `getitem` calls.
                Allows to get intermediate activations of a neural network.
        lock : bool
            If True, then model and loss computation operations are locked, thus allowing for multithreading.
        microbatch_size : int, bool or None
            If int, then size of chunks to split every batch into. Allows to process given data sequentially.
            If None, then value from config is used (default value is not to use microbatching).
            If False, then microbatching is not used.

        Returns
        -------
        Calculated values of tensors in `outputs` in the same order.

        Examples
        --------
        Layer ids allow to get intermediate activations. If the model has `batchflow_model.model.head[0]` layer,
        you can access it with::

        >>> batchflow_model.predict(inputs=B.images, outputs='model.head[0]')

        String keys for `getitem` calls are also allowed::

        >>> batchflow_model.predict(inputs=B.images, outputs='model.body.encoder["block-0"]')
        """
        # Acquire lock; release in any case
        try:
            if lock:
                self.model_lock.acquire()
            self.last_predict_info = {}

            # Parse inputs and targets: always a list
            inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
            targets = (list(targets) if isinstance(targets, (tuple, list)) else [targets]) if targets else []

            # Parse outputs: always a list
            single_output = isinstance(outputs, str)
            outputs = [outputs] if single_output else (outputs or [])

            # Raise error early
            if 'loss' in outputs and not targets:
                raise TypeError('`targets` should be provided to fetch `loss`!')

            # Split the data into `microbatch` size chunks
            (chunked_inputs, chunked_targets,
             batch_size, microbatch_size) = self.split_into_microbatches(inputs, targets,
                                                                         microbatch_size, drop_last=False)

            steps = len(chunked_inputs)
            inputs_shapes = [get_shape(item) for item in chunked_inputs[-1]]
            targets_shapes = [get_shape(item) for item in chunked_targets[-1]]
            self.last_predict_info.update({'inputs_shapes': inputs_shapes,
                                           'targets_shapes': targets_shapes})

            # Evaluate each microbatch separately
            self.model.eval()

            chunked_outputs = []
            for chunk_inputs, chunk_targets in zip(chunked_inputs, chunked_targets):
                # Evaluate requested outputs
                chunk_outputs = self._predict(inputs=chunk_inputs, targets=chunk_targets, outputs=outputs[:])
                chunked_outputs.append(chunk_outputs)

            # Aggregate the outputs from microbatches
            result = self.aggregate_microbatches(outputs, chunked_outputs, single_output)

            # Store info about current predict iteration
            self.last_predict_info.update({
                'amp': self.amp,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
                'steps': steps,
                'outputs': outputs,
            })

        finally:
            if lock:
                self.model_lock.release()
        return result

    def _predict(self, inputs, targets, outputs):
        # Parse inputs
        inputs = inputs[0] if len(inputs) == 1 else inputs
        targets = targets[0] if len(targets) == 1 else targets

        # Convert layer ids into LayerHooks
        outputs = self.prepare_outputs(outputs)

        output_container = {}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp):
            inputs = self.transfer_to_device(inputs)
            predictions = self.model(inputs)

            output_container['predictions'] = predictions

            if targets:
                targets = self.transfer_to_device(targets)
                loss = self.loss(predictions, targets)
                output_container['loss'] = loss

        # Make all possible outputs
        additional_outputs = self.compute_outputs(predictions=predictions)
        output_container.update(additional_outputs)

        # Log inner info
        predictions_ = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        self.last_train_info['predictions_shapes'] = [get_shape(item) for item in predictions_]
        self.last_predict_info['available_outputs'] = list(output_container.keys())

        # Retrieve requested outputs
        requested_outputs = self.extract_outputs(outputs, output_container)

        # Transfer only the requested outputs to CPU
        return self.transfer_from_device(requested_outputs)


    # Common utilities for train and predict
    def split_into_microbatches(self, inputs, targets, microbatch_size, drop_last):
        """ Split inputs and targets into microbatch-sized chunks. """
        # Parse microbatch size
        if microbatch_size is None:
            microbatch_size = self.microbatch_size

        # Compute batch_size and make sure it is the same for all inputs and targets
        batch_size = len(inputs[0])
        for i, item in enumerate(inputs):
            if len(item) != batch_size:
                raise ValueError('All of `inputs` should have the same batch_size, as the first one!'
                                    f'Input at position `{i}` has batch_size {len(item)}!={batch_size}')
        for i, item in enumerate(targets):
            if len(item) != batch_size:
                raise ValueError('All of `targets` should have the same batch_size, as the first of `inputs`!'
                                    f'Target at position `{i}` has batch_size {len(item)}!={batch_size}')

        # Split data into microbatches, if needed
        if microbatch_size:
            chunked_inputs = [[item[i:i + microbatch_size] for item in inputs]
                                for i in range(0, batch_size, microbatch_size)]
            chunked_targets = [[item[i:i + microbatch_size] for item in targets]
                                for i in range(0, batch_size, microbatch_size)]

            if drop_last and batch_size % microbatch_size != 0:
                chunked_inputs = chunked_inputs[:-1]
                chunked_targets = chunked_targets[:-1]
        else:
            chunked_inputs = [inputs]
            chunked_targets = [targets]

        return chunked_inputs, chunked_targets, batch_size, microbatch_size

    def aggregate_microbatches(self, outputs, chunked_outputs, single_output):
        """ Aggregate outputs from microbatches into outputs for the whole batch.
        Scalar values are aggregated by `mean`, array values are concatenated along the first (batch) axis.
        """
        result = []
        for i, _ in enumerate(outputs):
            # All tensors for current `output_name`
            chunked_output = [chunk_outputs[i] for chunk_outputs in chunked_outputs]

            if chunked_output[0].size != 1:
                result.append(np.concatenate(chunked_output, axis=0))
            else:
                result.append(np.mean(chunked_output))
        if single_output:
            result = result[0]

        return result


    def compute_outputs(self, predictions):
        """ Produce additional outputs, defined in the config, from `predictions`.
        Also adds a number of aliases to predicted tensors.
        """
        predictions = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]

        outputs = {}
        # Iterate over tensors in predictions and the corresponding output operations
        iterator = zip_longest(predictions, self.operations.items(), fillvalue=(None, None))
        for i, (tensor, (output_prefix, output_operations)) in enumerate(iterator):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'Network outputs are expected to be tensors, got {type(tensor)} instead.')

            if output_prefix is not None:
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
        """ Apply `operation`, possibly aliased with a string, to `tensor`. """
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


    def prepare_outputs(self, outputs):
        """ Add the hooks to all outputs that look like a layer id. """
        result = []
        for output_name in outputs:
            if self.is_layer_id(output_name):
                layer = self.get_layer(output_name)
                hook = LayerHook(layer)
                result.append(hook)
            else:
                result.append(output_name)
        return result

    def extract_outputs(self, outputs, output_container):
        """ Retrieve activation data from hooks, get other requested outputs from container. """
        requested_outputs = []
        for item in outputs:
            if isinstance(item, LayerHook):
                item.close()
                value = item.activation
            else:
                value = output_container[item]

            requested_outputs.append(value)
        return requested_outputs


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
        self._parse_devices()

        if kwargs.get('pickle_module') is None:
            kwargs['pickle_module'] = dill

        if self.device:
            checkpoint = torch.load(path, map_location=self.device, **kwargs)
        else:
            checkpoint = torch.load(path, **kwargs)

        # `load_config` is a reference to `self.external_config` used to update `config`
        # It is required since `self.external_config` is overwritten in the cycle below
        load_config = self.external_config

        for item in self.PRESERVE:
            setattr(self, item, checkpoint.get(item))
        self.config = self.config + load_config

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
