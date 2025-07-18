""" Eager version of TorchModel. """
import os
import re
import inspect
from math import ceil
from threading import Lock
from functools import partial
from contextlib import nullcontext
from collections import OrderedDict

import dill
import numpy as np

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR

from sklearn.decomposition import PCA

from ...utils_import import make_delayed_import
pd = make_delayed_import('pandas')

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .network import Network
from .base_mixins import LayerHook, ExtractionMixin, VisualizationMixin
from .base_batchsize_mixin import OptimalBatchSizeMixin
from .initialization import best_practice_resnet_init
from .losses import CrossEntropyLoss, BinaryLovaszLoss, LovaszLoss, SSIM, MSSIM
from .losses import binary as binary_losses, multiclass as multiclass_losses
from .utils import get_shape, get_size
from ..base import BaseModel
from ...config import Config
from ...utils_transforms import Normalizer
from .modules import EncoderModule, DecoderModule, MLPDecoderModule



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
    'cosw': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'cyclic': torch.optim.lr_scheduler.CyclicLR,
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
        - global cuda and memory parameters:
            - `device` sets the desired accelerator to use. Default is to use the single best available (GPU over CPU).
            - `benchmark` defines the `cuda` behavior: trade some GPU memory to get minor (~15%) acceleration.
            Default is True.
            - `channels_last` sets the model weights and tensors layout to `channels_last`,
            which may result in minor acceleration. Default is False.

        - PyTorch model configuration.
            - `model`. If provided, then value should be a ready-to-use nn.Module.
        Otherwise, relies on :class:`.network.Network` for building the model:
            - `order` defines the sequence of blocks to build the model from. Default is initial_block -> body -> head.
            Separation of the NN into multiple blocks is just for convenience, so we can split
            the preprocessing, main body of the model, and postprocessing into individual parts.
            In the simplest case, each element is a string that points to other key in the config,
            which is used to create a :class:`~.torch.layers.Block`.
            Check the detailed description for more complex cases.
            - `initial_block`, `body`, `head` are parameters for this respective parts of the neural network.
            Defaults are empty layouts, meaning no operations.
            - `common` parameters are passed to each of the neural network parts. Default is empty.
            - `init_weights` allows to initialize weights.

            - `output` defines additional operations, applied to the output after loss computation.
            By default, we have `predictions`, `predictions_{i}` and `predictions_{i}_{j}` aliases.
            Note that these do not interfere with loss computation and are here only for convenience.

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

        - additional parameters:
            - `sam` and `sam_rho` enable sharpness-aware minimization: a technique for improving model generatlization.
            - `weights_averaging` enables model weights averaging.
            - `gradient_clipping` enables backward hooks for gradient clipping.


    We recommend looking at :class:`~.torch.layers.Block` to learn about parameters for model building blocks,
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
        Default behavior is to use one (and only one) device of the best available type (priority to GPU over CPU).

    benchmark : bool
        Whether to optimize network's forward pass during the first batch.
        Leverages the memory-speed trade-off: the network may use more GPU memory to compute predictions faster.
        Speeds up the forward pass by ~15% if shapes of inputs are constant.
        Make sure not to use different shapes of inputs.


    # Model building configuration
    model : nn.Module, optional
        If provided, then this module is used as the model to train without any modifications.
        If provided, other model-related keys (`order`, `initial_block`, etc) are not used.

    order : sequence
        Defines sequence of network blocks in the architecture. Default is initial_block -> body -> head.
        Each element of the sequence must be either a string, which is used to retrieve module parameters from config.
        Module parameters should include `type` and other keyword arguments for its initialization.
        Refer to the documentation of :class:`.network.Network` for more details.

    initial_block : dict
        User-defined module or parameters for the preprocess layers, usually :class:`~.torch.layers.Block` parameters.
    body : dict or nn.Module
        User-defined module or parameters for the base network layers, usually :class:`~.torch.layers.Block` parameters.
    head : dict or nn.Module
        User-defined module or parameters for the postprocess layers, usually :class:`~.torch.layers.Block` parameters.
    common : dict
        Default parameters, passed for all modules.

    trainable : sequence, optional
        Names of model parts to train. Should be a subset of names in `order` and can be used to freeze parameters.

    init_weights : callable, 'best_practice_resnet', tuple, sequence of them or None
        Model weights initialization.
        If None, then default initialization is used.
        If 'best_practice_resnet', then common used non-default initialization is used.
        If callable, then callable applied to each layer.
        If tuple, then the first element should be of the types above, and the second defines a model part to apply on.
        If sequence, then each element should be of the types defined above: applies all init functions sequentially.

        Examples:

        - ``{'init_weights': 'best_practice_resnet'}``
        - .. code-block:: python

            def callable_init(module): # example of a callable for init
                if isinstance(module, nn.Linear):
                    nn.kaiming_normal_(module.weight)

            config = {'init_weights': callable_init}
        - ``{'init_weights': ('best_practice_resnet', 'body')}`` # applies only at `body` module


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

    weights_averaging : dict
        If provided, we create additional copy of the model,
        which is updated with weights from the main model during train.
        Subkeys `start_iter`, `frequency` and `last_iter` define the range and frequency of updates.
        `avg_fn` can be used to change the logic of updates:
            - `swa` makes it so that weights from each update contribute equally.
            - `ema` makes it so that weights are aggregated with exponential moving average.
            - a callable, that takes `averaged_parameter, model_parameter, num_averaged` can be passed.

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
        iteration. Each decay might have optional parameters 'start_iter' and 'last_iter'
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

        - ``{'decay': {'name: 'exp', 'frequency': 5, 'start_iter': 6, 'last_iter': 20}}``
        - ``{'decay': {'name': 'StepLR', 'steps_size': 10000, 'frequency': 5}}``
        - ``{'decay': {'name': MyCustomDecay, 'decay_rate': .5, 'frequency': 15, 'start_iter': 400}``
        - .. code-block:: python

            {'decay': [{'name': 'exp', 'gamma': 1, 'frequency': 1, 'last_iter': 900},
                       {'name': 'exp', 'gamma': 0.96, 'frequency': 2, 'start_iter': 901}]

    gradient_clipping : number or callable
        Enables gradient clipping as a backward hook.
        If number, then acts as a clipping threshold for clamp.
        If callable, then directly applied to each gradient during backprop.
        Note that this is different from the usual `PyTorch` way to modify gradients in-place after the entire backprop.


    Examples
    --------
    segmentation_config = {
        # Model layout
        'initial_block': {                                         # preprocessing
            'layout': 'cna cna cnap',                              # string layout: c=conv, n=BN, a=act, p=pool
            'channels': [INT, INT, INT],                           # individual channels for each convolution
            'kernel_size': 3                                       # common kernel_size for all convolutions
        },

        'body': {
            'base_block': ResBlock,                                # can use any nn.Module as base block
            'channels': INT, 'kernel_size': INT,
            'downsample': False, 'attention': 'scse'               # additional parameters of ResBlock module
        },

        'head': {                                                  # postprocessing
            'layout' : 'cna',
            'channels': 1
        },

        # Train configuration
        'loss': 'bdice',                                           # binary dice coefficient as loss function
        'optimizer': {'name': 'Adam', 'lr': 0.01,},                # optimizer configuration
        'decay': {'name': 'exp', 'gamma': 0.9, 'frequency': 100},  # lr decay scheduler
        'microbatch_size': 16,                                     # size of microbatches at training
    }
    """
    AVAILABLE_FORMATS = ("onnx", "openvino", "safetensors")

    PRESERVE = set([
        'full_config', 'config', 'model',
        'inputs_shapes', 'targets_shapes', 'classes',
        'loss', 'optimizer', 'scaler', 'decay', 'decay_step',
        'sync_counter', 'microbatch_size',
        'iteration', 'last_train_info', 'last_predict_info',
        'lr_list', 'syncs', 'decay_iters',
        '_loss_list', 'loss_list', 'operations'
    ])

    PRESERVE_ONNX = PRESERVE - set(['model', 'loss', 'optimizer', 'scaler', 'decay'])
    PRESERVE_OPENVINO = PRESERVE - set(['model', 'loss', 'optimizer', 'scaler', 'decay'])
    PRESERVE_SAFETENSORS = PRESERVE - set(['model', 'loss', 'optimizer', 'scaler', 'decay'])

    def __init__(self, config=None):
        if config is not None and not isinstance(config, (dict, Config)):
            config = {'load/path': config}
        self.model_lock = Lock()

        # Configs
        self.external_config = Config(config)
        self.full_config = Config(config)

        # Shapes of inputs and targets
        self.placeholder_batch_size = 2
        self.inputs_shapes = None
        self.targets_shapes = None
        self.classes = None

        # Pytorch model
        self.model = None
        self._model_cpu_backup = None
        self._loaded_from_onnx = None
        self._loaded_from_openvino = None

        # Leading device and list of all devices to use
        self.device = None
        self.devices = []

        # Train procedure and infrastructure
        self.disable_training = False
        self.loss = None
        self.optimizer = None
        self.decay = None
        self.decay_step = None

        self.amp = True
        self.scaler = None

        self.operations = {}
        self.callbacks = []
        self._hooks = []

        # Memory amortization: accumulate gradients to update weights later
        self.sync_frequency = 1
        self.sync_counter = 0
        self.microbatch_size = None

        # Sharpness-aware minimization
        self.sam_rho = 0.0
        self.sam_individual_norm = True

        # WA: model weight averaging
        self.weight_averaging = None
        self.wa_model = None
        self.wa_config = None
        self.wa_decay = None
        self.wa_iters = []
        self.wa_finalized = False

        # TTA: test time augmentations
        self.tta_wrapped = False

        # TRT: tensorRT
        self.trt_wrapped = False

        # Store info about passed train/predict iterations
        self.iteration = 0
        self.last_train_info = {}
        self.last_predict_info = {}
        self.lr_list = []
        self.syncs = []
        self.decay_iters = []
        self._loss_list = []
        self.loss_list = []

        # Profile
        self.profile = False
        self.profilers = []
        self.profile_info = None

        # Load model from file or initialize anew
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
        self.last_predict_info = {}


    # Create config of model creation: combine the external and default ones
    @classmethod
    def default_config(cls):
        """ Define model defaults.

        Put here all constants (like the number of channels, kernel sizes, block layouts, stride, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        Don't forget to use the default config from parent class.
        """
        config = Config({
            # Devices and memory control
            'amp': True,
            'device': None,
            'benchmark': True,
            'channels_last': False,
            'microbatch_size': False,
            'sync_frequency': 1,
            'profile': False,

            # Model building
            'order': ['initial_block', 'body', 'head'],
            'trainable': None,
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
            'gradient_clipping': None,
            'weights_averaging': None,

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
        # we need to use `or` instead of `get`
        config['head/classes'] = config.get('head/classes') or self.classes

        if config.get('head/features') is None:
            config['head/features'] = config.get('head/classes')
        if config.get('head/channels') is None:
            config['head/channels'] = config.get('head/classes')


    # Parse config keys into instance attributes
    def parse_attributes(self):
        """ Parse instance attributes from config. """
        config = self.config

        self.microbatch_size = config.get('microbatch', config.get('microbatch_size', False))
        self.sync_frequency = config.get('sync_frequency', 1)
        self.amp = config.get('amp', True)

        self.sam_rho = config.get('sam_rho', 0.0)
        self.sam_individual_norm = config.get('sam_individual_norm', False)
        self.profile = config.get('profile', False)

        self.callbacks = [callback.set_model(self) for callback in config.get('callbacks', [])]

        self._parse_devices()
        self._parse_placeholder_shapes()


    def _parse_devices(self):
        """ Extract `devices` and `benchmark` from config.
        If the config value is not set, use the best available accelerator.
        """
        config = self.external_config
        devices = config.get('device')

        if devices is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            devices = devices if isinstance(devices, list) else [devices]
            available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] + ['cpu']
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
                    raise TypeError(f'Wrong device type: {type(dev)}')
            self.devices = [device for i, device in enumerate(self.devices)
                            if device not in self.devices[:i]]
            self.device = self.devices[0]

        if self.device.type == 'cpu':
            #TODO: maybe, we should add warning
            self.amp = False
        torch.backends.cudnn.benchmark = config.get('benchmark', 'cuda' in self.device.type)

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

    def make_placeholder_data(self, batch_size=None, unwrap=True, to_device=True):
        """ Create a sequence of tensor, based on the parsed `inputs_shapes`. """
        batch_size = batch_size or self.placeholder_batch_size

        data = [np.random.random((batch_size, *shape[1:])).astype(np.float32)
                for shape in self.inputs_shapes]

        if unwrap:
            data = data[0] if len(data) == 1 else data
        if to_device:
            data = self.transfer_to_device(data)
        return data


    # Create training infrastructure: loss, optimizer, decay
    def make_infrastructure(self):
        """ Create loss, optimizer and decay, required for training the model. """
        self.make_loss()
        self.make_optimizer()
        self.make_decay()
        self.scaler = torch.GradScaler("cuda")

        self.setup_gradient_clipping()
        self.setup_weights_averaging()

    def unpack(self, value):
        """ Unpack argument to actual value and kwargs. """
        if isinstance(value, (dict, Config)):
            kwargs = value.copy()
            value = kwargs.pop('name', None)
        else:
            kwargs = {}

        return value, kwargs

    def make_loss(self):
        """ Set model loss. Changes the `loss` attribute. """
        if not self.config.get('loss'):
            raise ValueError('Set "loss" in model configuration!')
        loss, kwargs = self.unpack(self.config['loss'])

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
            raise ValueError(f'Unknown loss: {loss}')

        loss_fn = loss_fn if loss_fn is not None else loss(**kwargs)
        if isinstance(loss_fn, nn.Module):
            loss_fn.to(device=self.device)

        self.loss = loss_fn

    def make_optimizer(self):
        """ Set model optimizer. Changes the `optimizer` attribute. """
        optimizer, kwargs = self.unpack(self.config['optimizer'])

        # Choose the optimizer
        if callable(optimizer) or isinstance(optimizer, type):
            pass
        elif isinstance(optimizer, str) and hasattr(torch.optim, optimizer):
            optimizer = getattr(torch.optim, optimizer)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer}')

        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def make_decay(self):
        """ Set model decay. Changes the `decay` and `decay_step` attribute. """
        decay = self.config['decay']

        if decay is None:
            decays = []
        else:
            decays = decay if isinstance(decay, (tuple, list)) else [decay]

        self.decay, self.decay_step = [], []
        for decay_ in decays:
            decay_, decay_kwargs = self.unpack(decay_)

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
                raise ValueError(f'Unknown learning rate scheduler: {decay_}')

            # Parse step parameters
            step_params = {
                'start_iter': 0,
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
                decay_kwargs = Config({**decay_dict, **decay_kwargs})
            else:
                decay_kwargs = Config(decay_kwargs)

            # Remove unnecessary keys from kwargs
            for key in ['start_iter', 'last_iter', 'frequency']:
                decay_kwargs.pop(key, default=None)

            # Create decay or store parameters for later usage
            decay_ = decay_(self.optimizer, **decay_kwargs)

            self.decay.append(decay_)
            self.decay_step.append(step_params)

    def setup_gradient_clipping(self):
        """ Clip gradients to avoid explosion. """
        gradient_clipping = self.config.get('gradient_clipping')
        if gradient_clipping:
            if isinstance(gradient_clipping, (int, float)):
                function = lambda grad: torch.clamp(grad, -gradient_clipping, gradient_clipping)
            elif callable(gradient_clipping):
                function = gradient_clipping
            else:
                raise ValueError(f'gradient_clipping must be int, float or callable but it is{type(gradient_clipping)}')

            for p in self.model.parameters():
                hook = p.register_hook(function)
                self._hooks.append(hook)

    def setup_weights_averaging(self):
        """ Prepare WA-model: check all required keys and store copy on CPU. """
        wa_config = self.config.get('weights_averaging') or self.config.get('wa') or self.config.get('swa')


        if wa_config is not None:
            required_keys = ['start_iter', 'last_iter', 'frequency']
            for key in required_keys:
                if key not in wa_config:
                    raise ValueError(f'Key `{key}` is missing in weights averaging configuration!')

            avg_fn = wa_config.get('avg_fn', None)
            if avg_fn in ['stochastic', 'swa']:
                avg_fn = None
            elif avg_fn in ['exponential', 'ema']:
                avg_fn = lambda wa_parameter, model_parameter, num_averaged: 0.1 * wa_parameter + 0.9 * model_parameter

            self.weight_averaging = True
            self.wa_config = Config(wa_config)
            self.wa_model = AveragedModel(self.model, device='cpu', avg_fn=avg_fn)

            if 'swalr' in wa_config:
                self.wa_decay = SWALR(self.optimizer, **wa_config['swalr'])


    # Set pre-initialized model or chain multiple building blocks to create model
    def set_model(self, model):
        """ Set the underlying PyTorch model to a supplied one and update training infrastructure. """
        self.model = model
        self.initialize_weights()
        self.model_to_device()

        self.make_infrastructure()

    def build_model(self, inputs=None):
        """ Create an instance of PyTorch model or use one provided.
        After it, create training infrastructure (loss, optimizer, decay).
        """
        if inputs is not None:
            inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs, list) else inputs
            inputs = self.transfer_to_device(inputs)
        else:
            inputs = self.make_placeholder_data(to_device=True)

        if 'model' not in self.config:
            with torch.no_grad():
                self.model = Network(inputs=inputs, config=self.config, device=self.device)
        else:
            self.model = self.config['model']

        self.initialize_weights()
        if self.config['channels_last']:
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model_to_device()
        self.make_infrastructure()

    def finalize_wa(self):
        """ Replace the model with weight-averaged one. """
        if self.weight_averaging and not self.wa_finalized:
            self.wa_iters.append(self.iteration)
            self.model = self.wa_model.module
            self.model_to_device()

            self.make_optimizer()
            self.scaler = torch.cuda.amp.GradScaler()

            self.wa_finalized = True

    def wrap_tta(self, wrapper='ClassificationTTAWrapper', transforms=None, merge_mode='mean'):
        """ Wrap model with test-time augmentations. """
        import ttach
        transforms = transforms if transforms is not None else ttach.aliases.vlip_transform()
        self.model = getattr(ttach, wrapper)(self.model, transforms=transforms, merge_mode=merge_mode)
        self.tta_wrapped = True

    def wrap_trt(self, batch_size, use_onnx=True, fp16_mode=True, **kwargs):
        """ Convert PyTorch model to TensorRT engine. """
        from torch2trt import torch2trt
        inputs = self.make_placeholder_data(batch_size=batch_size, unwrap=False)

        self.model = torch2trt(self.model.eval(), inputs=inputs, max_batch_size=batch_size,
                               fp16_mode=fp16_mode, use_onnx=use_onnx, **kwargs)
        self.trt_wrapped = True
        self.disable_training = True


    def initialize_weights(self):
        """ Initialize model weights with a pre-defined or supplied callable. """
        init_weights = self.config.get('init_weights', None)
        if self.model is not None and init_weights is not None:
            # Parse model weights initialization
            init_weights = init_weights if isinstance(init_weights, list) else [init_weights]

            for init_weights_ in init_weights:
                if isinstance(init_weights_, tuple) and len(init_weights_) == 2:
                    init_weights_function, init_weights_module = init_weights_
                else:
                    init_weights_function, init_weights_module = init_weights_, None

                if init_weights_function in {'resnet', 'classic'}:
                    init_weights_function = best_practice_resnet_init

                # Actual weights initialization
                if init_weights_module is None:
                    self.model.apply(init_weights_function)
                else:
                    getattr(self.model, init_weights_module).apply(init_weights_function)


    # Transfer to/from device(s)
    def transfer_to_device(self, data, non_blocking=False):
        """ Transfer (possibly nested) data structure to device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return type(data)({key : self.transfer_to_device(value) for key, value in data.items()})

        if isinstance(data, (tuple, list)):
            return type(data)(self.transfer_to_device(item) for item in data)

        if isinstance(data, np.ndarray):
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            data = torch.from_numpy(data)

            if self.config['channels_last'] and data.ndim == 4:
                data = data.to(memory_format=torch.channels_last)
            data = data.to(self.device, non_blocking=non_blocking)
            return data

        if isinstance(data, torch.Tensor):
            data = data.to(self.device, non_blocking=non_blocking)
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

    def transfer_from_device(self, data, force_float32_dtype=True):
        """ Transfer (possibly nested) data structure from device and return the same structure. """
        if isinstance(data, (dict, Config)):
            return type(data)({key : self.transfer_from_device(value, force_float32_dtype)
                               for key, value in data.items()})

        if isinstance(data, (tuple, list)):
            return type(data)(self.transfer_from_device(item, force_float32_dtype) for item in data)

        if isinstance(data, (torch.Tensor, torch.autograd.Variable)):
            # cpu_tensor = data.detach().cpu().numpy()
            # if self.amp and cpu_tensor.dtype != np.float32:
            #     cpu_tensor = cpu_tensor.astype(np.float32)
            # return cpu_tensor

            if force_float32_dtype and data.dtype != torch.float32:
                data = data.float()
            cpu_tensor = data.detach().cpu().numpy()
            return cpu_tensor

        if isinstance(data, (np.ndarray, int, float)):
            return data
        raise TypeError('Passed data should either be a `np.ndarray`, `torch.Tensor`'
                        f' or a container of them, got {type(data)}.')

    def model_to_device(self, model=None):
        """ Put model on device(s). If needed, apply DataParallel wrapper. """
        model_ = model if model is not None else self.model

        if len(self.devices) > 1:
            model_ = nn.DataParallel(model_, self.devices)
        else:
            model_ = model_.to(self.device)

        if model is None:
            self.model = model_
        return model_


    # Apply model to train/predict on given data
    def train(self, inputs, targets, outputs=None, mode='train', lock=True, profile=False,
              sync_frequency=True, microbatch_size=None, microbatch_drop_last=True, microbatch_pad_last=False,
              sam_rho=None, sam_individual_norm=None, transfer_from_device=True, force_float32_dtype=True):
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
        outputs : str, callable, sequence or dict of them
            Desired outputs of the method.
            Each element defines a tensor to get and should be one of:
                - a string, which can be:
                    - `loss`, `predictions`, and `predictions_{i}` for multi-output models.
                    - `softplus`, `sigmoid`, `sigmoid_uint8`, `sigmoid_int16`, `proba`, `labels`.
                    Work only with len(predictions) == 1.
                    - layer id, which describes how to access the layer through a series of `getattr` and
                    `getitem` calls.
                - a callable, which defines a function that should be applied to predictions.
                - a sequence, where each item is one of the previous types. Result of this method is guaranteed
                to have the same order of elements.
                - a dict, where each value is one of the previous types. Result of this method is a dictionary
                with the same keys and requested tensors as values.
        lock : bool
            If True, then model, loss and gradient update operations are locked, thus allowing for multithreading.
        mode : None, str or callable
            If None, then does nothing.
            If str, then identifies mode to put the model in: one of ``'train'`` or ``'eval'``.
            If callable, then applied to the model directly.
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
        microbatch_pad_last : bool
            Whether to pad microbatches, that are smaller than the microbatch size. Default is False.
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
        transfer_from_device : bool
            Whether to transfer requested `outputs` from device to CPU.
        force_float32_dtype : bool
            Whether to force dtype float32 to the model outputs.
            Otherwise, the dtype is preserved and may be affected by AMP.

        Returns
        -------
        Calculated values of requested tensors from `outputs`.

        Examples
        --------
        .. code-block:: python

            model.train(B('images'), B('labels'), outputs='loss')
        """
        if self.disable_training:
            raise RuntimeError('Training model after ONNX conversion is not allowed!')

        # Lock the entire method; release in any case
        try:
            if lock:
                self.model_lock.acquire()
            self.last_train_info = {}

            # Parse inputs and targets: always a list
            inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
            targets = list(targets) if isinstance(targets, (tuple, list)) else [targets]

            # Parse outputs: always a dict
            outputs_dict = self.convert_outputs(outputs)

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
            split_result = self.split_into_microbatches(inputs, targets, microbatch_size,
                                                        drop_last=microbatch_drop_last, pad_last=microbatch_pad_last)
            (chunked_inputs, chunked_targets, chunk_sizes, batch_size, microbatch_size) = split_result

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
                if not self.classes and len(targets_shapes) > 2:
                    self.classes = [shape[1] for shape in targets_shapes]

                self.update_config()

                # Can use the first two items to build model: no need for the whole tensor
                build_inputs = [item[:2] for item in chunked_inputs[0]]
                self.build_model(build_inputs)

            self.set_model_mode(mode)

            # Set up the profiling, if needed
            profile = profile or self.profile
            if profile:
                profiler = torch.autograd.profiler.profile(use_cuda='cpu' not in self.device.type)
                profiler.__enter__()

            # Train on each of the microbatches
            chunked_outputs = []
            for chunk_inputs, chunk_targets in zip(chunked_inputs, chunked_targets):
                # Compute forward and backward passes of the model, apply gradients, evaluate requested outputs
                chunk_outputs = self._train(inputs=chunk_inputs, targets=chunk_targets, outputs_dict=outputs_dict,
                                            sync_frequency=sync_frequency*steps,
                                            sam_rho=sam_rho, sam_individual_norm=sam_individual_norm,
                                            transfer_from_device=transfer_from_device,
                                            force_float32_dtype=force_float32_dtype)
                chunked_outputs.append(chunk_outputs)

            # Exit the profiling
            if profile:
                profiler.__exit__(None, None, None)
                self.profilers.append(profiler)

            # Call the callbacks
            for callback in self.callbacks:
                callback.on_iter_end()

            # Use current weights for weights averaging
            if self.weight_averaging:
                start_iter, frequency, last_iter = self.wa_config.get(['start_iter', 'frequency', 'last_iter'])

                if self.iteration >= last_iter and not self.wa_finalized:
                    self.finalize_wa()

                elif (start_iter <= self.iteration <= last_iter and
                    (self.iteration - start_iter) % frequency == 0):
                    self.wa_model.update_parameters(self.model)
                    self.wa_iters.append(self.iteration)

                    if self.wa_decay:
                        self.wa_decay.step()

            # Aggregate the outputs from microbatches
            result = self.aggregate_microbatches(outputs_dict=outputs_dict, chunked_outputs=chunked_outputs,
                                                 chunk_sizes=chunk_sizes)

            # convert results to tensor or list depending on outputs type. If dict, do nothing
            if isinstance(outputs, str) or callable(outputs):
                result = list(result.values())[0]
            elif isinstance(outputs, (tuple, list, set)):
                result = list(result.values())
            elif outputs is None:
                result = None

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

    def _train(self, inputs, targets, outputs_dict, sync_frequency, sam_rho, sam_individual_norm,
               transfer_from_device, force_float32_dtype):
        # Parse inputs
        inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs, list) else inputs
        targets = targets[0] if len(targets) == 1 and isinstance(targets, list) else targets
        inputs = self.transfer_to_device(inputs)
        targets = self.transfer_to_device(targets, non_blocking=True)

        # Compute predictions; store shapes for introspection
        with torch.amp.autocast('cuda', enabled=self.amp):
            predictions = self.model(inputs)

        # SAM: store grads from previous microbatches
        if self.iteration >= 1 and bool(sam_rho):
            self._train_sam_store_gradients()

        # Compute loss and gradients; store loss value for every microbatch
        with torch.amp.autocast('cuda', enabled=self.amp):
            loss = self.loss(predictions, targets)
            loss_ = loss / sync_frequency

        # nan_in_loss = torch.isnan(loss).max().item()
        # if not nan_in_loss:
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
                    step_cond = (self.iteration - decay_step['start_iter']) % decay_step['frequency'] == 0
                    range_cond = decay_step['start_iter'] <= self.iteration <= decay_step['last_iter']
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
        output_container = self.compute_outputs(predictions=predictions, outputs_dict=outputs_dict, loss=loss)

        # Log inner info
        predictions_ = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        self.last_train_info['predictions_shapes'] = [get_shape(item) for item in predictions_]
        self.last_train_info['available_outputs'] = list(output_container.keys())

        # Transfer only the requested outputs to CPU
        if transfer_from_device:
            output_container = self.transfer_from_device(output_container, force_float32_dtype=force_float32_dtype)
        return output_container

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
        with torch.amp.autocast('cuda', enabled=self.amp):
            predictions_inner = self.model(inputs)
            loss_inner = self.loss(predictions_inner, targets) / sync_frequency
        (self.scaler.scale(loss_inner) if self.amp else loss_inner).backward()

        # Cancel the previous update to model parameters, add stored gradients from previous microbatches
        params_with_grads = [p - eps for p, eps in zip(params_with_grads, epsilons)]

        for p in self.model.parameters():
            previous_grad = self.optimizer.state[p].get('previous_grad')
            if previous_grad is not None:
                p.grad.add_(previous_grad)


    def predict(self, inputs, targets=None, outputs=None, lock=True, microbatch_size=False, microbatch_pad_last=False,
                amp=None, mode='eval', no_grad=True, transfer_from_device=True, force_float32_dtype=True):
        """ Get predictions on the data provided.

        Parameters
        ----------
        inputs : np.ndarray or sequence of them
            Model inputs. Passed directly to model.
        targets : np.ndarray or sequence of them
            Optional model targets to calculate loss with. Passed directly to model.
        outputs : str, callable, sequence or dict of them
            Desired outputs of the method.
            Each element defines a tensor to get and should be one of:
                - a string, which can be:
                    - `loss`, `predictions`, and `predictions_{i}` for multi-output models.
                    - `softplus`, `sigmoid`, `sigmoid_uint8`, `sigmoid_int16`, `proba`, `labels`.
                    Work only with len(predictions) == 1.
                    - layer id, which describes how to access the layer through a series of `getattr` and
                    `getitem` calls.
                - a callable, which defines a function that should be applied to predictions.
                - a sequence, where each item is one of the previous types. Result of this method is guaranteed
                to have the same order of elements.
                - a dict, where each value is one of the previous types. Result of this method is a dictionary
                with the same keys and requested tensors as values.
        lock : bool
            If True, then model and loss computation operations are locked, thus allowing for multithreading.
        microbatch_size : int, bool or None
            If int, then size of chunks to split every batch into. Allows to process given data sequentially.
            If None, then value from config is used (default value is not to use microbatching).
            If False, then microbatching is not used.
        microbatch_pad_last : bool
            Whether to pad microbatches, that are smaller than the microbatch size. Default is False.
        amp : None or bool
            If None, then use amp setting from config.
            If bool, then overrides the amp setting for prediction.
        mode : None, str or callable
            If None, then does nothing.
            If str, then identifies mode to put the model in: one of ``'train'`` or ``'eval'``.
            If callable, then applied to the model directly.
        no_grad : bool
            Whether to disable gradient computation during model evaluation.
        transfer_from_device : bool
            Whether to transfer requested `outputs` from device to CPU.
        force_float32_dtype : bool
            Whether to force dtype float32 to the model outputs.
            Otherwise, the dtype is preserved and may be affected by AMP.

        Returns
        -------
        Calculated values of tensors in `outputs`.

        Examples
        --------
        Layer ids allow to get intermediate activations. If the model has `batchflow_model.model.head[0]` layer,
        you can access it with::

        >>> batchflow_model.predict(inputs=B.images, outputs='model.head[0]')

        String keys for `getitem` calls are also allowed::

        >>> batchflow_model.predict(inputs=B.images, outputs='model.body.encoder["block-0"]')
        """
        if self._loaded_from_onnx or self._loaded_from_openvino:
            microbatch_size = self.microbatch_size
            microbatch_pad_last = True

        # Acquire lock; release in any case
        try:
            if lock:
                self.model_lock.acquire()
            self.last_predict_info = {}

            # Parse outputs: always a dict
            outputs_dict = self.convert_outputs(outputs)

            # Raise error early
            if 'loss' in outputs_dict.values() and targets is None:
                raise TypeError('`targets` should be explicitly provided to compute `loss`!')

            # Parse inputs and targets: always a list
            inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
            if targets is not None:
                targets = (list(targets) if isinstance(targets, (tuple, list)) else [targets])
            else:
                targets = []

            # Parse other parameters
            amp = amp if amp is not None else self.amp

            # Split the data into `microbatch` size chunks
            split_result = self.split_into_microbatches(inputs, targets, microbatch_size,
                                                        drop_last=False, pad_last=microbatch_pad_last)
            (chunked_inputs, chunked_targets, chunk_sizes, batch_size, microbatch_size) = split_result

            steps = len(chunked_inputs)
            inputs_shapes = [get_shape(item) for item in chunked_inputs[-1]]
            targets_shapes = [get_shape(item) for item in chunked_targets[-1]]
            self.last_predict_info.update({'inputs_shapes': inputs_shapes,
                                           'targets_shapes': targets_shapes})

            # Evaluate each microbatch separately
            self.set_model_mode(mode)

            chunked_outputs = []
            for chunk_inputs, chunk_targets in zip(chunked_inputs, chunked_targets):
                # Evaluate requested outputs
                chunk_outputs = self._predict(inputs=chunk_inputs, targets=chunk_targets, outputs_dict=outputs_dict,
                                              amp=amp, no_grad=no_grad, transfer_from_device=transfer_from_device,
                                              force_float32_dtype=force_float32_dtype)
                chunked_outputs.append(chunk_outputs)

            # Aggregate the outputs from microbatches
            result = self.aggregate_microbatches(outputs_dict=outputs_dict, chunked_outputs=chunked_outputs,
                                                 chunk_sizes=chunk_sizes)

            # convert results to tensor or list depending on outputs type. If dict, do nothing
            if isinstance(outputs, str) or callable(outputs):
                result = list(result.values())[0]
            elif isinstance(outputs, (tuple, list, set)):
                result = list(result.values())
            elif outputs is None:
                result = None

            # Store info about current predict iteration
            self.last_predict_info.update({
                'amp': amp,
                'batch_size': batch_size,
                'microbatch_size': microbatch_size,
                'steps': steps,
                'outputs': outputs,
            })

        finally:
            if lock:
                self.model_lock.release()
        return result

    def _predict(self, inputs, targets, outputs_dict, amp, no_grad, transfer_from_device, force_float32_dtype):
        # Parse inputs
        inputs = inputs[0] if len(inputs) == 1 and isinstance(inputs, list) else inputs
        targets = targets[0] if len(targets) == 1 and isinstance(targets, list) else targets

        with (torch.no_grad() if no_grad else nullcontext()), torch.amp.autocast('cuda', enabled=amp):
            inputs = self.transfer_to_device(inputs)
            predictions = self.model(inputs)

            if len(targets) > 0:
                targets = self.transfer_to_device(targets)
                loss = self.loss(predictions, targets)
            else:
                loss = None

        # Make all requested outputs
        output_container = self.compute_outputs(predictions=predictions, outputs_dict=outputs_dict, loss=loss)

        # Log inner info
        predictions_ = list(predictions) if isinstance(predictions, (tuple, list)) else [predictions]
        self.last_predict_info['predictions_shapes'] = [get_shape(item) for item in predictions_]
        self.last_predict_info['available_outputs'] = list(output_container.keys())

        # Transfer only the requested outputs to CPU
        if transfer_from_device:
            output_container = self.transfer_from_device(output_container, force_float32_dtype=force_float32_dtype)
        return output_container


    def __call__(self, inputs, targets=None, outputs='predictions', lock=True,
                 microbatch_size=False, microbatch_pad_last=False,
                 amp=False, no_grad=False, transfer_from_device=False):
        """ Evaluate model on provided data, while tracking gradients.
        Essentially, the same as `:meth:.predict` with overriden defaults.
        """
        return self.predict(inputs=inputs, targets=targets, outputs=outputs,
                            microbatch_size=microbatch_size, microbatch_pad_last=microbatch_pad_last,
                            lock=lock, amp=amp, no_grad=no_grad, transfer_from_device=transfer_from_device)


    # Common utilities for train and predict
    def set_model_mode(self, mode):
        """ Set model mode to either train or eval. If provided with a callable, applies it to the model directly. """
        if mode in {'train', 'training'}:
            self.model.train()
        elif mode in {'eval', 'predict', 'inference'}:
            self.model.eval()
        elif mode is None:
            pass
        elif callable(mode):
            self.model.apply(mode)
        else:
            raise ValueError(f'Unknown model mode={mode}')

    def split_into_microbatches(self, inputs, targets, microbatch_size, drop_last, pad_last):
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
            chunk_sizes = [len(item[0]) for item in chunked_inputs]

            if batch_size % microbatch_size != 0:
                if pad_last:
                    chunked_inputs[-1] = [self._tile(item, microbatch_size) for item in chunked_inputs[-1]]
                    chunked_targets[-1] = [self._tile(item, microbatch_size) for item in chunked_targets[-1]]
                elif drop_last:
                    chunked_inputs = chunked_inputs[:-1]
                    chunked_targets = chunked_targets[:-1]
        else:
            chunked_inputs = [inputs]
            chunked_targets = [targets]
            chunk_sizes = [len(inputs[0])]

        return chunked_inputs, chunked_targets, chunk_sizes, batch_size, microbatch_size

    def _tile(self, array, microbatch_size):
        """ Tile `array` to make its length equal to `microbatch_size`. """
        n_repeats = ceil(microbatch_size / len(array))
        if isinstance(array, (np.ndarray, torch.Tensor)):
            repeats = [1] * array.ndim
            repeats[0] = n_repeats
            array = np.tile(array, repeats) if isinstance(array, np.ndarray) else array.tile(*repeats)
            array = array[:microbatch_size]
        return array

    def aggregate_microbatches(self, outputs_dict, chunked_outputs, chunk_sizes):
        """ Aggregate outputs from microbatches into outputs for the whole batch.
        Scalar values are aggregated by `mean`, array values are concatenated along the first (batch) axis. """
        result = OrderedDict()

        for output_name in outputs_dict:
            chunked_output = [chunk_outputs[output_name] for chunk_outputs in chunked_outputs]
            if get_size(chunked_output[0]) != 1:
                if len(chunked_output) == 1:
                    output_ = chunked_output[0][:chunk_sizes[0]]
                elif isinstance(chunked_output[0], np.ndarray):
                    output_ = np.concatenate([chunk_output[:chunk_size]
                                              for chunk_output, chunk_size in zip(chunked_output, chunk_sizes)], axis=0)
                else:
                    output_ = torch.cat([chunk_output[:chunk_size]
                                         for chunk_output, chunk_size in zip(chunked_output, chunk_sizes)], dim=0)
                result[output_name] = output_
            else:
                if isinstance(chunked_output[0], np.ndarray):
                    result[output_name] = np.mean(chunked_output)
                else:
                    result[output_name] = torch.mean(torch.stack(chunked_output))
        return result


    def compute_outputs(self, predictions, outputs_dict, loss=None):
        """ Produce additional outputs, defined in the outputs parameter of `train`
        or `predict` functions from predictions.
        """
        result = OrderedDict()
        for output_name, operation in outputs_dict.items():
            if operation == 'predictions':
                result[output_name] = predictions
            elif operation == 'loss':
                result[output_name] = loss
            else:
                result[output_name] = self.apply_output_operation(predictions, operation)
        return result

    @staticmethod
    def apply_output_operation(predictions, operation):
        """ Apply `operation`, possibly aliased with a string, to `predictions`. """
        with torch.no_grad():
            if operation is None:
                result = predictions
            elif callable(operation):
                result = operation(predictions)
            elif isinstance(operation, LayerHook):
                operation.close()
                result = operation.activation
            elif (isinstance(operation, str) and # Check if operation matches `predictions_[0-9]+`
                  operation.startswith('predictions_') and
                  operation.split("_")[-1].isdigit()):

                i = int(operation.split("_")[-1])
                result = predictions[i]
            else:
                if isinstance(predictions, (tuple, list)) and not len(predictions) == 1:
                    raise ValueError('Default operations can`t be applied to multi output predictions.')

                predictions = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
                if operation == 'softplus':
                    result = torch.nn.functional.softplus(predictions)
                elif operation == 'sigmoid':
                    result = torch.sigmoid(predictions)
                elif operation == 'sigmoid_uint8':
                    result = (torch.sigmoid(predictions) * 255).to(dtype=torch.uint8)
                elif operation == 'sigmoid_int16':
                    result = (torch.sigmoid(predictions) * 255).to(dtype=torch.int16)
                elif operation == 'proba':
                    result = torch.nn.functional.softmax(predictions, dim=1)
                elif operation == 'labels':
                    result = predictions.argmax(dim=1)
                else:
                    raise ValueError(f'Unknown type of operation `{operation}`!')
        return result


    def convert_outputs(self, outputs):
        """ Add the hooks to all outputs that look like a layer id. Also convert outputs to dict. """
        result = OrderedDict()

        if outputs is None:
            return OrderedDict()

        if not isinstance(outputs, dict): # then outputs should be str, callable, list or tuple
            outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
            processed_outputs = OrderedDict()
            for output in outputs:
                if isinstance(output, str) or callable(output):
                    processed_outputs[output] = output
                else:
                    raise ValueError(f"Expected elements in outputs to be either str or callable, \
                                       but got type {type(output)}!")
            outputs = processed_outputs

        # the outputs here should be dict
        for output_name, operation in outputs.items():
            if self.is_layer_id(output_name):
                layer = self.get_layer(output_name)
                hook = LayerHook(layer)
                result[output_name] = hook
            else:
                result[output_name] = operation
        return result


    # Store model
    def save(self, path, fmt=None, pickle_metadata=True, batch_size=None, opset_version=13, pickle_module=dill,
             ignore_attributes=('optimizer', 'decay'), **kwargs):
        """ Save underlying PyTorch model along with meta parameters (config, device spec, etc).

        If `use_onnx` is set to True, then the model is converted to ONNX format and stored in a separate file.
        At loading time, this ONNX module is converted to PyTorch back: the benefit of this process is removing
        the reliance on code files. The drawback is the need to fix `batch_size`, allowed for such a model at inference.
        Moreover, reloaded ONNX module can't be trained or modified in any other way.

        Parameters
        ----------
        path : str
            Path to a file where the model data will be stored.
        fmt: Optional[str]
            Weights format. Available formats: "onnx", "openvino", "safetensors".
        pickle_metadata: bool
            Whether dump metadata (see `PRESERVE` attribute) to the file.
        batch_size : int, optional
            Used only if `use_onnx` is True.
            Fixed batch size of the ONNX module. This is the only viable batch size for this model after loading.
        opset_version : int
            Used only if `use_onnx` is True.
            Version of export standard to use.
        pickle_module : module
            Module to use for pickling.
        ignore_attributes : str or iterable, optional
            List of attributes to ignore when pickling (e.g. 'optimizer')
        kwargs : dict
            Other keyword arguments, passed directly to :func:`torch.save`.
        """
        pickle_module = dill if pickle_module is None else pickle_module

        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        # Unwrap DDP if needed
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module

        if isinstance(ignore_attributes, str):
            ignore_attributes = [ignore_attributes]
        elif ignore_attributes is None:
            ignore_attributes = []
        ignore_attributes = set(ignore_attributes)

        if fmt is None:
            self._save_torch(path, pickle_metadata, ignore_attributes, pickle_module, kwargs)
            return

        if fmt not in self.AVAILABLE_FORMATS:
            raise ValueError(f"fmt must be in {self.AVAILABLE_FORMATS} but got {fmt}!")

        if fmt == "onnx":
            self._save_onnx(path, pickle_metadata, batch_size, opset_version,
                            ignore_attributes, pickle_module, kwargs)
        elif fmt == "openvino":
            self._save_openvino(path, pickle_metadata, batch_size,
                                ignore_attributes, pickle_module, kwargs)
        elif fmt == "safetensors":
            self._save_safetensors(path, pickle_metadata,
                                ignore_attributes, pickle_module, kwargs)
        else:
            raise RuntimeError(f"Unsupported format: {fmt}")


    def _save_torch(self, path, pickle_metadata, ignore_attributes, pickle_module, kwargs):
        """ Save the model in PyTorch format. """
        if pickle_metadata:
            preserved = set(self.PRESERVE) - ignore_attributes
            saved_data = {item: getattr(self, item) for item in preserved}
            torch.save(saved_data, path, pickle_module=pickle_module, **kwargs)
        else:
            torch.save({'model': self.model}, path, pickle_module=pickle_module, **kwargs)


    def _save_onnx(self, path, pickle_metadata, batch_size, opset_version,
                ignore_attributes, pickle_module, kwargs):
        """ Save the model in ONNX format."""
        if batch_size is None:
            raise ValueError("`batch_size` must be specified when saving in ONNX format!")

        inputs = self.make_placeholder_data(batch_size=batch_size, unwrap=False)
        self.sanitize_module_names(self.model)

        if pickle_metadata:
            name, ext = os.path.splitext(path)
            if ext == ".onnx":
                raise ValueError("Path should not have .onnx extension when saving with metadata!")
            onnx_path = name + ".onnx"
        else:
            onnx_path = path

        torch.onnx.export(
            self.model.eval(),
            inputs,
            onnx_path,
            opset_version=opset_version,
        )

        if pickle_metadata:
            preserved = self.PRESERVE_ONNX - ignore_attributes
            meta = {
                "onnx": True,
                "path_onnx": onnx_path,
                "onnx_batch_size": batch_size,
                **{k: getattr(self, k) for k in preserved}
            }
            torch.save(meta, path, pickle_module=pickle_module, **kwargs)


    def _save_openvino(self, path, pickle_metadata, batch_size,
                    ignore_attributes, pickle_module, kwargs):
        """ Save the model in OpenVINO format. """
        import openvino as ov

        model = self.model.eval()
        if not isinstance(model, ov.Model):
            inputs = self.make_placeholder_data(batch_size=batch_size, unwrap=False)
            model = ov.convert_model(model, example_input=inputs)

        name, ext = os.path.splitext(path)
        if pickle_metadata:
            if ext == ".xml":
                raise ValueError("Path should not have .xml extension when saving with metadata!")
            openvino_path = name + ".xml"
        else:
            if ext != ".xml":
                raise ValueError("Path should have .xml extension when saving OpenVINO model!")
            openvino_path = path

        ov.save_model(model, output_model=openvino_path)

        if pickle_metadata:
            preserved = self.PRESERVE_OPENVINO - ignore_attributes
            meta = {
                "openvino": True,
                "path_openvino": openvino_path,
                **{k: getattr(self, k) for k in preserved}
            }
            torch.save(meta, path, pickle_module=pickle_module, **kwargs)


    def _save_safetensors(self, path, pickle_metadata,
                        ignore_attributes, pickle_module, kwargs):
        """ Save the model in Safetensors format."""
        from safetensors.torch import save_file

        if pickle_metadata:
            name, ext = os.path.splitext(path)
            if ext == ".safetensors":
                raise ValueError("Path should not have .safetensors extension when saving with metadata!")

            safetensors_path = os.path.splitext(path)[0] + ".safetensors"
        else:
            safetensors_path = path

        save_file(self.model.state_dict(), safetensors_path)

        if pickle_metadata:
            preserved = self.PRESERVE_SAFETENSORS - ignore_attributes
            meta = {
                "safetensors": True,
                "path_safetensors": safetensors_path,
                **{k: getattr(self, k) for k in preserved}
            }
            torch.save(meta, path, pickle_module=pickle_module, **kwargs)

    @classmethod
    def sanitize_module_names(cls, module):
        """
        Recursively rename submodules to ensure names are safe for ONNX export.
        Replaces spaces, quotes, and commas with underscores.
        """
        # Work on a list of keys to avoid mutating dict while iterating
        keys = list(module._modules.keys()) # noqa: SLF001
        for key in keys:
            child = module._modules[key] # noqa: SLF001
            clean_key = key.replace(' ', '_').replace('"', '').replace(',', '_')
            if clean_key != key:
                module._modules[clean_key] = module._modules.pop(key) # noqa: SLF001
            # Recurse
            cls.sanitize_module_names(child)

    def load(self, path, fmt=None, make_infrastructure=False, mode='eval', pickle_module=dill, **kwargs):
        """ Load a torch model from a file.

        If the model was saved in ONNX format (refer to :meth:`.save` for more info), we fix the microbatch size
        to the batch size of ONNX conversion. Moreover, we disable model ability to train.

        Parameters
        ----------
        path : str, PathLike, io.Bytes
            a file where a model is stored.
        fmt: optional str
            Weights format. Available formats: "pt", "onnx", "openvino", "safetensors"
        make_infrastructure : bool
            Whether to re-create model loss, optimizer, scaler and decay.
        mode : str
            Model mode.
        pickle_module : module
            Module to use for pickling.
        kwargs : dict
            Other keyword arguments, passed directly to :func:`torch.save`.
        """
        model_load_kwargs = kwargs.pop('model_load_kwargs', {})
        device = kwargs.pop('device', None)
        pickle_module = dill if pickle_module is None else pickle_module

        if device is not None:
            self.device = device

            if (self.device == 'cpu') or ((not isinstance(self.device, str)) and (self.device.type == 'cpu')):
                self.amp = False
        else:
            self._parse_devices()

        if isinstance(path, str):
            if fmt == "safetensors" or (fmt is None and path.endswith(".safetensors")):
                self._load_safetensors(path, make_infrastructure=make_infrastructure, mode=mode)
                return
            if fmt == "onnx" or (fmt is None and path.endswith(".onnx")):
                self._load_onnx(path, make_infrastructure=make_infrastructure, mode=mode)
                return
            if fmt == "openvino" or (fmt is None and path.endswith(".xml")):
                self._load_openvino(path, **model_load_kwargs)
                return

        kwargs['map_location'] = self.device

        # Load items from disk storage and set them as insance attributes
        checkpoint = torch.load(path, pickle_module=pickle_module, **kwargs)

        # `load_config` is a reference to `self.external_config` used to update `config`
        # It is required since `self.external_config` may be overwritten in the cycle below
        load_config = self.external_config

        for key, value in checkpoint.items():
            setattr(self, key, value)
        self.config = self.config + load_config

        if 'openvino' in checkpoint:
            # Load openvino model
            self._load_openvino(checkpoint['path_openvino'], **model_load_kwargs)
        elif 'onnx' in checkpoint:
            self._load_onnx(checkpoint['path_onnx'], microbatch_size=checkpoint['onnx_batch_size'],
                           **model_load_kwargs)
        elif "safetensors" in checkpoint:
            self._load_safetensors(checkpoint['path_safetensors'], make_infrastructure=make_infrastructure, mode=mode)

    def _load_onnx(self, file, make_infrastructure=False, mode='eval', microbatch_size=None):
        """Load a model from ONNX file."""
        try:
            from onnx2torch import convert
        except ImportError as e:
            raise ImportError('Loading model, stored in ONNX format, requires `onnx2torch` library.') from e

        model = convert(file).eval()
        self.model = model
        if microbatch_size:
            self.microbatch_size = microbatch_size

        self.model_to_device()

        if make_infrastructure:
            self.make_infrastructure()

        self.set_model_mode(mode)

    def _load_safetensors(self, file, make_infrastructure=False, mode='eval'):
        """Load a model from Safetensors file."""
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError('Loading model, stored in Safetensors format, requires `safetensors` library.') from e

        state_dict = load_file(file)

        inputs = self.make_placeholder_data(to_device=True)
        with torch.no_grad():
            self.model = Network(inputs=inputs, config=self.config, device=self.device)

        self.model.load_state_dict(state_dict)

        self.model_to_device()

        if make_infrastructure:
            self.make_infrastructure()

        self.set_model_mode(mode)

        return

    def _load_openvino(self, file, **model_load_kwargs):
        """Load a model from OpenVINO file."""
        model = OVModel(model_path=file, **model_load_kwargs)
        self.model = model

        self._loaded_from_openvino = True
        self.disable_training = True


    # Utilities to use when working with TorchModel
    @staticmethod
    def get_model_reference(obj=None):
        """ Get the instance of a `TorchModel`, if called inside :meth:`.train` or :meth:`.predict` contexts.
        A possible example of usage is to call inside loss module forward to get the reference of the model.
        """
        if hasattr(obj, 'model_reference') and obj.model_reference is not None:
            return obj.model_reference

        for frame in inspect.stack():
            if frame.function not in {'_train', '_predict'}:
                continue
            if 'self' not in frame.frame.f_locals:
                continue

            model_reference = frame.frame.f_locals['self']
            if isinstance(model_reference, TorchModel):
                return model_reference
        return None

    # Debug and profile the performance
    def set_requires_grad(self, requires_grad):
        """ Set `requires_grad` flag for the underlying Pytorch model.
        Helpful when training multiple chained models.
        """
        for p in self.model.parameters():
            p.requires_grad = requires_grad

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
            result = (self.profile_info.reset_index().groupby(['name']).agg(aggs, numeric_only=True)
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


    # Utilities for activations
    def get_activation_blocks(self, modules=None):
        """ Retrieve intermediate blocks of the neural network model.

        Parameters
        ----------
        modules : str or list of str or None, default None
            The modules of the model for retrieving activations from.
            If None, all of the parts from the `model.config['order']` will be used.

        Returns
        -------
        activation_blocks : list of str
            Intermediate activation blocks.

        Examples
        --------
        >>> model.get_activation_blocks(modules=['encoder', 'embedding', 'decoder'])
        >>> print(activation_blocks)
        ['model.encoder["block-0"]', 'model.encoder["block-1"]', 'model.embedding', 'model.decoder["block-0"]']
        """
        modules = [modules] if isinstance(modules, str) else modules or self.model.config['order']
        activation_blocks = []

        for module_name in modules:
            extracted_module = getattr(self.model, module_name)
            if isinstance(extracted_module, (EncoderModule, DecoderModule, MLPDecoderModule)):
                for block_name in extracted_module:
                    if 'block' in block_name:
                        activation_blocks.append(f'model.{module_name}["{block_name}"]')
            else:
                activation_blocks.append(f'model.{module_name}')

        return activation_blocks

    @staticmethod
    def reduce_channels(array, normalize=True, n_components=3):
        """ Convert multichannel array to low-dimenional array using PCA.

        Parameters
        ----------
        array : np.ndarray
            4D array of shape (B, C, H, W).
        normalize : bool, default True
            Apply 'minmax' normalization to the images or not.
        n_components : int, default 3
            Number of components for PCA.

        Returns
        -------
        compressed_array : np.ndarray
            Compressed array of shape (B, H, W, n_components).
        explained_variance_ratio : np.ndarray
            Percentage of variance explained by each of the selected components.
        """
        array = array.transpose(0, 2, 3, 1)
        pca_instance = PCA(n_components=n_components)
        compressed_array = pca_instance.fit_transform(array.reshape(-1, array.shape[-1]))
        compressed_array = compressed_array.reshape(*array.shape[:3], n_components)
        if normalize:
            normalizer = Normalizer(mode='minmax')
            compressed_array = normalizer.normalize(compressed_array, inplace=True)

        explained_variance_ratio = pca_instance.explained_variance_ratio_

        return compressed_array, explained_variance_ratio

class OVModel:
    """ Class-wrapper for openvino models to interact with them through :class:`~.TorchModel` interface.

    Note, openvino models are loaded on 'cpu' only.

    Parameters
    ----------
    model_path : str
        Path to compiled openvino model.
    core_config : tuple or dict, optional
        Openvino core properties.
        If you want set properties globally provide them as tuple: `('CPU', {name: value})`.
        For local properties just provide `{name: value}` dict.
        For more, read the documentation:
        https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_query_api.html#setting-properties-globally
    compile_config : dict, optional
        Openvino model compilation config.
    """
    def __init__(self, model_path, core_config=None, compile_config=None):
        import openvino as ov

        core = ov.Core()

        if core_config is not None:
            if isinstance(core_config, tuple):
                core.set_property(core_config[0], core_config[1])
            else:
                core.set_property(core_config)

        self.model = core.read_model(model=model_path)

        if compile_config is None:
            compile_config = {}

        self.model = core.compile_model(self.model, 'CPU', config=compile_config)

    def eval(self):
        """ Placeholder for compatibility with :class:`~TorchModel` methods."""
        pass

    def __call__(self, input_tensor):
        """ Evaluate model on the provided data. """
        results = self.model(input_tensor)

        results = torch.from_numpy(results[self.model.output(0)])
        return results
