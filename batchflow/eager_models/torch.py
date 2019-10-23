""""""
import os
import re
import threading
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .. import Config
# from .. import BaseModel
# from ..utils import unpack_fn_from_config
# from .layers import ConvBlock
# from .losses import CrossEntropyLoss



class CrossEntropyLoss(nn.CrossEntropyLoss):
    """ Custom loss which casts target dtype if needed """
    def forward(self, input, target):
        target = target.to(dtype=torch.long)
        return super().forward(input, target)

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


def unpack_args(args, layer_no, layers_max):
    """ Return layer parameters """
    new_args = {}
    for arg in args:
        if isinstance(args[arg], list) and layers_max > 1:
            if len(args[arg]) >= layers_max:
                arg_value = args[arg][layer_no]
            else:
                arg_value = args[arg]
        else:
            arg_value = args[arg]
        new_args.update({arg: arg_value})
    return new_args


def unpack_fn_from_config(param, config=None):
    """ Return params from config """
    par = config.get(param)

    if par is None:
        return None, {}

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
        par = par.copy()
        par_name, par_args = par.pop('name', None), par
    else:
        par_name, par_args = par, {}

    return par_name, par_args




def get_shape(inputs, shape=None):
    """ Return inputs shape """
    if inputs is None:
        pass
    elif isinstance(inputs, np.ndarray):
        shape = inputs.shape
    elif isinstance(inputs, torch.Tensor):
        shape = tuple(inputs.shape)
    elif isinstance(inputs, (torch.Size, tuple, list)):
        shape = tuple(inputs)
    elif isinstance(inputs, torch.nn.Module):
        shape = get_output_shape(inputs, shape)
    else:
        raise TypeError('inputs can be array, tensor, tuple/list or layer', type(inputs))
    return shape



class Dense(nn.Module):
    """ A dense layer """
    def __init__(self, units=None, out_features=None, bias=True, inputs=None):
        super().__init__()

        units = units or out_features

        shape = get_shape(inputs)
        self.linear = nn.Linear(np.prod(shape[1:]), units, bias)

    def forward(self, x):
        """ Make forward pass """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)



class ConvBlock(nn.Module):
    def __init__(self, inputs, layout='', filters=None):
        super().__init__()
        self.layout = layout
        self.filters = filters

        layers = []
        for i, letter in enumerate(layout):
            if letter == 'c':
                block = nn.Conv2d(inputs.shape[1], filters[i], 3)
                inputs = block(inputs)
                layers.append(block)

        # linear = nn.Linear(np.prod(get_shape(inputs)[1:]), 10, True)
        linear = Dense(units=10, inputs=inputs)
        layers.append(linear)

        self.block = nn.Sequential(*layers)



    def forward(self, inputs):
        return self.block(inputs)




class EagerTorch:
    """ With eager. """

    def __init__(self, config=None, *args, **kwargs):

        self.config = Config(config)
        self._train_lock = threading.Lock()
        self.n_iters = None
        self.current_iter = 0
        self.device = None
        self.loss_fn = None
        self.lr_decay = None
        self.optimizer = None
        self.model = None
        self._inputs = dict()
        self.predictions = None
        self.loss = None
        self.microbatch = None
        self._full_config = None


        load = self.config.get('load')
        build = self.config.get('build', default=load is None)
        if not isinstance(build, bool) and build in [1, 'first']:
            self.build(*args, **kwargs)
            build = False
        if load:
            self.load(**load)
        if build:
            self.build(*args, **kwargs)



    def _get_device(self):
        device = self.config.get('device')
        if isinstance(device, torch.device) or device is None:
            _device = device
        elif isinstance(device, str):
            _device = device.split(':')
            unit, index = _device if len(_device) > 1 else (device, '0')
            if unit.lower() == 'gpu':
                _device = torch.device('cuda', int(index))
            elif unit.lower() == 'cpu':
                _device = torch.device('cpu')
            else:
                raise ValueError('Unknown device type: ', device)
        else:
            raise TypeError('Wrong device type: ', type(device))
        return _device


    def reset(self,):
        pass

    def build(self,):
        """ Build the model """
        config = self.build_config()
        self._full_config = config

        self.device = self._get_device()
        if config.get('inputs'):
            print('_BUILD IN BUILD')
            self._build()

        self.microbatch = config.get('microbatch', None)


    def _build(self, inputs=None):
        inputs = inputs or self._placeholder_data()
        self.model = ConvBlock(*inputs, 'cc', [16, 35])

        config = self._full_config
        if self.loss_fn is None:
            self._make_loss(config)
        if self.optimizer is None:
            self._make_optimizer(config)


    def _placeholder_data(self):
        config = self._full_config
        shape = config['inputs'][config.get('initial_block/inputs')]['shape']
        shape = (2, *shape)

        data = np.zeros(shape, dtype=np.float32)
        data = torch.from_numpy(data)
        if self.device:
            data = data.to(self.device)
        return [data]


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
        config['common'] = {}
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['predictions'] = None
        config['output'] = None
        config['optimizer'] = ('Adam', dict())
        config['microbatch'] = None

        return config



    def build_config(self, names=None):
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
                config['head/num_classes'] = self.num_classes('targets')
                return config
        """
        config = self.default_config()
        config = config + self.config

        if config.get('inputs'):
            # self._make_inputs(names, config)
            inputs = config.get('initial_block/inputs')
            if isinstance(inputs, str):
                config['common/data_format'] = config['inputs'][inputs].get('data_format')

        return config




    def _make_loss(self, config):
        loss, args = unpack_fn_from_config('loss', config)

        if isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        elif isinstance(loss, str) and hasattr(nn, loss):
            loss = getattr(nn, loss)
        elif isinstance(loss, str) and hasattr(nn, loss + "Loss"):
            loss = getattr(nn, loss + "Loss")
        elif isinstance(loss, type):
            pass
        elif isinstance(loss, nn.Module):
            self.loss_fn = loss
        elif callable(loss):
            self.loss_fn = partial(loss, **args)
        else:
            raise ValueError("Loss is not defined in the model %s" % self.__class__.__name__)

        if self.loss_fn is None:
            self.loss_fn = loss(**args)

        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(device=self.device)

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name) or isinstance(optimizer_name, type):
            pass
        elif isinstance(optimizer_name, str) and hasattr(torch.optim, optimizer_name):
            optimizer_name = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        if optimizer_name:
            self.optimizer = optimizer_name(self.model.parameters(), **optimizer_args)
        else:
            raise ValueError("Optimizer is not defined", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            self.lr_decay = decay_name(self.optimizer, **decay_args)

    def _make_decay(self, config):
        decay_name, decay_args = unpack_fn_from_config('decay', config)

        if decay_name is None:
            return decay_name, decay_args
        if 'n_iters' not in config:
            raise ValueError('Missing required key ```n_iters``` in the cofiguration dict.')
        self.n_iters = config.pop('n_iters')

        if callable(decay_name) or isinstance(decay_name, type):
            pass
        elif isinstance(decay_name, str) and hasattr(torch.optim.lr_scheduler, decay_name):
            decay_name = getattr(torch.optim.lr_scheduler, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(decay_name)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        if decay_name in DECAYS_DEFAULTS:
            decay_dict = DECAYS_DEFAULTS.get(decay_name).copy()
            if decay_name == DECAYS['cos']:
                decay_dict.update(T_max=self.n_iters)
            decay_dict.update(decay_args)
            decay_args = decay_dict.copy()
        return decay_name, decay_args




    def _fill_value(self, inputs):
        inputs = torch.from_numpy(inputs)
        if self.device:
            inputs = inputs.to(self.device)
        return inputs

    def _fill_param(self, inputs):
        if inputs is None:
            pass
        elif isinstance(inputs, tuple):
            inputs_list = []
            for i in inputs:
                v = self._fill_value(i)
                inputs_list.append(v)
            inputs = inputs_list
        else:
            inputs = self._fill_value(inputs)
        return inputs

    def _fill_input(self, *args):
        inputs = []
        for arg in args:
            inputs.append(self._fill_param(arg))
        return tuple(inputs)

    def _fill_output(self, fetches):
        _fetches = [fetches] if isinstance(fetches, str) else fetches

        output = []
        for f in _fetches:
            if hasattr(self, f):
                v = getattr(self, f)
                if isinstance(v, (torch.Tensor, torch.autograd.Variable)):
                    v = v.detach().cpu().numpy()
                output.append(v)
            else:
                raise KeyError('Unknown value to fetch', f)

        output = output[0] if isinstance(fetches, str) else type(fetches)(output)

        return output




    def train(self, *args, fetches=None, use_lock=False, microbatch=None):    # pylint: disable=arguments-differ
        """ Train the model with the data provided

        Parameters
        ----------
        args
            Arguments to be passed directly into the model.

        fetches : tuple, list
            Sequence of tensor names to calculate and return.

        use_lock : bool
            If True, the whole train step is locked, thus allowing for multithreading.

        microbatch : int or None
            Size of chunks to split every batch into. Allows to process given data sequentially, accumulating gradients
            from microbatches and applying them once in the end.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same order.

        Examples
        --------
        .. code-block:: python

            model.train(B('images'), B('labels'), fetches='loss')
        """
        config = self._full_config
        *inputs, targets = self._fill_input(*args)
        if self.model is None:
            print('_BUILD IN TRAIN')
            self._build(inputs)




        if use_lock:
            self._train_lock.acquire()
        self.model.train()

        self.optimizer.zero_grad()
        self.predictions = self.model(*inputs)
        self.loss = self.loss_fn(self.predictions, targets)
        self.loss.backward()
        self.optimizer.step()

        if self.lr_decay:
            if self.current_iter == self.n_iters:
                self.lr_decay.step()
                self.current_iter = 0
            self.current_iter += 1

        if use_lock:
            self._train_lock.release()

        config = self._full_config
        self.output(inputs=self.predictions, predictions=config['predictions'],
                    ops=config['output'], **config['common'])
        output = self._fill_output(fetches)

        return output




    def predict(self, ):
        pass

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Add output operations to the model, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : torch.Tensor or a sequence of torch.Tensors
            input tensors

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
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            prefix = [*ops.keys()][i]
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '', **kwargs)
            for oper in ops[prefix]:
                self._add_output_op(tensor, oper, oper, attr_prefix, **kwargs)

    def _add_output_op(self, inputs, oper, name, attr_prefix, **kwargs):
        if oper is None:
            self._add_output_identity(inputs, name, attr_prefix, **kwargs)
        elif oper == 'softplus':
            self._add_output_softplus(inputs, name, attr_prefix, **kwargs)
        elif oper == 'sigmoid':
            self._add_output_sigmoid(inputs, name, attr_prefix, **kwargs)
        elif oper == 'proba':
            self._add_output_proba(inputs, name, attr_prefix, **kwargs)
        elif oper == 'labels':
            self._add_output_labels(inputs, name, attr_prefix, **kwargs)
        elif callable(oper):
            self._add_output_callable(inputs, oper, None, attr_prefix, **kwargs)

    def _add_output_identity(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        setattr(self, attr_prefix + name, inputs)
        return inputs

    def _add_output_softplus(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        proba = torch.nn.functional.softplus(inputs)
        setattr(self, attr_prefix + name, proba)

    def _add_output_sigmoid(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        proba = torch.nn.functional.sigmoid(inputs)
        setattr(self, attr_prefix + name, proba)

    def _add_output_proba(self, inputs, name, attr_prefix, **kwargs):
        axis = self.channels_axis(kwargs.get('data_format'))
        proba = torch.nn.functional.softmax(inputs, dim=axis)
        setattr(self, attr_prefix + name, proba)

    def _add_output_labels(self, inputs, name, attr_prefix, **kwargs):
        class_axis = self.channels_axis(kwargs.get('data_format'))
        predicted_classes = inputs.argmax(dim=class_axis)
        setattr(self, attr_prefix + name, predicted_classes)

    def _add_output_callable(self, inputs, oper, name, attr_prefix, **kwargs):
        _ = kwargs
        x = oper(inputs)
        name = name or oper.__name__
        setattr(self, attr_prefix + name, x)
        return x

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
        _ = args, kwargs
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save({
            'model_state_dict': self.model,
            'optimizer_state_dict': self.optimizer,
            'loss': self.loss_fn,
            'config': self.config,
            'full_config': self._full_config
            }, path)

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
        _ = args, kwargs
        device = self._get_device()
        if device:
            checkpoint = torch.load(path, map_location=device)
        else:
            checkpoint = torch.load(path)
        self.model = checkpoint['model_state_dict']
        self.optimizer = checkpoint['optimizer_state_dict']
        self.loss_fn = checkpoint['loss']
        self.config = self.config + checkpoint['config']
        self._full_config = checkpoint['full_config']

        self.device = device

        if device:
            self.model.to(device)

        if eval:
            self.model.eval()