""""""
import os
import re
import threading
import inspect
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .layers import ConvBlock
from .losses import CrossEntropyLoss
from ..utils import unpack_fn_from_config
from ... import Config


# TODO:
# layers/ConvBlock
# microbatch (rename/alias to virtual batch?)
# multi-device
# async
# 



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


class EagerTorch:
    """ Eagerly! """

    def __init__(self, config=None, *args, **kwargs):
        self.config = Config(config)
        self.full_config = None
        self.train_lock = threading.Lock()

        self.model = None
        self.device = None
        self.microbatch = None

        self.n_iters = None
        self.current_iter = 0

        self.loss = None
        self.loss_fn = None
        self.lr_decay = None
        self.optimizer = None

        self.predictions = None


        load = self.config.get('load')
        build = self.config.get('build', default=load is None)
        if load:
            self.load(**load)
        if build:
            self.build(*args, **kwargs)

    def reset(self,):
        pass


    def build(self,):
        """ Build the model """
        config = self.build_config()
        self.full_config = config

        self.device = self._get_device()

        # If the inputs were set in config with their shapes we can build rightaway
        if config.get('inputs'):
            print('_BUILD IN BUILD')
            self._build()

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


    @classmethod
    def default_config(cls):
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

    def build_config(self):
        config = self.default_config()
        config = config + self.config

        if config.get('inputs'):
            inputs = config.get('initial_block/inputs')
            if isinstance(inputs, str):
                config['common/data_format'] = config['inputs'][inputs].get('data_format')
        return config


    def _build(self, inputs=None):
        config = self.full_config

        inputs = inputs or self._placeholder_data()

        blocks = []
        for loc in ['initial_block', 'body', 'head']:
            inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs
            block = self._make_block(loc, config, inputs)
            if block is not None:
                inputs = block(inputs)
                blocks.append(block)

        self.model = nn.Sequential(*blocks)

        if self.loss_fn is None:
            self._make_loss(config)
        if self.optimizer is None:
            self._make_optimizer(config)

    def _placeholder_data(self, batch_size=2):
        config = self.full_config

        input_names = config.pop('initial_block/inputs', default=None) or list(config.get('inputs').keys())
        input_names = input_names if isinstance(input_names, (tuple, list)) else [input_names]
        shapes = [(batch_size, *config['inputs'][name]['shape']) for name in input_names]

        data = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        data = self._fill_param(data)
        data = data[0] if len(data) == 1 else data
        return data

    def _make_block(self, name, config, inputs):
        config = {**config['common'], **config[name]}

        if 'module' in config:
            module = config['module']
            kwargs = config.get('module_kwargs')
            if 'inputs' in inspect.getfullargspec(module.__init__)[0]:
                kwargs = {**kwargs, **{'inputs': inputs}}
            block = module(*config.get('module_args', []), **kwargs)

        elif isinstance(config, dict):
            block = getattr(self, name)(inputs, **config)
        else:
            raise ValueError('Bad')
        return block


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


    @classmethod
    def get_defaults(cls, name, kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = config.get(name)
        kwargs = kwargs or {}
        config = {**config['common'], **_config, **kwargs}
        return config

    @classmethod
    def initial_block(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('initial_block', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def body(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('body', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def head(cls, inputs, **kwargs):
        kwargs = cls.get_defaults('head', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None


    def _fill_value(self, inputs):
        inputs = torch.from_numpy(inputs)
        if self.device:
            inputs = inputs.to(self.device)
        return inputs

    def _fill_param(self, inputs):
        if inputs is None:
            pass
        elif isinstance(inputs, (tuple, list)):
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
        config = self.full_config
        *inputs, targets = self._fill_input(*args)

        if self.model is None:
            print('_BUILD IN TRAIN')
            self._build(inputs)


        if use_lock:
            self.train_lock.acquire()
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
            self.train_lock.release()

        config = self.full_config
        self.output(inputs=self.predictions, predictions=config['predictions'],
                    ops=config['output'], **config['common'])
        output = self._fill_output(fetches)
        return output

    def predict(self, *args, targets=None, fetches=None):    # pylint: disable=arguments-differ
        inputs = self._fill_input(*args)
        if targets is not None:
            targets = self._fill_input(targets)[0]

        self.model.eval()

        with torch.no_grad():
            self.predictions = self.model(*inputs)
            if targets is None:
                self.loss = None
            else:
                self.loss = self.loss_fn(self.predictions, targets)

        config = self.full_config
        self.output(inputs=self.predictions, predictions=config['predictions'],
                    ops=config['output'], **config['common'])
        output = self._fill_output(fetches)
        return output

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
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
        _ = args, kwargs
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save({
            'model_state_dict': self.model,
            'optimizer_state_dict': self.optimizer,
            'loss': self.loss_fn,
            'config': self.config,
            'full_config': self.full_config
            }, path)

    def load(self, path, *args, eval=False, **kwargs):
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
        self.full_config = checkpoint['full_config']

        self.device = device

        if device:
            self.model.to(device)

        if eval:
            self.model.eval()
