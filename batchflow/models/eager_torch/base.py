""" Eager version of TorchModel. """
import os
import re
import threading
import inspect
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from .utils import unpack_fn_from_config
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



class EagerTorch:
    """ Eagerly! """

    def __init__(self, config=None):
        self.config = Config(config)
        self.full_config = None
        self.train_lock = threading.Lock()

        self.model = None
        self.device = None
        self.devices = []
        self.train_steps = None

        self.sync_counter = 0
        self.microbatch = None

        load = self.config.get('load')
        build = self.config.get('build', default=load is None)
        if load:
            self.load(**load)
        if build:
            self.build()

    def reset(self):
        pass


    def build(self):
        """ Build the model """
        self.full_config = self.combine_configs()
        self.build_config()

        self._get_devices()

        # If the inputs were set in config with their shapes we can build right away
        if self.full_config.get('inputs'):
            print('_BUILD IN BUILD')
            self._build()

    @classmethod
    def default_config(cls):
        """ Truly amazing docstring. """
        config = Config()
        config['inputs'] = {}
        config['placeholder_batch_size'] = 2

        config['device'] = None
        config['benchmark'] = True
        config['microbatch'] = None
        config['step_on_each'] = 1

        config['train_steps'] = None
        config['loss'] = None
        config['optimizer'] = ('Adam', dict())
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
        config = self.default_config() + self.config
        return config

    def build_config(self):
        """ Truly amazing docstring. """
        config = self.full_config

        if config.get('inputs'):
            inputs = config.get('initial_block/inputs')
            if isinstance(inputs, str):
                config['common/data_format'] = config['inputs'][inputs].get('data_format')

    def _get_devices(self):
        devices = self.full_config.get('device')
        if devices is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            devices = self.full_config.get('device')
            devices = devices if isinstance(devices, list) else [devices]
            available_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())] + ['cpu']

            for dev in devices:
                if isinstance(dev, torch.device):
                    self.devices.append(dev)
                elif isinstance(dev, str):
                    dev_ = dev.lower()
                    dev_ = dev.replace('gpu', 'cuda')

                    devices = [torch.device(device) for device in available_devices
                               if re.search(dev_, device.lower()) is not None]
                    self.devices.extend(devices)
                else:
                    raise TypeError('Wrong device type: {}'.format(type(dev)))
            self.devices = [device for i, device in enumerate(self.devices)
                            if device not in self.devices[:i]]
            self.device = self.devices[0]

        torch.backends.cudnn.benchmark = self.full_config.get('benchmark', 'cuda' in self.device.type)


    def _build(self, inputs=None):
        config = self.full_config
        order = config.get('order')

        inputs = inputs or self._placeholder_data()

        blocks = []
        for item in order:
            if isinstance(item, str):
                block_name = config_name = method = item
            elif isinstance(item, tuple):
                if len(item) == 2:
                    block_name, method = config_name, _ = item
                elif len(item) == 3:
                    block_name, config_name, method = item
            elif isinstance(item, dict):
                block_name = item['block_name']
                config_name = item.get('config_name', block_name)
                method = item.get('method', config_name)

            inputs = inputs[0] if isinstance(inputs, (tuple, list)) and len(inputs) == 1 else inputs
            block = self._make_block(config_name, method, config, inputs)
            if block is not None:
                inputs = block(inputs)
                blocks.append((block_name, block))

        self.model = nn.Sequential(OrderedDict(blocks))
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, self.devices)
        else:
            self.model.to(self.device)

        self.train_steps = self._make_train_steps(config)

    def _placeholder_data(self):
        config = self.full_config
        batch_size = config.get('placeholder_batch_size', 2)

        input_names = config.pop('initial_block/inputs', default=None)
        input_names = input_names if isinstance(input_names, (tuple, list)) else [input_names]
        shapes = []
        for name in input_names:
            cfg = config['inputs'][name]
            if 'shape' in cfg:
                shapes.append((batch_size, *cfg['shape']))
            elif 'classes' in cfg:
                shapes.append((batch_size, *cfg['classes']))
            else:
                raise ValueError('Input {} must contain `shape` configuration'.format(name))

        data = [np.zeros(shape, dtype=np.float32) for shape in shapes]
        data = self._fill_param(data)
        data = data[0] if len(data) == 1 else data
        return data

    def _make_block(self, name, method, config, inputs):
        config = {**config['common'], **config[name]}

        if 'module' in config:
            module = config['module']
            kwargs = config.get('module_kwargs')
            if 'inputs' in inspect.getfullargspec(module.__init__)[0]:
                kwargs = {**kwargs, **{'inputs': inputs}}
            block = module(*config.get('module_args', []), **kwargs)

        elif isinstance(config, dict):
            method = getattr(self, method) if isinstance(method, str) else method
            block = method(inputs, **config)
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

    def _make_loss(self, config, device=None):
        device = device or self.device
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

            loss_fn = loss_fn or loss(*args)
            if isinstance(loss_fn, nn.Module):
                loss_fn.to(device=device)
            losses.append(loss_fn)
        return losses

    def _make_optimizer(self, config):
        optimizer, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer is None or callable(optimizer) or isinstance(optimizer, type):
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
            raise ValueError('Missing required key ```n_iters``` in the cofiguration dict.')

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
        _config = config.get(name)
        kwargs = kwargs or {}
        config = {**config['common'], **_config, **kwargs}
        return config

    @classmethod
    def initial_block(cls, inputs, **kwargs):
        """ Truly amazing docstring. """
        kwargs = cls.get_defaults('initial_block', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def body(cls, inputs, **kwargs):
        """ Truly amazing docstring. """
        kwargs = cls.get_defaults('body', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None

    @classmethod
    def head(cls, inputs, **kwargs):
        """ Truly amazing docstring. """
        kwargs = cls.get_defaults('head', kwargs)
        if kwargs.get('layout'):
            return ConvBlock(inputs=inputs, **kwargs)
        return None


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

    def _fill_input(self, *args):
        return tuple([self._fill_param(arg) for arg in args])

    def _fill_output(self, fetches, outputs):
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


    def train(self, *args, fetches=None, use_lock=False, train_mode='',
              sum_grads=True, step_on_each=True, microbatch=False):
        """ Truly amazing docstring. """
        # pylint: disable=arguments-differ
        config = self.full_config
        *inputs, targets = self._fill_input(*args)

        if step_on_each is True:
            step_on_each = config.get('step_on_each', 1)
        elif step_on_each is False or step_on_each is None:
            step_on_each = 1

        if microbatch is True:
            microbatch = config.get('microbatch', len(targets))

        train_mode = train_mode if isinstance(train_mode, (tuple, list)) else [train_mode]

        steps = len(targets) // microbatch if microbatch else 1 # microbatch acts as size
        splitted_inputs = [np.array_split(item, steps) for item in inputs] if microbatch else [inputs]
        splitted_targets = np.array_split(targets, steps) if microbatch else [targets]

        if self.model is None:
            print('_BUILD IN TRAIN')
            self._build(splitted_inputs[0])
        self.model.train()

        if use_lock:
            self.train_lock.acquire()

        outputs = []
        for i in range(steps):
            _inputs = [item[i] for item in splitted_inputs]
            _targets = splitted_targets[i]

            output = self._train(*_inputs, _targets, fetches=fetches, train_mode=train_mode,
                                 sum_grads=sum_grads, step_on_each=step_on_each*steps)

            outputs.append(output)

        if use_lock:
            self.train_lock.release()

        outputs = [outputs] if isinstance(fetches, str) else outputs
        output = []
        for i in range(len(outputs)):
            lst = [item[i] for item in outputs]
            output.append(np.concatenate(lst, axis=0) if lst[0].size != 1 else np.mean(lst))
        output = output[0] if isinstance(fetches, str) else output
        return output

    def _train(self, *args, fetches=None, train_mode='', sum_grads=True, step_on_each=True):
        *inputs, targets = args

        output_container = {}

        if not sum_grads:
            predictions = self.model(*inputs)

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

                if sum_grads:
                    predictions = self.model(*inputs)
                loss = sum([loss_fn_(predictions, targets) for loss_fn_ in loss_fn]) / len(loss_fn)
                mode_loss += loss
                loss.backward()

                if self.sync_counter >= step_on_each:
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
                                         ops=config['output'], **config['common'])
        output_container = {**output_container, **additional_outputs}
        output = self._fill_output(fetches, output_container)
        return output

    def predict(self, *args, targets=None, train_mode='', fetches=None):    # pylint: disable=arguments-differ
        """ Truly amazing docstring. """
        inputs = self._fill_input(*args)
        if targets is not None:
            targets = self._fill_input(targets)[0]

        self.model.eval()
        train_mode = train_mode if isinstance(train_mode, (tuple, list)) else [train_mode]

        with torch.no_grad():
            output_container = {}
            predictions = self.model(*inputs)

            if targets is not None:
                for mode in train_mode:
                    if mode in self.train_steps.keys():
                        train_fetches = [(mode, self.train_steps[mode])]
                    else:
                        train_fetches = [(name, train_step) for name, train_step in self.train_steps.items()
                                         if re.search(mode, name) is not None]

                    mode_loss = 0
                    for name, step in train_fetches:
                        loss_fn = step['loss']
                        loss = sum([loss(predictions, targets) for loss in loss_fn]) / len(loss_fn)
                        mode_loss += loss
                        output_container['loss' + '_'*int(len(name) > 0) + name] = loss
                    output_container['loss' + '_'*int(len(mode) > 0) + mode] = mode_loss
            output_container['predictions'] = predictions

        config = self.full_config
        additional_outputs = self.output(inputs=predictions, predictions=config['predictions'],
                                         ops=config['output'], **config['common'])
        output_container = {**output_container, **additional_outputs}
        output = self._fill_output(fetches, output_container)
        return output

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Truly amazing docstring. """
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
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            prefix = [*ops.keys()][i]
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '', **kwargs)
            for oper in ops[prefix]:
                name, output = self._add_output_op(tensor, oper, oper, attr_prefix, **kwargs)
                outputs[name] = output
        return outputs


    def _add_output_op(self, inputs, oper, name, attr_prefix, **kwargs):
        if oper is None:
            output = inputs
        elif oper == 'softplus':
            output = torch.nn.functional.softplus(inputs)
        elif oper == 'sigmoid':
            output = torch.nn.functional.sigmoid(inputs)
        elif oper == 'proba':
            axis = self.channels_axis(kwargs.get('data_format'))
            output = torch.nn.functional.softmax(inputs, dim=axis)
        elif oper == 'labels':
            class_axis = self.channels_axis(kwargs.get('data_format'))
            output = inputs.argmax(dim=class_axis)
        elif callable(oper):
            output = oper(inputs)
            name = name or oper.__name__
        return attr_prefix + name, output


    @classmethod
    def channels_axis(cls, data_format='channels_first'):
        """ Get channel axis. """
        data_format = data_format if data_format else 'channels_first'
        return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1


    def save(self, path, *args, **kwargs):
        """ Truly amazing docstring. """
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
        """ Truly amazing docstring. """
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
