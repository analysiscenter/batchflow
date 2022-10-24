""" Contains mixin for :class:`~.torch.TorchModel` to provide textual and graphical visualizations. """
import sys
from ast import literal_eval
from pprint import pformat as _pformat

import numpy as np
import torch

from ...monitor import GPUMemoryMonitor
from ...notifier import Notifier
from ...plotter import plot
from ...decorators import deprecated

# Also imports `tensorboard`, if necessary


def pformat(object, indent=1, width=80, depth=None, *, compact=False, sort_dicts=True, underscore_numbers=False):
    """ Backwards compatible version of pformat. """
    # pylint: disable=unexpected-keyword-arg
    _ = underscore_numbers
    if sys.version_info.minor < 8:
        result = _pformat(object=object, indent=indent, width=width, depth=depth, compact=compact)
    else:
        result = _pformat(object=object, indent=indent, width=width, depth=depth, compact=compact,
                          sort_dicts=sort_dicts)
    return result



class VisualizationMixin:
    """ Collection of visualization (both textual and graphical) tools for a :class:`~.torch.TorchModel`. """
    # Textual visualization of the model
    def information(self, config=False, devices=True, misc=True, bold=True):
        """ Show information about model configuration, used devices, train steps and more. """
        print(self._information(config=config, devices=devices, misc=misc, bold=bold))

    def _information(self, config=False, devices=True, misc=True, bold=False):
        """ Create information string. """
        message = ''
        bold_code = '\033[1m\033[4m' if bold else ''
        endl_code = '\033[0m' if bold else ''

        template_header = '\n' + bold_code + '{}:' + endl_code + '\n'

        if config:
            message += template_header.format('Config')
            message += pformat(self.config.config, sort_dicts=False) + '\n'

        if devices:
            message += template_header.format('Devices')
            message += f'Leading device is {self.device}\n'
            if self.devices:
                message += '\n'.join([f'Device {i} is {d}' for i, d in enumerate(self.devices)]) + '\n'

        if misc:
            message += template_header.format('Shapes')

            if self.inputs_shapes:
                message += '\n'.join([f'Input  {i}: {s}' for i, s in enumerate(self.inputs_shapes)]) + '\n'
            if self.targets_shapes:
                message += '\n'.join([f'Target {i}: {s}' for i, s in enumerate(self.targets_shapes)]) + '\n'
            if self.classes:
                message += f'Number of classes: {self.classes}\n'

            message += template_header.format('Model info')
            if self.model:
                message += f'Order  of model parts: {self.model.config["order"]}\n'
                message += f'Number of all model parameters in the model: {self.num_parameters:,}\n'

                if self.model.config["order"] != self.model.config["trainable"]:
                    message += f'Trainable model parts: {self.model.config["trainable"]}\n'
                    message += f'Number of trainable parameters in the model: {self.num_trainable_parameters:,}\n'


            message += f'\nTotal number of passed training iterations: {self.iteration}\n'

            if self.last_train_info:
                message += template_header.format('Last train iteration info')
                message += pformat(self.last_train_info, sort_dicts=False) + '\n'

            if self.last_predict_info:
                message += template_header.format('Last predict iteration info')
                message += pformat(self.last_predict_info, sort_dicts=False) + '\n'
        return message[1:-1]


    def repr(self, verbosity=1, collapsible=True, show_num_parameters=False, extra=False,):
        """ Show simplified model layout. """
        self.model.repr(verbosity=verbosity, collapsible=collapsible,
                        show_num_parameters=show_num_parameters, extra=extra)

    def prepare_repr(self, verbosity=1, collapsible=True, show_num_parameters=False, extra=False):
        return self.model.prepare_repr(verbosity=verbosity, collapsible=collapsible,
                                       show_num_parameters=show_num_parameters, extra=extra)


    @property
    def num_frozen_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    # Graphs to describe model
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


    def plot_lr(self, **kwargs):
        """ Plot graph of learning rate over iterations. """
        params = {
            'title': 'Learning rate',
            'xlabel': 'Iterations',
            'ylabel': r'$\lambda$',
            'ylabel_rotation': 0,
            'ylabel_size': 15,
            'legend': False,
            'grid': 'major',
            **kwargs
        }

        data = [(None, lr) for lr in np.array(self.lr_list).T]

        return plot(data=data, mode='loss', **params)

    def plot_loss(self, overlay_lr=True, start_iteration=0, frequency=50, **kwargs):
        """ Plot loss and learning rate over the same figure.

        Parameters
        ----------
        overlay_lr : bool
            Whether show learning rate on the same plot with loss or not.
        start_iteration : int
            Starting iteration for display.
        frequency : int
            Tick frequency. Used only if `start_iteration` is not zero.
        kwargs : misc
            For `plot` in `mode='loss'`:

            figsize : tuple of ints
                Size of the figure.
            window : int or None
                If int, then averaged graph of loss values is displayed over the regular one.
                The average for each point is computed as the mean value of the kernel with the center in this point.
                Around the edges of loss graph (in the beginning and at the end) we use the nearest value to pad.
                If None, no additional graphs are displayed.
            final_window : int or None
                If int, then we additionally display the mean value of the last `final_window` iterations in the legend.
                If None, no additional info is displayed.
            minor_grid_frequency : number or tuple of two numbers
                If a single number, defines grid frequency for both subplot axes.
                If a tuple of two numbers, they define grid frequencies for x-axis and y-axis correspondingly.
            log_loss, log_lr : bool
                Whether to take the log of respective graph values.
            return figure : bool
                Whether to return the figure.
            savepath : str or None
                If str, then path to save the figure.
                If None, the figure is not saved to disk.
            save_kwargs : dict or None
                If dict, then additional parameters for figure saving.
        """
        slc = slice(start_iteration, None)
        locations = np.arange(0, self.iteration - start_iteration, frequency)

        if overlay_lr:
            data = (self.loss_list[slc],
                    [l[0] for l in self.lr_list][slc])
        else:
            data = (self.loss_list[slc], None)

        kwargs['title'] = 'Loss values and learning rate' if overlay_lr else 'Loss values'
        if start_iteration:
            kwargs['xtick_locations'] = locations
            kwargs['xtick_labels'] = locations + start_iteration
        if 'final_window' in kwargs:
            kwargs['final_window'] = min(kwargs['final_window'], self.iteration)

        return plot(data=data, mode='loss', **kwargs)

    # Deprecated aliases
    deprecation_msg = "`{}` is deprecated and will be removed in future versions, use `{}` instead."
    show_lr = deprecated(deprecation_msg.format('TorchModel.show_lr', 'TorchModel.plot_lr'))(plot_lr)
    show_loss = deprecated(deprecation_msg.format('TorchModel.show_loss', 'TorchModel.plot_loss'))(plot_loss)


class OptimalBatchSizeMixin:
    """ Compute optimal batch size for training/inference to maximize GPU memory usage.
    Works by using `train`/`predict` with different batch sizes, and measuring how much memory is taken.
    Then, we solve the system of `measured_memory = batch_size * item_size + model_size + eps` equations for both
    `item_size` and `model_size`.

    For stable measurements, we make `n` iterations of `train`/`predict`, until the memory consumption stabilizes.
    """
    def compute_optimal_batch_size(self, method='train', max_memory=90, inputs=None, targets=None, pbar='n',
                                   start_batch_size=4, delta_batch_size=4, max_batch_size=128, max_iters=16,
                                   n=20, frequency=0.05, time_threshold=3, tail_size=20, std_threshold=0.1):
        """ Compute memory usage for multiple batch sizes. """
        #pylint: disable=consider-iterating-dictionary
        table = {}
        batch_size = start_batch_size
        for _ in Notifier(pbar)(range(max_iters)):
            info = self.get_memory_utilization(batch_size, method=method,
                                               inputs=inputs, targets=targets, n=n, frequency=frequency,
                                               time_threshold=time_threshold,
                                               tail_size=tail_size, std_threshold=std_threshold)
            table[batch_size] = info

            # Exit condition
            batch_size += delta_batch_size
            if info['memory'] > max_memory or batch_size > max_batch_size :
                break

        # Make and solve a system of equations for `item_size`, `model_size`
        matrix = np.array([[batch_size, 1] for batch_size in table.keys()])
        vector = np.array([value['memory'] for value in table.values()])
        item_size, model_size = np.dot(np.linalg.pinv(matrix), vector)

        # Compute the `batch_size` to use up to `max_memory`
        optimal_batch_size = (max_memory - model_size) / item_size
        optimal_batch_size = int(optimal_batch_size)

        return {'batch_size': optimal_batch_size,
                'item_size': item_size,
                'model_size': model_size,
                'table': table}

    def get_memory_utilization(self, batch_size, method='train', inputs=None, targets=None, n=20, frequency=0.05,
                               time_threshold=3, tail_size=20, std_threshold=0.1):
        """ For a given `batch_size`, make `inputs` and `targets` and compute memory utilization. """
        inputs = inputs or self.make_placeholder_data(batch_size)
        inputs = list(inputs) if isinstance(inputs, (tuple, list)) else [inputs]
        inputs = [item[:batch_size] for item in inputs]

        targets = targets or self.predict(inputs=inputs, outputs='predictions')
        targets = list(targets) if isinstance(targets, (tuple, list)) else [targets]
        targets = [item[:batch_size] for item in targets]

        # Clear the GPU from potential previous runs
        torch.cuda.empty_cache()
        return self._get_memory_utilization(method=method, inputs=inputs, targets=targets, n=n, frequency=frequency,
                                            time_threshold=time_threshold,
                                            tail_size=tail_size, std_threshold=std_threshold)

    def _get_memory_utilization(self, method, inputs, targets, n, frequency,
                                time_threshold, tail_size, std_threshold):
        """ Run method `n` times and make sure that memory measurements are stable. """
        with GPUMemoryMonitor(frequency=frequency) as monitor:
            for _ in range(n):
                if method == 'train':
                    _ = self.train(inputs=inputs, targets=targets, microbatch_size=False)
                elif method == 'predict':
                    _ = self.predict(inputs=inputs, microbatch_size=False)

        # Check if the measurement is stable. If so, return the value and confidence
        data = monitor.data
        time = len(data) * frequency # in seconds
        if time > time_threshold:
            tail = data[-tail_size:]
            if np.std(tail) < std_threshold:
                return {'memory': np.mean(tail), 'n': n, 'monitor': monitor}

        # If the measurement is not stable, run for twice as long
        return self._get_memory_utilization(method=method, inputs=inputs, targets=targets,
                                            n=2*n, frequency=frequency, time_threshold=time_threshold,
                                            tail_size=tail_size, std_threshold=std_threshold)



class LayerHook:
    """ Hook to get both activations and gradients for a layer. """
    # TODO: work out the list activations / gradients
    def __init__(self, module):
        self.activation = None
        self.gradient = None

        self.forward_handle = module.register_forward_hook(self.forward_hook)
        self.backward_handle = module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """ Save activations: if multi-output, the last one is saved. """
        _ = module, input
        if isinstance(output, (tuple, list)):
            output = output[-1]
        self.activation = output

    def backward_hook(self, module, grad_input, grad_output):
        """ Save gradients: if multi-output, the last one is saved. """
        _ = module, grad_input
        if isinstance(grad_output, (tuple, list)):
            grad_output = grad_output[-1]
        self.gradient = grad_output

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __del__(self):
        self.close()


class ExtractionMixin:
    """ Extract information about intermediate layers: activations, gradients and weights. """
    def is_layer_id(self, string):
        """ Check if the passed `string` is a layer id: a string, that can be used to access a layer in the model.
        For more about layer ids, refer to `:meth:~.get_layer` documentation.
        """
        try:
            _ = self.get_layer(string)
            return True
        except (AttributeError, LookupError):
            return False

    def get_layer(self, layer_id):
        """ Get layer instance by its layer id.

        The layer id describes how to access the layer through a series of `getattr` and `getitem` calls.
        For example, if the model has `batchflow_model.model.head[0]` layer, you can access it with::

        >>> batchflow_model.get_layer('model.head[0]')

        String keys for `getitem` calls are also allowed::

        >>> batchflow_model.get_layer('model.body.encoder["block-0"]')
        """
        layer_id = layer_id.replace('[', ':').replace(']', '')
        prefix, *attributes = layer_id.split('.')
        if prefix != 'model':
            raise AttributeError('Layer id should start with `model`. i.e. `model.head`!')

        result = self.model
        for attribute in attributes:
            attribute, *item_ids = attribute.split(':')

            result = getattr(result, attribute)
            for item_id in item_ids:
                result = result[literal_eval(item_id)]
        return result


    def get_intermediate_activations(self, inputs, layers=None):
        """ Compute the intermediate activations of a given layers in the same structure (tuple/list/dict).

        Under the hood, a forward hook is registered to capture the output of a targeted layers,
        and it is removed after extraction of all the activations.

        Parameters
        ----------
        inputs : np.array or sequence of them
            Inputs to the model.
        layers : nn.Module, sequence or dict
            If nn.Module, then must be a part of the `model` attribute to get activations from.
            If sequence, then multiple such modules.
            If dictionary, then values must be such modules.

        Returns
        -------
        Intermediate activations in the same structure, as `layers`.
        """
        #pylint: disable=unnecessary-comprehension
        if layers is None:
            raise TypeError('get_intermediate_activations() missing 1 required argument: `layers`')

        # Parse `activations`: make it a dictionary
        if not isinstance(layers, (tuple, list, dict)):
            container = {0: layers}
        elif isinstance(layers, (tuple, list)):
            container = {i: item for i, item in enumerate(layers)}
        else:
            container = dict(layers) # shallow copy is fine

        container = {key: LayerHook(module)
                     for key, module in container.items()}

        # Parse inputs to model, run model
        inputs = self.transfer_to_device(inputs)
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)

        # Remove hooks; parse hooked data into desired format
        for value in container.values():
            value.close()

        container = {key : extractor.activation.detach().cpu().numpy()
                     for key, extractor in container.items()}

        if isinstance(layers, (tuple, list)):
            container = [container[i] for i, _ in enumerate(layers)]
        elif not isinstance(layers, dict):
            container = container[0]
        return container

    def get_layer_representation(self, layer, index=0, input_shape=None, ranges=(-1, 1), iters=100, return_loss=False):
        """ Compute a representation of an intermediate layer.

        Under the hood, this function generates random image and then optimizes it
        with respect to the mean value of activations at the target layer.

        Parameters
        ----------
        layer : nn.Module
            Part of the model to visualize.
        index : int or slice
            Valid slice for the activations at the targeted layer. Default is 0.
        input_shape : sequence
            Shape of the image to generate. Default is the shape of the last model input.
        ranges : sequence
            Lower and upper bound of values in generated image.
        iters : int
            Number of optimization iterations.
        return_loss : bool
            Whether to return the loss values of optimization procedure.
        """
        # Create starting image: random uniform noise
        input_shape = input_shape or self.inputs_shapes[0][1:]
        image = np.random.uniform(*ranges, input_shape)[None]
        image_var = torch.from_numpy(image.astype(np.float32)).to(self.device)
        image_var.requires_grad = True

        # Set up optimization procedure
        extractor = LayerHook(layer)
        optimizer = torch.optim.Adam([image_var], lr=0.1, weight_decay=1e-6)
        self.model.eval()

        # Iteratively make image visualize desired layer/filter
        losses = []
        for _ in range(iters):
            optimizer.zero_grad()
            # Clone is needed due to bug in PyTorch v1.3. May be removed later
            self.model(image_var.clone())

            loss = - extractor.activation[0, index].mean()
            loss.backward()
            optimizer.step()

            image_var.data.clamp_(*ranges)
            losses.append(loss.detach())

        # Clean-up: one final clamp and closing handles
        image_var.data.clamp_(*ranges)
        image_var = image_var.detach().cpu().numpy()
        extractor.close()

        if return_loss:
            return image_var, losses
        return image_var

    def get_gradcam(self, inputs, targets=None,
                    layer=None, gradient_mode='onehot', cam_class=None):
        """ Create visual explanation of a network decisions, based on the intermediate layer gradients.
        Ramprasaath R. Selvaraju, et al "`Grad-CAM: Visual Explanations
        from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_"

        Under the hood, forward and backward hooks are used to extract the activation and the gradient,
        and the mean value of gradients along channels are used as weights for activation summation.

        Parameters
        ----------
        inputs : np.array or sequence of them
            Inputs to the model.
        layers : nn.Module, sequence or dict
            Part of the model to base visualizations on.
        gradient_mode : Tensor or str
            If Tensor, then used directly to backpropagate from.
            If `onehot`, then OHE is created with `cam_class` parameter.
            If `targets`, then targets argument is used.
            Otherwise, model prediction is used.
        cam_class : int
            If gradient mode is `onehot`, then class to visualize. Default is the model prediction.
        """
        extractor = LayerHook(layer)
        inputs = self.transfer_to_device(inputs)

        self.model.eval()
        prediction = self.model(inputs)

        if isinstance(gradient_mode, np.ndarray):
            gradient = self.transfer_to_device(gradient_mode)
        elif 'target' in gradient_mode:
            gradient = targets
        elif 'onehot' in gradient_mode:
            gradient = torch.zeros_like(prediction)[0:1]
            cam_class = cam_class or np.argmax(prediction.detach().cpu().numpy()[0])
            gradient[0][cam_class] = 1
        else:
            gradient = prediction.clone()

        self.model.zero_grad()
        prediction.backward(gradient=gradient, retain_graph=True)
        self.model.zero_grad()

        activation = extractor.activation.detach().cpu().numpy()[0]
        gradient = extractor.gradient.detach().cpu().numpy()[0]

        weights = np.mean(gradient, axis=(1, 2))
        camera = np.zeros(activation.shape[1:], dtype=activation.dtype)

        for i, w in enumerate(weights):
            camera += w * activation[i]

        camera = np.maximum(camera, 0)
        camera = (camera - np.min(camera)) / (np.max(camera) - np.min(camera) + 0.0001)
        camera = np.uint8(camera * 255)
        return camera

    # Visualize signal propagation statistics
    def get_signal_propagation(self, model=None, input_tensor=None):
        """ Compute signal propagation statistics of all layers in the network.
        Brock A. et al "`Characterizing signal propagation to close the performance gap in unnormalized ResNets
        <https://arxiv.org/pdf/2101.08692.pdf>`_"

        Under the hood, forward hooks are registered to capture outputs of all layers,
        and they are removed after extraction of all the activations.

        Parameters
        ----------
        model : nn.Module
            Model to base visualizations on.
        input_tensor : Tensor
            Input tensor for signal propagation.
        """
        model = model or self.model

        if input_tensor is None:
            input_shape = self.inputs_shapes[-1]
            input_tensor = torch.randn(input_shape, device=self.device)

        statistics = {
            'Average Channel Squared Mean': [],
            'Average Channel Variance': [],
            'Modules instances': []
        }
        extractors = []
        signals = []

        try:
            for module in model.modules():
                submodules_amount = sum(1 for _ in module.modules())
                module_instance = module.__class__.__name__
                if (submodules_amount == 1) and (module_instance != 'Identity'):
                    extractors.append(LayerHook(module))
                    statistics['Modules instances'].append(module_instance)
            _ = model(input_tensor)
        finally:
            for extractor in extractors:
                signals.append(extractor.activation)
                extractor.close()

        for tensor in signals:
            avg_ch_squared_mean = torch.mean(torch.mean(tensor, axis=1) ** 2).item()
            avg_ch_var =  torch.mean(torch.var(tensor, axis=1)).item()

            statistics['Average Channel Squared Mean'].append(avg_ch_squared_mean)
            statistics['Average Channel Variance'].append(avg_ch_var)

        return statistics

    def plot_signal_propagation(self, model=None, input_tensor=None,
                                statistics=('Average Channel Squared Mean', 'Average Channel Variance'), **kwargs):
        """ Visualize signal propagation plot.

        Parameters
        ----------
        model : nn.Module
            Model to base visualizations on.
        input_tensor : Tensor
            Input tensor for signal propagation.
        statistics : list or dict
            If list, must contain keys for dict returned by `ExtractionMixin.get_signal_propagation`.
            If dict, must map signal propagation statistics names to 1d arrays.
        kwargs : misc
            For `batchflow.plot`
        """
        if isinstance(statistics, tuple):
            names = list(statistics)
            statistics = self.get_signal_propagation(model=model, input_tensor=input_tensor)
            data = [statistics[name] for name in names]
        elif isinstance(statistics, dict):
            data = list(statistics.values())
            names = list(statistics.keys())

        plot_params = {
            'title': [f"{text} over network units" for text in names],
            'xlabel': 'Network depth',
            'ylabel': names,
            **kwargs
        }

        plot(data=data, mode='curve', combine='separate', **plot_params)
