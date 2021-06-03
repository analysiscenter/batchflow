""" Contains mixin for :class:`~.torch.TorchModel` to provide textual and graphical visualizations. """
from pprint import pformat
import matplotlib.pyplot as plt

import numpy as np
import torch

# Also imports `tensorboard`, if necessary



class VisualizationMixin:
    """ Collection of visualization (both textual and graphical) tools for a :class:`~.torch.TorchModel`. """
    # Textual visualization of the model
    def information(self, config=True, devices=True, model=False, misc=True):
        """ Show information about model configuration, used devices, train steps and more. """
        print(self._information(config=config, devices=devices, model=model, misc=misc))

    def info(self):
        """ Return the info message with default parameters. """
        return self._information()

    def _information(self, config=True, devices=True, model=False, misc=True):
        """ Create information string. """
        message = ''
        template = '\n##### {}:\n'

        if config:
            message += template.format('Config')
            message += pformat(self.full_config.config) + '\n'

        if devices:
            message += template.format('Devices')
            message += f'Leading device is {self.device}\n'
            if self.devices:
                message += '\n'.join([f'Device {i} is {d}' for i, d in enumerate(self.devices)])

        if model:
            message += template.format('Model')
            message += str(self.model) + '\n'

        if misc:
            message += template.format('Additional info')

            if self.input_shapes:
                message += '\n'.join([f'Input {i} has shape {s}' for i, s in enumerate(self.input_shapes)])
            if self.target_shape:
                message += f'\nTarget has shape {self.target_shape}'
            if self.classes:
                message += f'\nNumber of classe: {self.classes}'

            if self.model:
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                message += f'\nTotal number of parameters in model: {num_params}'

            message += f'\nTotal number of passed training iterations: {self.iteration}\n'

            message += template.format('Last iteration params')
            message += pformat(self.iter_info)
        return message

    def short_repr(self):
        """ Show simplified model layout. """
        print(self._short_repr())

    def _short_repr(self):
        self.model.apply(lambda module: setattr(module, 'short_repr', True))
        msg = repr(self.model)
        self.model.apply(lambda module: setattr(module, 'short_repr', False))
        return msg

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

    def show_lr(self):
        """ Plot graph of learning rate over iterations. """
        plt.figure(figsize=(8, 6))
        plt.plot(self.lr_list)

        plt.title('Learning rate', fontsize=18)
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(r'$\lambda$', fontsize=18, rotation=0)
        plt.grid(True)
        plt.show()

    # Visualize intermediate layers: activations and features
    def get_intermediate_activations(self, *args, layers=None, feed_dict=None, **kwargs):
        """ Compute the intermediate activations of a given layers in the same structure (tuple/list/dict).

        Under the hood, a forward hook is registered to capture the output of a targeted layers,
        and it is removed after extraction of all the activations.

        Parameters
        ----------
        layers : nn.Module, sequence or dict
            If nn.Module, then must be a part of the `model` attribute to get activations from.
            If sequence, then multiple such modules.
            If dictionary, then values must be such modules.
        args : sequence
            Arguments to be passed directly into the model.
        feed_dict : dict
            If ``initial_block/inputs`` are set, then this argument allows to pass data inside,
            with keys being names and values being actual data.
        kwargs : dict
            Additional named arguments directly passed to `feed_dict`.

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

        container = {key: LayerExtractor(module)
                     for key, module in container.items()}

        # Parse inputs to model, run model
        inputs, _ = self._make_prediction_inputs(*args, targets=None, feed_dict=feed_dict, **kwargs)
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
        input_shape = input_shape or self.input_shapes[0][1:]
        image = np.random.uniform(*ranges, input_shape)[None]
        image_var = torch.from_numpy(image.astype(np.float32)).to(self.device)
        image_var.requires_grad = True

        # Set up optimization procedure
        extractor = LayerExtractor(layer)
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

    def get_gradcam(self, *args, targets=None, feed_dict=None,
                    layer=None, gradient_mode='onehot', cam_class=None, **kwargs):
        """ Create visual explanation of a network decisions, based on the intermediate layer gradients.
        Ramprasaath R. Selvaraju, et al "`Grad-CAM: Visual Explanations
        from Deep Networks via Gradient-based Localization <https://arxiv.org/abs/1610.02391>`_"

        Under the hood, forward and backward hooks are used to extract the activation and the gradient,
        and the mean value of gradients along channels are used as weights for activation summation.

        Parameters
        ----------
        layers : nn.Module, sequence or dict
            Part of the model to base visualizations on.
        gradient_mode : Tensor or str
            If Tensor, then used directly to backpropagate from.
            If `onehot`, then OHE is created with `cam_class` parameter.
            If `targets`, then targets argument is used.
            Otherwise, model prediction is used.
        cam_class : int
            If gradient mode is `onehot`, then class to visualize. Default is the model prediction.
        args : sequence
            Arguments to be passed directly into the model.
        feed_dict : dict
            If ``initial_block/inputs`` are set, then this argument allows to pass data inside,
            with keys being names and values being actual data.
        kwargs : dict
            Additional named arguments directly passed to `feed_dict`.
        """
        extractor = LayerExtractor(layer)
        inputs, targets = self._make_prediction_inputs(*args, targets=targets, feed_dict=feed_dict, **kwargs)
        inputs = self.transfer_to_device(inputs)

        self.model.eval()
        prediction = self.model(inputs)

        if isinstance(gradient_mode, np.ndarray):
            gradient = self.transfer_to_device(gradient_mode)
        elif 'targ' in gradient_mode:
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
            input_shape = self.input_shapes[-1]
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
                    extractors.append(LayerExtractor(module))
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

    def get_signal_propagation_plot(self, model=None, input_tensor=None, statistics=None):
        """ Visualize signal propagation plot.

        Parameters
        ----------
        model : nn.Module
            Model to base visualizations on.
        input_tensor : Tensor
            Input tensor for signal propagation.
        statistics : dict
            Dict with signal propagation statistics.
        """
        if statistics is None:
            statistics = self.get_signal_propagation(model=model, input_tensor=input_tensor)

        fig, axes = plt.subplots(1, len(statistics)-1, figsize=(15, 5))
        for (ax, (title, data)) in zip(axes, statistics.items()):
            ax.plot(data)
            ax.set_title(title + " over network units", fontsize=14)
            ax.set_xlabel("Network depth", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True)
        fig.show()


class LayerExtractor:
    """ Create hook to get layer activations and gradients. """
    def __init__(self, module):
        self.activation = None
        self.gradient = None

        self.forward_handle = module.register_forward_hook(self.forward_hook)
        self.backward_handle = module.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """ Save activations: if multi-output, the first one is saved. """
        _ = module, input
        if isinstance(output, (tuple, list)):
            output = output[-1]
        self.activation = output

    def backward_hook(self, module, grad_input, grad_output):
        """ Save gradients: if multi-output, the first one is saved. """
        _ = module, grad_input
        if isinstance(grad_output, (tuple, list)):
            grad_output = grad_output[0]
        self.gradient = grad_output

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __del__(self):
        self.close()
