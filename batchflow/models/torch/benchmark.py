""" Time and memory trackers for benchmark operations with nn.Module. """
import numpy as np
import torch
try:
    import ptflops
except ImportError:
    ptflops = None

from .utils import make_initialization_inputs



# Different units for memory representation
MEMORY_UNIT_CONSTANTS = {
    'GB': 1024 ** 3,
    'MB': 1024 ** 2,
    'KB': 1024,
    'B': 1.
}

class TimeTracker:
    """ Measure time taken for an operation on GPU. """
    def __init__(self):
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()

    @property
    def value(self):
        """ Get total operation time. """
        return self.start.elapsed_time(self.end)

class MemoryTracker:
    """ Measure peak used memory for an operation on GPU. """
    def __init__(self, device=None):
        self.device = device
        self.start_memory = None
        self.end_memory = None

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_memory = torch.cuda.max_memory_allocated(self.device)
        return self

    def __exit__(self, type, value, traceback):
        self.end_memory = torch.cuda.max_memory_allocated(self.device)

    @property
    def value(self):
        """ Get peak allocated memory. """
        return self.end_memory - self.start_memory


def get_module_performance(module, inputs, track_backward=True, n_repeats=300, warmup=40, device=None,
                           channels_last=False, amp=False, memory_unit='MB'):
    """ Measure module performance: forward/backward time and memory consumption, number of parameters and operations.
    Under the hood, works by passing `inputs` `n_repeats` times while fetching data from device sensors.

    If `ptflops` library is installed, also outputs the number of multiply-add operations (MACs).

    Parameters
    ----------
    module : nn.Module
        Input module for which we track performance.
    inputs : Tensor or sequence of ints
        If Tensor, then directly used as module input.
        If sequence of ints, then interpreted as shape of a tensor to make. The shape must include batch dimension.
    track_backward : bool
        If True, then track time and memory for the backward operation.
    n_repeats : int
        Number of times to repeat forward and backward pass.
    warmup : int
        Number of starting iterations that won't be tracked.
    device : str or torch.cuda.Device
        Device for computations. Any option of PyTorch device configuration is supported.
    channels_last : bool
        Whether to use `torch.channels_last` memory format.
    amp : bool
        Whether to enable :class:`torch.cuda.amp.autocast`.
    memory_unit : str
        Memory units that are used for memory representation in the result.
        Possible options are: 'GB', 'MB', 'KB', 'B'.

    Returns
    -------
    dict
        Dictionary with results of measure module performance.
        It contains following keys:
            - `params`. Number of parameters in the module.
            - `macs`. MACs, Estimated for `PyTorch` native operations only.
            - forward and backward (if required) statistics:
                - time (mean and std).
                - allocated memory.
            - `total_time`. Total time taken for all measurements.
    """
    memory_unit_constant = MEMORY_UNIT_CONSTANTS[memory_unit]

    # Prepare inputs
    torch.cuda.empty_cache()
    inputs = make_initialization_inputs(inputs=inputs, device=device)
    module.to(device)

    if channels_last:
        inputs.to(memory_format=torch.channels_last)
        module.to(memory_format=torch.channels_last)

    # One container for all results
    result = {}
    parameters = sum(p.numel() for p in module.parameters())
    result['parameters'] = parameters

    with TimeTracker() as total_timer:
        forward_timings, backward_timings = [], []

        for i in range(n_repeats + warmup):
            with torch.cuda.amp.autocast(enabled=amp):
                # Calculate forward operation time
                with TimeTracker() as forward_timer:
                    outputs = module(inputs)

                if i >= warmup:
                    forward_time = forward_timer.value
                    forward_timings.append(forward_time)

            if track_backward and parameters > 0:
                # Calculate backward operation time
                with TimeTracker() as backward_timer:
                    outputs.backward(outputs)

                if i >= warmup:
                    backward_time = backward_timer.value
                    backward_timings.append(backward_time)

        result['forward time mean, ms'] = np.mean(forward_timings)
        result['forward time std, ms'] = np.std(forward_timings)

        # Calculate forward memory
        with MemoryTracker(device=device) as memory:
            outputs = module(inputs)

        forward_memory = memory.value
        result[f'forward memory, {memory_unit}'] = forward_memory / memory_unit_constant

        if track_backward and parameters > 0:
            result['backward time mean, ms'] = np.mean(backward_timings)
            result['backward time std, ms'] = np.std(backward_timings)

            # Calculate backward memory
            with MemoryTracker(device=device) as memory:
                outputs.backward(outputs)

            backward_memory = memory.value
            result[f'backward memory, {memory_unit}'] = backward_memory / memory_unit_constant

        if ptflops is not None:
            macs, _ = ptflops.get_model_complexity_info(module, tuple(inputs.shape[1:]),
                                                        as_strings=False, print_per_layer_stat=False)
            result['macs'] = macs

    result['time total, ms'] = total_timer.value
    return result
