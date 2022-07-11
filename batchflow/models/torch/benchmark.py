""" Time and memory trackers for benchmark operations with nn.Module. """
import numpy as np
import torch
from ptflops import get_model_complexity_info

from .utils import make_initialization_inputs

# Different units for memory representation
MEMORY_UNIT_CONSTANTS = {
    'GB': 1 / (1024 ** 3),
    'MB': 1 / (1024 ** 2),
    'KB': 1 / 1024,
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
        """ Get an operation time. """
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
        """ Get allocated memory for a module. """
        return self.end_memory - self.start_memory

def get_module_performance(module, inputs, n_repeats=300, warmup=40, device=None, track_backward=True,
                           channels_last=False, amp=False, memory_unit='MB'):
    """ Track module #macs, #parameters, time and memory consumption on forward and backward
    pass for a given input tensor or inputs shape.

    Parameters
    ----------
    module : nn.Module
        Input module for which we track performance.
    inputs : Tensor or sequence of ints
        If Tensor, then it is a data which we use for tracking performance.
        If sequence of ints, then it is a shape of tensor which will be generated for tracking performance.
    n_repeats : int
        Number of times to repeat forward and backward pass for tracking performance.
    warmup : int
        Number of starting iterations that won't be tracked.
    device : str or torch.cuda.Device
        Device for computations.
        If str, then any option of device configuration from :class:`torch.nn.Module` is supported.
    track_backward : bool
        If True, then track time and memory for the backward operation.
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
        Dictionary of results of the module performance.
        It contains forward and backward (if required):
            - parameters number
            - MACs
            - time (mean and std)
            - allocated memory
            - total time for forward and backward together.
    """
    memory_unit_constant = MEMORY_UNIT_CONSTANTS[memory_unit]

    with TimeTracker() as total_timer:
        result = {}
        forward_timings = []
        backward_timings = []

        torch.cuda.empty_cache()

        inputs = make_initialization_inputs(inputs=inputs, device=device)
        module.to(device)

        if channels_last:
            inputs.to(memory_format=torch.channels_last)
            module.to(memory_format=torch.channels_last)

        for i in range(n_repeats + warmup):
            if i < warmup:
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = module(inputs)

                outputs.backward(outputs)
                continue

            with torch.cuda.amp.autocast(enabled=amp):
                # Calculate forward operation time
                with TimeTracker() as forward_timer:
                    outputs = module(inputs)

                forward_time = forward_timer.value
                forward_timings.append(forward_time)

            if track_backward:
                # Calculate backward operation time
                with TimeTracker() as backward_timer:
                    outputs.backward(outputs)

                backward_time = backward_timer.value
                backward_timings.append(backward_time)

        result['forward time mean(ms)'] = np.mean(forward_timings)
        result['forward time std(ms)'] = np.std(forward_timings)

        # Calculate forward memory
        with MemoryTracker(device=device) as memory:
            outputs = module(inputs)

        forward_memory = memory.value
        result['forward memory'] = forward_memory * memory_unit_constant

        if track_backward:
            result['backward time mean(ms)'] = np.mean(backward_timings)
            result['backward time std(ms)'] = np.std(backward_timings)

            # Calculate backward memory
            with MemoryTracker(device=device) as memory:
                outputs.backward(outputs)

            backward_memory = memory.value
            result['backward memory'] = backward_memory * memory_unit_constant

        macs, params = get_model_complexity_info(module, tuple(inputs.shape[1:]),
                                                 as_strings=False, print_per_layer_stat=False)
        result['macs'] = macs
        result['parameters'] = float(params)

    result['time total(ms)'] = total_timer.value
    return result
