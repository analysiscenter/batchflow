import numpy as np
import pandas as pd

import torch
from ptflops import get_model_complexity_info
from batchflow.models.torch.utils import make_initialization_inputs

# different units for memory representation
MEMORY_UNIT_CONSTANTS = {'GB': 1/(1024*1024*1024), 'MB': 1/(1024*1024), 'KB': 1/1024, 'B': 1.}

class TimeTracker:
    """Track time for operation on gpu."""
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
        return self.start.elapsed_time(self.end)

class MemoryTracker:
    """Track used memory for operation on gpu."""
    def __init__(self, device=None):
        self.device = device
        
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_memory = torch.cuda.max_memory_allocated(self.device)
        return self

    def __exit__(self, type, value, traceback):
        self.end_memory = torch.cuda.max_memory_allocated(self.device) 
        
    @property
    def value(self):
        return self.end_memory - self.start_memory 

def get_module_info(module, inputs, repeats=300, warmup=40, device=None, track_backward=True,
                    channels_last=False, amp=False, memory_unit='MB') -> dict:
    """
    Track module #macs, #parameters, time and memory consumption on forward and backward 
    pass for a given inputs tensor or inputs shape.
    
    Parameters
    ----------
    module : torch.nn.modules    
    inputs : (tensor, tuple, list)
             Data which enter to the module.
    repeats : int
              The number shows how many times we want to repeat
              module for tracking memory and time.
    warmup : int
             Do a few iterations for stabilize std.
    device : str
             Device can be 'cpu' or 'gpu'.
    track_backward : bool
                     If True we want to track time and memory for backward operation.
    channels_last : bool
                    You can change strides by choosing memory format = channels_last 
                    for your module and check how module perfomance changes.
    amp : bool
          Set True or False for the parameter enabled in torch.cuda.amp.autocast(enabled=amp)
    memory_unit : str
                  See MEMORY_UNIT_CONSTANTS which units you can choose for memory
    
    Returns
    -------
    dict
        Keys of dict are forward/backward time/memory and values are their results.
    """
    
    memory_unit_constant = MEMORY_UNIT_CONSTANTS[memory_unit]
    
    with TimeTracker() as total_time:
    
        result = {}

        torch.cuda.empty_cache()

        inputs = make_initialization_inputs(inputs=inputs, device=device)
        module.to(device)

        if channels_last:
            inputs.to(memory_format=torch.channels_last)
            module.to(memory_format=torch.channels_last)          

        forward_timings = []
        backward_timings = []

        for i in range(repeats + warmup):

            if i < warmup:
                with torch.cuda.amp.autocast(enabled=amp):
                    outputs = module(inputs)
                outputs.backward(outputs)
                del outputs
                torch.cuda.empty_cache()
                i += 1
                continue
                
            with torch.cuda.amp.autocast(enabled=amp):
                # calculate forward operation time  
                with TimeTracker() as ft:
                    outputs = module(inputs)

                forward_time = ft.value        
                forward_timings.append(forward_time) 

            if track_backward:
                # calculate backward operation time 
                with TimeTracker() as bt:
                    outputs.backward(outputs)
                backward_time = bt.value
                backward_timings.append(backward_time)

        result['forward time mean(ms)'] = np.mean(forward_timings) 
        result['forward time std(ms)'] = np.std(forward_timings)

        # calculate forward memory
        with MemoryTracker(device=device) as memory:
            module(inputs)
        forward_memory = memory.value
        result['forward memory'] = forward_memory * memory_unit_constant

        if track_backward:
            result['backward time mean(ms)'] = np.mean(backward_timings)
            result['backward time std(ms)'] = np.std(backward_timings)
            
            # calculate backward memory
            outputs = module(inputs)
            with MemoryTracker(device=device) as memory:
                outputs.backward(outputs)
            backward_memory = memory.value
            result['backward memory'] = backward_memory * memory_unit_constant

        macs, params = get_model_complexity_info(module, tuple(inputs.shape[1:]), 
                                                 as_strings=False, print_per_layer_stat=False)
        result['macs'] = macs
        result['parameters'] = float(params)

    result['time total(ms)'] = total_time.value
    return result
