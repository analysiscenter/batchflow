import numpy as np
import pandas as pd

import torch
from ptflops import get_model_complexity_info


memory_unit_constants = {'GB': 1e-9, 'MB': 1e-6, 'KB': 1e-3, 'B': 1.}


class Time_tracker:
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        
    def timer(self):
        return self.start.elapsed_time(self.end)


class Memory_tracker:
    def __init__(self, device=None):
        self.device = device
        
    def __enter__(self):
        self.start_memory = get_memory(device=self.device)
        return self

    def __exit__(self, type, value, traceback):
        self.end_memory = get_memory(reset_memory=False, device=self.device) 
        
    def memory_allocated(self):
        return self.end_memory - self.start_memory 



def get_memory(reset_memory=True, device=None):
    """Take current max allocated memory, either with or without resetting"""
    if reset_memory:
        torch.cuda.reset_peak_memory_stats()
            
    max_memory = torch.cuda.max_memory_allocated(device)
        
    return max_memory


def make_initialization_inputs(inputs, device=None):
    """ Take either tensor, shape tuple or list of them, and always return tensor or list of them. """
    if isinstance(inputs, torch.Tensor):
        pass
    elif isinstance(inputs, tuple):
        inputs = torch.rand(*inputs, device=device)
    elif isinstance(inputs, list):
        inputs = [make_initialization_inputs(item, device=device) for item in inputs]
    return inputs

    

def tracker(module, inputs, repeats=300, warmup=40, device=None, track_backward=True, # .py file ???
            channels_last=False, amp=False, memory_unit='MB') -> dict:
    """Track module #macs, #parameters, time and memory consumption on forward and backward pass for a given inputs tensor or inputs shape"""
    
    memory_unit_constant = memory_unit_constants[memory_unit]
    
    with Time_tracker() as total_time:
    
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
                with Time_tracker() as t:
                    outputs = module(inputs)

                forward_time = t.timer()        
                forward_timings.append(forward_time) 

                
            if track_backward:
                # calculate backward operation time 
                with Time_tracker() as t:
                    outputs.backward(outputs)
                backward_time = t.timer()
                backward_timings.append(backward_time)

        result['forward time mean(ms)'] = np.mean(forward_timings) 
        result['forward time std(ms)'] = np.std(forward_timings)

        
        # calculate forward memory
        with Memory_tracker(device=device) as memory:
            module(inputs)
        forward_memory = memory.memory_allocated()
        result['forward memory'] = forward_memory * memory_unit_constant

        
        if track_backward:
            result['backward time mean(ms)'] = np.mean(backward_timings)
            result['backward time std(ms)'] = np.std(backward_timings)
            
            # calculate backward memory
            outputs = module(inputs)
            with Memory_tracker(device=device) as memory:
                outputs.backward(outputs)
            backward_memory = memory.memory_allocated()
            result['backward memory'] = backward_memory * memory_unit_constant

            
        macs, params = get_model_complexity_info(module, tuple(inputs.shape[1:]), as_strings=False, print_per_layer_stat=False)
        result['macs'] = macs
        result['parameters'] = float(params)

    result['time total(ms)'] = total_time.timer()
    
    return result
