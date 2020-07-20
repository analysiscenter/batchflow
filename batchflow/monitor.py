""" Monitoring (memory usage, cpu/gpu utilization) tools. """
import os
import time
from multiprocessing import Process, Queue
from contextlib import contextmanager

import psutil
import numpy as np
import matplotlib.pyplot as plt

try:
    import nvidia_smi
except ImportError:
    pass



class ResourceMonitor:
    """ Periodically runs supplied function in a separate process and stores its outputs.

    Parameters
    ----------
    function : callable
        Function to use. If not provided, defaults to the `get_usage` static method.
    frequency : number
        Periodicity of function calls in seconds.
    **kwargs
        Passed directly to `function` calls.

    Attributes
    ----------
    data : list
        Collected function outputs. Preserved between multiple runs.
    ticks : list
        Times of function calls. Preserved between multiple runs.
    """
    def __init__(self, function=None, frequency=0.1, **kwargs):
        self.function = function or self.get_usage
        self.frequency = frequency
        self.kwargs = kwargs

        self.pid = os.getpid()
        self.repeat_queue, self.data_queue = None, None
        self.running = False

        self.start_time, self.end_time = None, None
        self.ticks, self.data = [], []


    @staticmethod
    def endless_repeat(function, frequency, repeat_queue, data_queue, **kwargs):
        """ Repeat `function` and storing results, until `stop` signal is recieved.. """
        res = []
        while repeat_queue.empty():
            res.append(function(**kwargs))
            time.sleep(frequency)
        data_queue.put(res)


    def start(self):
        """ Start a separate process with function calls. """
        self.running = True
        self.repeat_queue = Queue()
        self.data_queue = Queue()
        self.start_time = time.time()

        args = self.function, self.frequency, self.repeat_queue, self.data_queue
        p = Process(target=self.endless_repeat, args=args,
                    kwargs={'pid': self.pid, **self.kwargs})
        p.start()

    def stop(self):
        """ Stop separate process; append collected data to the already stored. """
        self.repeat_queue.put('stop')
        data = self.data_queue.get()
        self.end_time = time.time()
        self.running = False

        self.ticks.extend(np.linspace(self.start_time, self.end_time, num=len(data)).tolist())
        self.data.extend(data)
        return data

    def visualize(self):
        """ Simple plots of collected data-points. """
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(self.ticks) - self.ticks[0], self.data)
        plt.title(self.__class__.__name__)
        plt.xlabel('Time, s', fontsize=12)
        plt.ylabel(self.UNIT, fontsize=12, rotation='horizontal', labelpad=15)
        plt.grid(True)
        plt.show()



class CPUMonitor(ResourceMonitor):
    """ Track CPU usage. """
    UNIT = '%'

    @staticmethod
    def get_usage(**kwargs):
        """ Track CPU usage. """
        _ = kwargs
        return psutil.cpu_percent()


class MemoryMonitor(ResourceMonitor):
    """ Track total virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(**kwargs):
        """ Track total virtual memory usage. """
        _ = kwargs
        return psutil.virtual_memory().used / (1024 **3)


class RSSMonitor(ResourceMonitor):
    """ Track non-swapped physical memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(pid=None, **kwargs):
        """ Track non-swapped physical memory usage. """
        _ = kwargs
        process = psutil.Process(pid)
        return process.memory_info().rss / (1024 ** 2) # mbytes


class VMSMonitor(ResourceMonitor):
    """ Track current process virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(pid=None, **kwargs):
        """ Track current process virtual memory usage. """
        _ = kwargs
        process = psutil.Process(pid)
        return process.memory_info().vms / (1024 ** 3) # gbytes


class USSMonitor(ResourceMonitor):
    """ Track current process unique virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(pid=None, **kwargs):
        """ Track current process unique virtual memory usage. """
        _ = kwargs
        process = psutil.Process(pid)
        return process.memory_full_info().uss / (1024 ** 3) # gbytes


class GPUMonitor(ResourceMonitor):
    """ Track GPU usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(gpu_list=None, **kwargs):
        """ Track GPU usage. """
        _ = kwargs
        nvidia_smi.nvmlInit()
        handle = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
        res = [nvidia_smi.nvmlDeviceGetUtilizationRates(item) for item in handle]
        return [item.gpu for item in res]




MONITOR_ALIASES = {
    MemoryMonitor: ['mmonitor', 'memory', 'memorymonitor'],
    CPUMonitor: ['cmonitor', 'cpu', 'cpumonitor'],
    RSSMonitor: ['rss'],
    VMSMonitor: ['vms'],
    USSMonitor: ['uss'],
    GPUMonitor: ['gpu'],
}

MONITOR_ALIASES = {alias: monitor for monitor, aliases in MONITOR_ALIASES.items()
                   for alias in aliases}


@contextmanager
def monitor_resource(resource='memory', frequency=0.5, **kwargs):
    """ A convenient context manager to profile a part of code. Can use one or more monitors. """
    resource = [resource] if not isinstance(resource, (tuple, list)) else resource
    monitors = [MONITOR_ALIASES[res.lower()](frequency=frequency, **kwargs) if isinstance(res, str) else res
                for res in resource]

    try:
        for monitor in monitors:
            monitor.start()
        yield monitors[0] if len(monitors) == 1 else monitors
    finally:
        for monitor in monitors:
            monitor.stop()


def monitor_memory(frequency=0.5):
    return monitor_resource('memory', frequency=frequency)

def monitor_cpu(frequency=0.5):
    return monitor_resource('cpu', frequency=frequency)

def monitor_gpu(frequency=0.5, gpu_list=None):
    return monitor_resource('gpu', frequency=frequency, gpu_list=gpu_list)
