""" Monitoring (memory usage, cpu/gpu utilization) tools. """
import os
import time
from ast import literal_eval
from multiprocessing import Process, Manager, Queue

import numpy as np
try:
    import psutil
except ImportError:
    pass
try:
    import nvidia_smi
except ImportError:
    # Use this value to raise ImportError later
    nvidia_smi = None

from .plotter import plot
from .decorators import deprecated



class ResourceMonitor:
    """ Periodically runs supplied function in a separate process and stores its outputs.

    The created process runs infinitely until it is killed by SIGKILL signal.

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
        self.running = False

        self.manager = None

        self.stop_queue = None
        self.shared_list = None
        self.process = None

        self.start_time, self.prev_time, self.end_time = None, None, None
        self.ticks, self.data = [], []


    @staticmethod
    def endless_repeat(shared_list, stop_queue, function, frequency, **kwargs):
        """ Repeat `function` and storing results, until `stop` signal is recieved. """
        while stop_queue.empty():
            # As this process is killed ungracefully, it can be shut down in the middle of data appending.
            # We let Python handle it by ignoring the exception.
            try:
                shared_list.append(function(**kwargs))
            except (BrokenPipeError, ConnectionResetError):
                pass
            time.sleep(frequency)

    def start(self):
        """ Start a separate process with function calls every `frequency` seconds. """
        self.running = True
        self.manager = Manager()
        self.shared_list = self.manager.list()
        self.stop_queue = Queue()

        self.start_time = time.time()
        self.prev_time = self.start_time

        args = self.shared_list, self.stop_queue, self.function, self.frequency
        self.process = Process(target=self.endless_repeat, args=args, kwargs={'pid': self.pid, **self.kwargs})
        self.process.start()

    def fetch(self):
        """ Append collected data to the instance attributes. """
        n = len(self.data)
        # We copy data so additional points don't appear during this function execution
        self.data = self.shared_list[:]
        self.end_time = time.time()

        # Compute one more entry
        point = self.function(pid=self.pid, **self.kwargs)
        tick = time.time()

        # Update timestamps, append additional entries everywhere
        # If data was appended to `shared_list` during the execution of this function, the order might be wrong;
        # But, as it would mean that the time between calls to `self.function` is very small, it is negligeable.
        self.ticks.extend(np.linspace(self.prev_time, self.end_time, num=len(self.data) - n).tolist())
        self.data.append(point)
        self.shared_list.append(point)
        self.ticks.append(tick)

        self.prev_time = time.time()

    def stop(self):
        """ Stop separate process. """
        if self.running:
            self.stop_queue.put(True)
            self.process.join()
            self.running = False
            self.manager.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.fetch()
        self.stop()

    def plot(self, plotter=None, positions=None, slice=None, **kwargs):
        """ Simple plots of collected data-points. """
        x = np.array(self.ticks) - self.ticks[0]
        x = x if slice is None else x[slice]

        data, stats = [], []
        y = np.array(self.data).reshape(len(x), -1)
        for s in range(y.shape[1]):
            y_ = y[:, s]
            y_ = y_ if slice is None else y_[slice]
            data.append((x, y_))
            stats.append(f'MEAN: {np.mean(y_):4.4} STD: {np.std(y_):4.4}')

        name = self.__class__.__name__
        if 'GPU' in name:
            used_gpus = self.kwargs.get('gpu_list', get_current_gpus())
            if len(used_gpus) == 1:
                name = f'{name} on device `{used_gpus[0]}`'
            else:
                name = f'{name} on devices `{str(used_gpus)[1:-1]}`'

        plot_config = {
            'title': name,
            'label': stats,
            'smoothed_label': '',
            'legend_loc': 'best',
            'xlabel': 'Time, s',
            'ylabel': self.UNIT,
            'ylabel_rotation': 'horizontal',
            'ylabel_labelpad': 15,
            'grid': 'major',
            **kwargs
        }

        if plotter is None:
            plotter = plot(mode='curve', combine='separate', ratio=1, scale=0.5)

        plot_config = {**plotter.config, **plot_config}
        return plotter(data=data, mode='curve', positions=positions, **plot_config)

    deprecation_msg = "`{}` is deprecated and will be removed in future versions, use `{}` instead."
    visualize = deprecated(deprecation_msg.format('ResourceMonitor.visualize', 'ResourceMonitor.plot'))(plot)

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
        return process.memory_info().rss / (1024 ** 3) # gbytes


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


def get_current_gpus():
    """ If the `CUDA_VISIBLE_DEVICES` is set, check it and return device numbers. Otherwise, return [0]. """
    env_variable = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    env_variable = literal_eval(env_variable)
    return list(env_variable) if isinstance(env_variable, tuple) else [env_variable]

class GPUMonitor(ResourceMonitor):
    """ Track GPU usage. """
    UNIT = '%'

    def __init__(self, *args, **kwargs):
        if nvidia_smi is None:
            raise ImportError('Install Python interface for nvidia_smi')
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_usage(gpu_list=None, **kwargs):
        """ Track GPU usage. """
        _ = kwargs
        gpu_list = gpu_list or get_current_gpus()
        nvidia_smi.nvmlInit()
        handle = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
        res = [nvidia_smi.nvmlDeviceGetUtilizationRates(item) for item in handle]
        return [item.gpu for item in res]


class GPUMemoryUtilizationMonitor(ResourceMonitor):
    """ Track GPU memory utilization. """
    UNIT = '%'

    def __init__(self, *args, **kwargs):
        if nvidia_smi is None:
            raise ImportError('Install Python interface for nvidia_smi')
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_usage(gpu_list=None, **kwargs):
        """ Track GPU memory utilization. """
        _ = kwargs
        gpu_list = gpu_list or get_current_gpus()
        nvidia_smi.nvmlInit()
        handle = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
        res = [nvidia_smi.nvmlDeviceGetUtilizationRates(item) for item in handle]
        return [item.memory for item in res]


class GPUMemoryMonitor(ResourceMonitor):
    """ Track GPU memory usage. """
    UNIT = '%'

    def __init__(self, *args, **kwargs):
        if nvidia_smi is None:
            raise ImportError('Install Python interface for nvidia_smi')
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_usage(gpu_list=None, **kwargs):
        """ Track GPU memory usage. """
        _ = kwargs
        gpu_list = gpu_list or get_current_gpus()
        nvidia_smi.nvmlInit()
        handle = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
        res = [nvidia_smi.nvmlDeviceGetMemoryInfo(item) for item in handle]
        res = [100 * item.used / item.total for item in res]
        nvidia_smi.nvmlShutdown()
        return res



MONITOR_ALIASES = {
    MemoryMonitor: ['mmonitor', 'memory', 'memorymonitor'],
    CPUMonitor: ['cmonitor', 'cpu', 'cpumonitor'],
    RSSMonitor: ['rss'],
    VMSMonitor: ['vms'],
    USSMonitor: ['uss'],
    GPUMonitor: ['gpu'],
    GPUMemoryMonitor: ['gpu_memory'],
    GPUMemoryUtilizationMonitor: ['gpu_memory_utilization']
}

MONITOR_ALIASES = {alias: monitor for monitor, aliases in MONITOR_ALIASES.items()
                   for alias in aliases}


class Monitor(list):
    """ Holder for multiple monitors with simple visualization method. """
    def __init__(self, monitors=('cpu', 'memory', 'gpu'), frequency=0.1, **kwargs):
        monitors = [monitors] if not isinstance(monitors, (tuple, list)) else monitors
        monitors = [MONITOR_ALIASES[monitor.lower()](frequency=frequency, **kwargs)
                    if isinstance(monitor, str) else monitor
                    for monitor in monitors]

        super().__init__(monitors)


    def __enter__(self):
        for monitor in self:
            monitor.start()
        return self[0] if len(self) == 0 else self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for monitor in self:
            monitor.fetch()
            monitor.stop()

    def plot(self, plotter=None, positions=None, savepath=None, **kwargs):
        """ Visualize multiple monitors in a single figure. """
        plot_config = {
            'ratio': 1 / len(self),
            'scale': 0.5,
            'ncols': None if 'nrows' in kwargs else len(self),
            **kwargs
        }

        if plotter is None:
            plotter = plot(data=[None] * len(self), mode='curve', combine='separate', **plot_config)

        positions = range(len(self))
        for position, monitor in zip(positions, self):
            monitor.plot(plotter=plotter, positions=position, **kwargs)

        if savepath is not None:
            plotter.save(savepath=savepath)

        return plotter
