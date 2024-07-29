""" Monitoring (memory usage, cpu/gpu utilization) tools. """
import os
import time
from copy import copy
from ast import literal_eval
# from multiprocessing import Process, Manager, Queue
from threading import Thread

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

from .decorators import deprecated



class ResourceMonitor:
    """ Periodically runs supplied function in a separate thread and stores its outputs.

    Parameters
    ----------
    function : callable
        Function to use. If not provided, defaults to the `get_usage` static method.
    frequency : number
        Periodicity of function calls in seconds.
    **kwargs
        Passed directly to `function` calls.
        Update `self.kwargs` dictionary in subclasses to pass arguments to their `get_usage` methods.

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

        self.running = False
        self.thread = None

        self.ticks, self.data = [], []

    def endless_repeat(self):
        """ Call `function` and record timestamp, until the monitor is stopped. """
        while self.running:
            self.data.append(self.function(**self.kwargs))
            self.ticks.append(time.time())
            time.sleep(self.frequency)

    def start(self):
        """ Start a separate thread with `function` calls every `frequency` seconds. """
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.endless_repeat, name=f'BatchFlow-{type(self).__name__}')
            self.thread.start()

    def stop(self):
        """ Stop separate thread.
        Add another data point and timestamp to the containers, so that monitor has at least two measurements.
        """
        if self.running:
            self.running = False
            self.thread.join()
            self.thread = None

            self.data.append(self.function(**self.kwargs))
            self.ticks.append(time.time())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

    def __del__(self):
        self.stop()

    def plot(self, plotter=None, positions=None, slice=None, **kwargs):
        """ Simple plots of collected data-points. """
        #pylint: disable=invalid-name
        x = np.array(self.ticks) - self.ticks[0]
        y = np.array(self.data)[:len(x)].reshape(len(x), -1)

        x = x if slice is None else x[slice]

        data, stats = [], []
        for s in range(y.shape[1]):
            y_ = y[:, s]
            y_ = y_ if slice is None else y_[slice]
            data.append((x, y_))
            stats.append(f'MEAN: {np.mean(y_):4.4} STD: {np.std(y_):4.4}')

        name = self.__class__.__name__
        if 'GPU' in name:
            used_gpus = self.kwargs.get('gpu_list')
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
            from .plotter import plot
            plotter = plot([None], mode='curve', combine='separate', ratio=1, scale=0.5)

        plot_config = {**plotter.config, **plot_config}
        return plotter(data=data, mode='curve', positions=positions, **plot_config)

    deprecation_msg = "`{}` is deprecated and will be removed in future versions, use `{}` instead."
    visualize = deprecated(deprecation_msg.format('ResourceMonitor.visualize', 'ResourceMonitor.plot'))(plot)



# General system resource monitors: don't need any extra info
class TotalCPUMonitor(ResourceMonitor):
    """ Track total CPU usage. """
    UNIT = '%'

    @staticmethod
    def get_usage(**kwargs):
        """ Track CPU usage. """
        _ = kwargs
        return psutil.cpu_percent()

class TotalMemoryMonitor(ResourceMonitor):
    """ Track total virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(**kwargs):
        """ Track total virtual memory usage. """
        _ = kwargs
        return psutil.virtual_memory().used / (1024 **3)

class TotalDiskMonitor(ResourceMonitor):
    """ Track total disk usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(**kwargs):
        """ Track total disk usage. """
        _ = kwargs
        return psutil.disk_usage('/').used / (1024 **3)


# Process resource monitors: track resources of a given process
class ProcessResourceMonitor(ResourceMonitor):
    """ Pre-init `psutil` Process instance for the current process or provided `pid`.
    Even though `psutil` keeps cached table of processes, it is still faster to have it in the instance itself.
    """
    def __init__(self, function=None, frequency=0.1, pid=None, **kwargs):
        super().__init__(function=function, frequency=frequency, **kwargs)
        self.kwargs['process'] = psutil.Process(pid or os.getpid())
        self.kwargs['start_data'] = self.get_usage(**self.kwargs)

    def __getstate__(self):
        state = copy(self.__dict__)
        state['kwargs']['process'] = state['kwargs']['process'].pid
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.kwargs['process'] = psutil.Process(self.kwargs['process'])


class DiskIOMonitor(ProcessResourceMonitor):
    """ Track disk i/o operations amount. """
    UNIT = 'number'

    @staticmethod
    def get_usage(process=None, start_data=0, **kwargs):
        """ Track disk i/o operations amount. """
        _ = kwargs

        counters = process.io_counters()
        value = counters.read_count + counters.write_count

        return value - start_data

class CPUMonitor(ProcessResourceMonitor):
    """ Track process CPU usage. """
    UNIT = '%'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process CPU usage. """
        _ = kwargs
        return process.cpu_percent()


class RSSMonitor(ProcessResourceMonitor):
    """ Track process non-swapped physical memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track non-swapped physical memory usage. """
        _ = kwargs
        return process.memory_info().rss / (1024 ** 3) # gbytes

class VMSMonitor(ProcessResourceMonitor):
    """ Track process virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process virtual memory usage. """
        _ = kwargs
        return process.memory_info().vms / (1024 ** 3) # gbytes

class SHRMonitor(ProcessResourceMonitor):
    """ Track process memory, potentially shared with other processes. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process unique virtual memory usage. """
        _ = kwargs
        return process.memory_info().shared / (1024 ** 3) # gbytes

class CodeMonitor(ProcessResourceMonitor):
    """ Track process memory, devoted to executable code. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process unique virtual memory usage. """
        _ = kwargs
        return process.memory_info().text / (1024 ** 3) # gbytes

class DataMonitor(ProcessResourceMonitor):
    """ Track process memory, devoted to anything but executable code. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process unique virtual memory usage. """
        _ = kwargs
        return process.memory_info().data / (1024 ** 3) # gbytes

class USSMonitor(ProcessResourceMonitor):
    """ Track process unique virtual memory usage. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track current process unique virtual memory usage. """
        _ = kwargs
        return process.memory_full_info().uss / (1024 ** 3) # gbytes

class RSSminusSHRMonitor(ProcessResourceMonitor):
    """ Track non-swapped physical memory usage minus the memory, potentially shared with the other processes. """
    UNIT = 'Gb'

    @staticmethod
    def get_usage(process=None, **kwargs):
        """ Track process unique virtual memory usage. """
        _ = kwargs
        info = process.memory_info()
        return (info.rss - info.shared) / (1024 ** 3) # gbytes


# GPU monitors: require list of devices, default to `CUDA_VISIBLE_DEVICES` env variable
class GPUResourceMonitor(ResourceMonitor):
    """ If the `CUDA_VISIBLE_DEVICES` is set, check it and return device numbers. Otherwise, return [0]. """
    def __init__(self, function=None, frequency=0.1, gpu_list=None, **kwargs):
        if nvidia_smi is None:
            raise ImportError('Install Python interface for nvidia_smi')
        super().__init__(function=function, frequency=frequency, **kwargs)

        # Fallback to env variable
        if not gpu_list:
            env_variable = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            env_variable = literal_eval(env_variable)
            gpu_list = list(env_variable) if isinstance(env_variable, tuple) else [env_variable]

        nvidia_smi.nvmlInit()
        gpu_handles = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in gpu_list]
        self.kwargs.update({'gpu_list': gpu_list, 'gpu_handles': gpu_handles})

    def __getstate__(self):
        state = copy(self.__dict__)
        state['kwargs']['gpu_handles'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        nvidia_smi.nvmlInit()
        self.kwargs['gpu_handles'] = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in self.kwargs['gpu_list']]

    def __del__(self):
        super().__del__()
        nvidia_smi.nvmlShutdown()


class GPUMonitor(GPUResourceMonitor):
    """ Track GPU usage. """
    UNIT = '%'

    @staticmethod
    def get_usage(gpu_handles=None, **kwargs):
        """ Track GPU usage. """
        _ = kwargs
        return [nvidia_smi.nvmlDeviceGetUtilizationRates(item).gpu for item in gpu_handles]

class GPUMemoryUtilizationMonitor(GPUResourceMonitor):
    """ Track GPU memory utilization. """
    UNIT = '%'

    @staticmethod
    def get_usage(gpu_handles=None, **kwargs):
        """ Track GPU memory utilization. """
        _ = kwargs
        return [nvidia_smi.nvmlDeviceGetUtilizationRates(item).memory for item in gpu_handles]

class GPUMemoryMonitor(GPUResourceMonitor):
    """ Track GPU memory usage. """
    UNIT = '%'

    @staticmethod
    def get_usage(gpu_handles=None, **kwargs):
        """ Track GPU memory usage. """
        _ = kwargs
        result = [nvidia_smi.nvmlDeviceGetMemoryInfo(item) for item in gpu_handles]
        result = [100 * item.used / item.total for item in result]
        return result



MONITOR_ALIASES = {
    # System-wide monitors
    TotalCPUMonitor: ['total_cpu'],
    TotalMemoryMonitor: ['total_memory', 'total_rss'],
    TotalDiskMonitor: ['total_disk'],

    # Process monitors
    DiskIOMonitor: ['disk_io', 'io_disk'],
    CPUMonitor: ['cpu'],
    RSSMonitor: ['rss'],
    VMSMonitor: ['vms'],
    SHRMonitor: ['shr', 'shared'],
    CodeMonitor: ['code', 'text'],
    DataMonitor: ['data'],
    USSMonitor: ['uss'],
    RSSminusSHRMonitor: ['rss-shared', 'rss_corrected'],

    # GPU monitors
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

    def start(self):
        """ Start all of the monitors. """
        for monitor in self:
            monitor.start()
        return self[0] if len(self) == 1 else self

    def stop(self):
        """ Stop all of the monitors. """
        for monitor in self:
            monitor.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()


    def plot(self, plotter=None, positions=None, savepath=None, **kwargs):
        """ Visualize multiple monitors in a single figure. """
        plot_config = {
            'ratio': 1 / min(len(self), 4),
            'scale': 0.5,
            'ncols': None if 'nrows' in kwargs else min(len(self), 4),
            **kwargs
        }

        if plotter is None:
            from .plotter import plot
            plotter = plot(data=[None] * len(self), mode='curve', combine='separate', **plot_config)

        positions = range(len(self))
        for position, monitor in zip(positions, self):
            monitor.plot(plotter=plotter, positions=position, **kwargs)

        if savepath is not None:
            plotter.save(savepath=savepath)

        return plotter
