""" Base class for model callbacks: describes API. """
from abc import ABC, abstractmethod
from time import gmtime, strftime

from ....monitor import USSMonitor, GPUMonitor, GPUMemoryMonitor


def file_print(path, msg):
    """ Print to a file. """
    with open(path, 'a+') as f:
        print(msg, file=f)



class BaseCallback(ABC):
    """ Base class for callbacks.

    Parameters
    ----------
    stream : None, callable or str
        If None, then no logging is performed.
        If callable, then used to display message, for example, `print`.
        If str, then must be path to file to write log to.
    """
    def __init__(self, stream=None):
        self.model = None
        self.stream = self.make_stream(stream)

    def make_stream(self, stream):
        """ Parse the `stream` argument into callable. """
        if stream is None:
            return lambda *args: args

        if callable(stream):
            return stream

        if isinstance(stream, str):
            with open(stream, 'w') as _:
                pass
            return lambda msg: file_print(stream, msg)
        raise TypeError('`Stream` argument must be either None, callable or string.')

    def set_model(self, model):
        """ Save reference to the model.
        Depending on the callback, following attribytes may be accessed:
            - `iteration`
            - `loss_list`
            - `microbatch`
            - `iter_info`
        """
        self.model = model
        return self

    @abstractmethod
    def on_iter_end(self, **kwargs):
        """ Action to be performed on the end of iteration. Called at the end of :meth:`TorchModel.train`. """


class LogCallback(BaseCallback):
    """ Log loss and various model information at the desired stream.

    Parameters
    ----------
    stream : None, callable or str
        If None, then no logging is performed.
        If callable, then used to display message, for example, `print`.
        If str, then must be path to file to write log to.
    frequency : int
        Frequency of log messages.
    resources : bool
        Whether to log memory, GPU memory and GPU utilization.
    shapes : bool
        Whether to log model input and output shapes.
    microbatch : bool
        Whether to log size and number of microbatches.
    """
    def __init__(self, stream=None, frequency=1, resources=True, shapes=False, microbatch=False):
        self.frequency = frequency
        self.resources = resources
        self.shapes = shapes
        self.microbatch = microbatch

        header = self.make_header()
        super().__init__(stream=stream)
        self.stream(header)

    def make_header(self):
        """ Create the log description. """
        header = 'TIMESTAMP   ITERATION : LOSS'

        if self.resources:
            header += ' | MEMORY : GPU_MEMORY : GPU_UTILIZATION'
        if self.shapes:
            header += ' | IN_SHAPES : OUT_SHAPES'
        if self.microbatch:
            header += ' | MICROBATCH_SIZE : N_MICROBATCHES'
        return header

    def on_iter_end(self, **kwargs):
        """ Log requested information. Called at the end of :meth:`TorchModel.train`. """
        _ = kwargs
        i = self.model.iteration

        if i % self.frequency == 0:
            # Default message: timestamp, iteration and loss value
            timestamp = strftime('%Y-%m-%d  %H:%M:%S', gmtime())
            avg_loss = sum(self.model.loss_list[-self.frequency:]) / self.frequency

            msg = f'{timestamp}   {i:5}: {avg_loss:6.6f}'

            if self.resources:
                # Monitor resources
                memory = round(USSMonitor.get_usage(), 2)
                gpu_memory = GPUMemoryMonitor.get_usage()
                gpu_utilization = GPUMonitor.get_usage()

                msg += f' | {memory:6.3f} : '
                msg += ' : '.join([f'{round(item, 2):6.3f}' for item in gpu_memory]) + ' : '
                msg += ' : '.join([f'{round(item, 2):6.3f}' for item in gpu_utilization])

            if self.shapes:
                # Model in/out shapes
                in_shapes = self.model.iter_info['actual_model_inputs_shape']
                out_shapes = self.model.iter_info['actual_model_outputs_shape']

                msg += f' | {in_shapes} : {out_shapes}'

            if self.microbatch:
                # Internal microbatch parameters of the last iteration
                microbatch = self.model.microbatch
                num_microbatches = self.model.iter_info['steps']

                msg += f' | {microbatch} : {num_microbatches}'

            self.stream(msg)
