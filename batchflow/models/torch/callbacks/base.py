""" Base class for model callbacks: describes API. """
from abc import ABC, abstractmethod
from time import gmtime, strftime

from ....monitor import USSMonitor, GPUMonitor, GPUMemoryMonitor


def file_print(path, msg):
    """ Print to a file. """
    with open(path, 'a+') as f:
        print(msg, file=f)



class BaseCallback(ABC):
    """ Base class for callbacks. """
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
        self.model = model
        return self

    @abstractmethod
    def on_iter_end(self, **kwargs):
        """ !!. """


class LogCallback(BaseCallback):
    """
    1 - iteration and loss
    2 - resources
    3 - shapes
    4 - microbatch
    separate into arguments?
    """
    def __init__(self, stream=None, frequency=1, verbosity=2):
        self.frequency = frequency
        self.verbosity = verbosity
        self.header = self.make_header()

        super().__init__(stream=stream)
        self.stream(self.header)

    def make_header(self):
        """ Create the log description. """
        header = 'TIMESTAMP   ITERATION : LOSS'

        if self.verbosity >= 2:
            header += ' | MEMORY : GPU_MEMORY : GPU_UTILIZATION'
        if self.verbosity >= 3:
            header += ' | IN_SHAPES : OUT_SHAPES'
        if self.verbosity >= 4:
            header += ' | MICROBATCH_SIZE : N_MICROBATCHES'
        return header

    def on_iter_end(self, **kwargs):
        _ = kwargs
        i = self.model.iteration

        if i % self.frequency == 0 and self.verbosity >= 1:
            # Default message: timestamp, iteration and loss value
            timestamp = strftime('%Y-%m-%d  %H:%M:%S', gmtime())
            avg_loss = sum(self.model.loss_list[-self.frequency:]) / self.frequency

            msg = f'{timestamp}   {i:5}: {avg_loss:6.6f}'

            if self.verbosity >= 2:
                # Monitor resources
                memory = round(USSMonitor.get_usage(), 2)
                gpu_memory = GPUMemoryMonitor.get_usage()
                gpu_utilization = GPUMonitor.get_usage()

                msg += f' | {memory:6.3f} : '
                msg += ' : '.join([f'{round(item, 2):6.3f}' for item in gpu_memory]) + ' : '
                msg += ' : '.join([f'{round(item, 2):6.3f}' for item in gpu_utilization])

            if self.verbosity >= 3:
                # Model in/out shapes
                in_shapes = self.model.iter_info['actual_model_inputs_shape']
                out_shapes = self.model.iter_info['actual_model_outputs_shape']

                msg += f' | {in_shapes} : {out_shapes}'

            if self.verbosity >= 4:
                # Internal microbatch parameters of the last iteration
                microbatch = self.model.microbatch
                num_microbatches = self.model.iter_info['steps']

                msg += f' | {microbatch} : {num_microbatches}'

            self.stream(msg)
