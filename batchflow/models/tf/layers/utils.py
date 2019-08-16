""" Common utility functions for all of TensorFlow layers. """
import inspect



def add_as_function(cls):
    """ Decorator for classes. Automatically adds functional interface for `call` method of class. """
    name = cls.__name__
    func_name = ''.join('_' + c.lower()
                        if (c.isupper() and (i != len(name)-1) and name[i+1].islower()) else c.lower()
                        for i, c in enumerate(name)).strip('_')

    def _func(inputs, *args, **kwargs):
        return cls(*args, **kwargs)(inputs)

    module = inspect.getmodule(inspect.stack()[1][0])#sys.modules[__name__]#__import__(__name__)
    setattr(module, func_name, _func)
    return cls


def get_channel_axis(data_format):
    return -1 if data_format == 'channels_last' else 1


def get_spatial_dim(inputs):
    return inputs.shape.ndims - 2
