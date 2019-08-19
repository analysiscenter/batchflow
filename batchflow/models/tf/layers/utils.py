""" Common utility functions for all of TensorFlow layers. """
import inspect



def add_as_function(cls):
    """ Decorator for classes. Automatically adds functional interface for `call` method of class. """
    name = cls.__name__
    func_name = ''.join('_' + c.lower()
                        if (c.isupper() and (i != len(name)-1) and name[i+1].islower()) else c.lower()
                        for i, c in enumerate(name)).strip('_')

    def func(inputs, *args, **kwargs):
        call_args = []
        training = kwargs.get('training') or kwargs.get('is_training')
        if training is not None:
            call_args = [training]
        return cls(*args, **kwargs)(inputs, *call_args)

    module = inspect.getmodule(inspect.stack()[1][0])
    setattr(module, func_name, func)
    return cls
