""" Layer class for all the others to inherit from. """
import inspect



def add_as_function(cls):
    """ Decorator for classes. Automatically adds functional interface for `call` method of class.
    For example, `ConvBlock` class is transformed into `conv_block` function, while
    `Conv1DTranspose` class is transformed into `conv1d_transpose` function.
    """
    name = cls.__name__
    func_name = ''.join('_' + c.lower()
                        if (c.isupper() and (i != len(name)-1) and name[i+1].islower()) else c.lower()
                        for i, c in enumerate(name)).strip('_')

    def func(inputs, *args, **kwargs):
        # We also want to use `training` or `is_training` indicators as arguments for call
        call_args = [inputs]
        training = kwargs.get('training') or kwargs.get('is_training')
        if (training is not None) and (len(inspect.getfullargspec(cls.__call__)[0]) > 2):
            call_args.append(training)
        return cls(*args, **kwargs)(*call_args)
    func.__doc__ = cls.__doc__

    module = inspect.getmodule(inspect.stack()[1][0])
    setattr(module, func_name, func)
    return cls



class Classproperty:
    """ Adds property to the class, not to its instances. """
    def __init__(self, prop):
        self.prop = prop
    def __get__(self, obj, owner):
        return self.prop(owner)



class Layer:
    """ Base class for different layers.

    Allows to easily get list of parameters for the class,
    as well as getting dict of (param, value) pairs for instances.
    """
    #pylint: disable=not-an-iterable
    @Classproperty
    def params(cls):
        """ Returns initialization named arguments except for `self`. Property of a class itself. """
        _params = inspect.getfullargspec(cls.__init__)[0]
        _params.remove('self')
        return _params

    @property
    def params_dict(self):
        """ Returns named arguments and their values of the initialization. Property of an instance. """
        return {name: getattr(self, name) for name in self.params}
