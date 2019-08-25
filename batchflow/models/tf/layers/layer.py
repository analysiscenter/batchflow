""" Layer class for all the others to inherit from. """
import inspect


class Classproperty:
    """ That will become the greatest docstring of all times"""
    def __init__(self, prop):
        self.prop = prop
    def __get__(self, obj, owner):
        return self.prop(owner)



class Layer:
    """ That will become the greatest docstring of all times"""
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
