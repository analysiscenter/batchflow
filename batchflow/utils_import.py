""" Utility functions for delayed / missing imports. """
import os
import sys
from importlib import import_module



class DelayedImport:
    """ Proxy class to postpone import until the first access.
    Useful, when the import takes a lot of time (PyTorch, TensorFlow, ahead-of-time compiled functions).

    Note that even access for module inspection (like its documentation or dir) would trigger module loading.

    Examples::
        >>> from .xxx import yyy                                                              # before
        >>> yyy = DelayedImport(module='.xxx', package=__name__, attribute='yyy')             # after

    Parameters
    ----------
    module : str
        Name of module to load.
    package : str, optional
        Anchor for resolving the package name. Used only for relative imports.
    attribute : str, optional
        Name of attribute to get from loaded module.
    help : str, optional
        Additional help on import errors.
    """
    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def __init__(self, module, package=None, attribute=None, help=None):
        self.module, self.package, self.attribute = module, package, attribute
        self.help = help
        self._loaded_module = None

    @property
    def loaded_module(self):
        """ Try loading the module at the first access. """
        if self._loaded_module is None:
            try:
                self._loaded_module = import_module(self.module, self.package)
                if self.attribute is not None:
                    self._loaded_module = getattr(self._loaded_module, self.attribute)
            except ImportError as e:
                if self.help:
                    raise ImportError(f"No module named '{self.module}'! {self.help}") from e
                raise

        return self._loaded_module

    def __dir__(self):
        return dir(self.loaded_module)

    @property
    def __doc__(self):
        return self.loaded_module.__doc__

    def __getattr__(self, name):
        if name != 'loaded_module':
            return getattr(self.loaded_module, name)
        return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        #pylint: disable=not-callable
        return self.loaded_module(*args, **kwargs)

    def __getitem__(self, key):
        return self.loaded_module[key]


def make_delayed_import(module, package=None, attribute=None, help=None):
    """ Make delayed import only if needed.
    Setting `BATCHFLOW_IMMEDIATE_IMPORT` environment variable to any value makes all imports immediate.
    """
    if module in sys.modules or os.environ.get('BATCHFLOW_IMMEDIATE_IMPORT', False):
        loaded_module = import_module(module, package)
        if attribute is not None:
            loaded_module = getattr(loaded_module, attribute)
        return loaded_module

    return DelayedImport(module=module, package=package, attribute=attribute, help=help)

def make_delayed_imports(module, package=None, attributes=tuple(), help=None):
    """ Helper function to create multiple delayed imports. """
    return [make_delayed_import(module=module, package=package, attribute=attribute, help=help)
            for attribute in attributes]



class MissingImport:
    """ Proxy to delay ImportError for modules with missing dependencies. """
    def __init__(self, module, package=None, attribute=None, help=None):
        self.module, self.package, self.attribute = module, package, attribute
        self.help = help

    def trigger_import_error(self):
        """ Trigger import error. """
        try:
            import_module(self.module, self.package)
        except ImportError as e:
            if self.help:
                raise ImportError(f"No module named '{self.module}'! {self.help}") from e
            raise

    def __getattr__(self, name):
        _ = name
        self.trigger_import_error()

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        self.trigger_import_error()

    def __getitem__(self, key):
        _ = key
        self.trigger_import_error()



def try_import(module, package='.', attribute=None, help=None):
    """ Try importing the module; otherwise, return a proxy that would raise ImportError on the first access. """
    try:
        loaded_module = import_module(module, package)
        if attribute is None:
            return loaded_module
        if isinstance(attribute, str):
            return getattr(loaded_module, attribute)
        return [getattr(loaded_module, attr) for attr in attribute]
    except ImportError:
        return MissingImport(module=module, package=package, attribute=attribute, help=help)
