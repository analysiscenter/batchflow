""" Namespace pipeline """
from functools import partial
import numpy as np

from .named_expr import NamedExpression, eval_expr


class NamespacePipeline:
    """ Namespace pipeline allows declarative chains of methods from namespaces given """
    def __init__(self, pipeline=None, *namespaces):
        self.pipeline = pipeline
        self._namespaces = list(namespaces)
        self._actions = []

    def __add__(self, other):
        if isinstance(other, NamespacePipeline):
            return self.pipeline + other
        return other + self

    def has_method(self, name):
        return any(hasattr(namespace, name) for namespace in self._namespaces)

    def get_method(self, name):
        """ Return a method by the name """
        for namespace in self._namespaces:
            if hasattr(namespace, name):
                return getattr(namespace, name)

    def _add_action(self, method, *args, save_to=None, **kwargs):
        self._actions.append({'method': method, 'args': args, 'kwargs': kwargs,
                              'save_to': save_to})
        return self

    def __getattr__(self, name):
        method = self.get_method(name)
        if method is None:
            raise AttributeError("Unknown method: %s" % name)
        return partial(self._add_action, method)

    def add_namespace(self, *namespaces):
        self._namespaces.extend(namespaces)
        return self

    def _exec_action(self, action):
        args_value = eval_expr(action['args'], pipeline=self.pipeline)
        kwargs_value = eval_expr(action['kwargs'], pipeline=self.pipeline)
        res = action['method'](*args_value, **kwargs_value)

        if isinstance(action['save_to'], NamedExpression):
            action['save_to'].set(res, pipeline=self.pipeline)
        elif isinstance(action['save_to'], np.ndarray):
            action['save_to'][:] = res

    def run(self):
        """ Execute all actions """
        for action in self._actions:
            self._exec_action(action)
        return self
