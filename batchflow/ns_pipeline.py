""" Namespace pipeline """
import sys
from functools import partial
import numpy as np

from .named_expr import NamedExpression, eval_expr


class NamespacePipeline:
    """ Namespace pipeline allows declarative chains of methods from namespaces given """
    def __init__(self, pipeline=None, *namespaces):
        self.pipeline = pipeline
        self._namespaces = list(namespaces)
        self._actions = []

    @classmethod
    def concat(cls, pipe1, pipe2):
        """ Concatenate two pipelines """
        # pylint: disable=protected-access
        new_p = NamespacePipeline(pipe1.pipeline)
        new_p._actions = pipe1._actions + pipe2._actions
        new_p._namespaces = pipe1._namespaces + [a for a in pipe2._namespaces if a not in pipe1._namespaces]
        return new_p

    def __getstate__(self):
        state = dict(actions=self._actions, namespaces=self._namespaces, pipeline=self.pipeline)
        return state

    def __setstate__(self, state):
        self._actions = state['actions']
        self._namespaces = state['namespaces']
        self.pipeline = state['pipeline']

    def __add__(self, other):
        if isinstance(other, NamespacePipeline):
            return self.pipeline + other
        return other + self

    @property
    def _all_namespaces(self):
        return [sys.modules["__main__"]] + self._namespaces

    def has_method(self, name):
        return any(hasattr(namespace, name) for namespace in self._all_namespaces)

    def get_method(self, name):
        """ Return a method by the name """
        for namespace in self._all_namespaces:
            if hasattr(namespace, name):
                return getattr(namespace, name)

    def _add_action(self, name, *args, save_to=None, **kwargs):
        self._actions.append({'name': name, 'args': args, 'kwargs': kwargs, 'save_to': save_to})
        return self

    def __getattr__(self, name):
        if self.has_method(name):
            return partial(self._add_action, name)
        raise AttributeError("Unknown name: %s" % name)

    def add_namespace(self, *namespaces):
        self._namespaces.extend(namespaces)
        return self

    def _exec_action(self, action):
        args_value = eval_expr(action['args'], pipeline=self.pipeline)
        kwargs_value = eval_expr(action['kwargs'], pipeline=self.pipeline)

        method = self.get_method(action['name'])
        if method is None:
            raise ValueError("Unknown method: %s" % action['name'])

        res = method(*args_value, **kwargs_value)

        if isinstance(action['save_to'], NamedExpression):
            action['save_to'].set(res, pipeline=self.pipeline)
        elif isinstance(action['save_to'], np.ndarray):
            action['save_to'][:] = res

    def run(self):
        """ Execute all actions """
        for action in self._actions:
            self._exec_action(action)
        return self
