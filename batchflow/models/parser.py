""" Contains node-class of a syntax tree, builder of mathematical tokens (`sin`, `cos` and others)
and tree-parser.
"""
import numpy as np
import tensorflow as tf

try:
    from autograd import grad
    import autograd.numpy as autonp

except ImportError:
    pass


def add_binary_magic(cls, operators=('__add__', '__radd__', '__mul__', '__rmul__', '__sub__', '__rsub__',
                                     '__truediv__', '__rtruediv__', '__pow__', '__rpow__')):
    """ Add binary-magic operators to `SyntaxTreeNode`-class. Allows to create and parse syntax trees
    using binary operations like '+', '-', '*', '/'.

    Parameters
    ----------
    cls : class
        The class to be processed by the decorator.
    operators : sequence
        Sequence of magic-method names to be added to `cls`.

    Returns
    -------
    modified class.
    """
    for magic_name in operators:
        def magic(self, other, magic_name=magic_name):
            return cls(lambda x, y: getattr(x, magic_name)(y), self, other, name=magic_name)

        setattr(cls, magic_name, magic)
    return cls


@add_binary_magic
class SyntaxTreeNode():
    """ Node of parse tree. Stores operation representing the node along with its arguments.

    Parameters
    ----------
    name : str
        name of the node. Used for creating a readable string-repr of a tree.
    *args:
        args[0] : method representing the node of parse tree.
        args[1:] : arguments of the method.
    """
    def __init__(self, *args, name=None, **kwargs):
        arg = args[0]
        if isinstance(arg, str):
            if len(arg) == 1:
                nums_of_args = {'u': 0, 'x': 1, 'y': 2, 'z': 3, 't': -1}
                if arg in nums_of_args:
                    self.method = lambda *args: args[nums_of_args[arg]]
                else:
                    raise ValueError("Cannot parse variable-number from " + arg)
            else:
                try:
                    var_num = int(arg[1:])
                    self.method = lambda *args: args[var_num]
                except ValueError:
                    raise ValueError("Cannot parse variable-number from " + arg)
            self.name = arg
        elif callable(arg):
            self.method = arg
            self.name = name
        else:
            raise ValueError("Cannot create a NodeTree-instance from ", *args)
        self._args = args[1:]
        self._kwargs = kwargs

    def __len__(self):
        return len(self._args)

    def __repr__(self):
        return tuple((self.name, *self._args)).__repr__()


def parse(tree):
    """ Make the method (callable) represented by a parse-tree.

    Parameters
    ----------
    tree : SyntaxTreeNode
        instance of node-class representing the tree.

    Returns
    -------
    resulting callable.
    """
    if isinstance(tree, (int, float)):
        # constants
        return lambda *args: tree

    def result(*args):
        # pylint: disable=protected-access
        if len(tree) > 0:
            all_args = [parse(operand)(*args) for operand in tree._args]
            return tree.method(*all_args)
        return tree.method(*args)
    return result


def get_unique_perturbations(tree):
    """ Get unique names of perturbation-variables (those containing 'R' in its name) from a parse-tree.
    """
    if isinstance(tree, (int, float)):
        return []
    if 'R' in tree.name:
        return [tree.name]

    if len(tree) == 0:
        return []

    result = []
    for arg in tree._args:                          # pylint: disable=protected-access
        result += get_unique_perturbations(arg)

    return list(np.unique(result))


def make_tokens(module='tf', names=('sin', 'cos', 'exp', 'log', 'tan', 'acos', 'asin', 'atan',
                                    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'D', 'R'),
                namespaces=None, grad_func=None):
    """ Make a collection of mathematical tokens.

    Parameters
    ----------
    module : str
        Can be 'np' (stands for `numpy`) or 'tf'(stands for `tensorflow`). Either choice binds tokens to
        correspondingly named operations from a module. For instance, token 'sin' for module 'np' stands for
        operation `np.sin`.
    names : sequence
        names of module-funcs used for binding tokens.

    Returns
    -------
    Sequnce of tokens - callables, that can be applied to a parse-tree adding another node in there.
    """
    # parse namespaces-arg
    if module in ['tensorflow', 'tf']:
        namespaces = namespaces or [tf.math, tf, tf.nn]
        grad_func = lambda f, x: tf.gradients(f, x)[0]
    elif module == 'torch':
        raise NotImplementedError('Torch not implemented yet.')
    elif module in ['numpy', 'np']:
        namespaces = namespaces or [np, np.math]
        if 'D' in names:
            namespaces = namespaces or [autonp, autonp.math]
            grad_func = lambda f, x: grad(f)(x)
    else:
        if namespaces is None:
            raise ValueError('Module ' + module + ' is not supported: you should directly pass namespaces-arg!')

    def _fetch_method(name, modules):
        for module in modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise ValueError('Cannot find method ' + name + ' in ' + [str(module) for module in modules].join(', '))

    # fill up tokens-list
    tokens = []
    for name in names:
        # make the token-method
        # pylint: disable=unused-variable
        if name == 'D':
            method = grad_func
        elif name == 'R':
            pass
        else:
            method = _fetch_method(name, namespaces)

        # make the token
        if name == 'R':
            def token(*args, name=name):
                """ Token for PDE-perturbations.
                """
                return SyntaxTreeNode(args[0].method, *args[0]._args, name='R_' + args[0].name, **args[0]._kwargs)      # pylint: disable=protected-access
        else:
            token = lambda *args, method=method, name=name: SyntaxTreeNode(method, *args, name=name)

        tokens.append(token)

    return tokens
