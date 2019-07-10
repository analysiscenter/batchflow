""" Contains node-class of a syntax tree, builder of mathematical tokens (`sin`, `cos` and others)
and tree-parser.
"""
import inspect

import numpy as np
import tensorflow as tf

from .tf.layers import conv_block

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
        return tuple((self.name, *self._args, self._kwargs)).__repr__()


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
    if isinstance(tree, (int, float, str)):
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
    # pylint: disable=protected-access
    if isinstance(tree, (int, float, str)):
        return []
    if 'R' in tree.name:
        return [tree.name]
    if len(tree) == 0:
        return []

    result = []
    for arg in tree._args:
        result += get_unique_perturbations(arg)
    return list(np.unique(result))


MATH_TOKENS = ['sin', 'cos', 'tan',
               'asin', 'acos', 'atan',
               'sinh', 'cosh', 'tanh',
               'asinh', 'acosh', 'atanh',
               'exp', 'log', 'pow',
               'sqrt', 'sign',
               ]

CUSTOM_TOKENS = ['D', 'R', 'V', 'C']


def make_token(module='tf', name=None, namespaces=None):
    """ Make a mathematical tokens.

    Parameters
    ----------
    module : str
        Can be 'np' (stands for `numpy`) or 'tf'(stands for `tensorflow`). Either choice binds tokens to
        correspondingly named operations from a module. For instance, token 'sin' for module 'np' stands for
        operation `np.sin`.
    name : str
        name of module function used for binding tokens.

    Returns
    -------
    callable
        Function that can be applied to a parse-tree, adding another node in there.
    """
    # pylint: disable=protected-access, unused-variable
    # parse namespaces-arg
    if module in ['tensorflow', 'tf']:
        namespaces = namespaces or [tf.math, tf, tf.nn]
        d_func = lambda f, x: tf.gradients(f, x)[0]
        v_func = tf_v
        c_func = tf_c
    elif module in ['numpy', 'np']:
        namespaces = namespaces or [np, np.math]
        if name == 'D':
            namespaces = namespaces or [autonp, autonp.math]
            d_func = lambda f, x: grad(f)(x)
    elif module == 'torch':
        raise NotImplementedError('Torch is not implemented yet.')

    # None of the passed modules are supported
    if namespaces is None:
        raise ValueError('Module ' + module + ' is not supported: you should directly pass namespaces-arg!')

    # make method
    letters = {'D': d_func,
               'V': v_func,
               'C': c_func,
               'R': lambda x: x}

    method = letters.get(name) or fetch_method(name, namespaces)

    # make the token
    # def token(*args, **kwargs):
    #     if name == 'R':
    #         # Just pass all the arguments to the next Node
    #         return SyntaxTreeNode(args[0].method, *args[0]._args, name='R_' + args[0].name, **args[0]._kwargs)
    #     if name in  ['V', 'C']:
    #         # Use both args and kwargs for method call
    #         return SyntaxTreeNode(lambda *args: method(*args, **kwargs), *args, name=name)
    #     # Use args for method call, pass kwargs to the next Node
    #     return SyntaxTreeNode(method, *args, name=name, **kwargs)
    return method

def fetch_method(name, modules):
    """ Get function from list of modules. """
    for module in modules:
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError('Cannot find method ' + name + ' in ' + ', '.join([module.__name__ for module in modules]))



# Tf implementations of custom letters
def tf_v(*args, prefix='addendums', **kwargs):
    """ Tensorflow implementation of `V` letter: adjustable variation of the coefficient. """
    # Parsing arguments
    _ = kwargs
    *args, name = args
    if not isinstance(name, str):
        raise ValueError('`W` last positional argument should be its name. Instead got {}'.format(name))
    if len(args) > 1:
        raise ValueError('`W` can work only with one initial value. ')
    x = args[0] if len(args) == 1 else 0.0

    # Try to get already existing variable with the given name from current graph.
    # If it does not exist, create one
    try:
        var = tf_check_tensor(prefix, name)
        return var
    except KeyError:
        var_name = prefix + '/' + name
        var = tf.Variable(x, name=var_name, dtype=tf.float32, trainable=True)
        var = tf.identity(var, name=var_name + '/_output')
        return var

def tf_c(*args, prefix='addendums', **kwargs):
    """ Tensorflow implementation of `C` letter: small neural network inside equation. """
    *args, name = args
    if not isinstance(name, str):
        raise ValueError('`C` last positional argument should be its name. Instead got {}'.format(name))

    defaults = dict(layout='faf',
                    units=[15, 1],
                    activation=tf.nn.tanh)
    kwargs = {**defaults, **kwargs}

    try:
        block = tf_check_tensor(prefix, name)
        return block
    except KeyError:
        block_name = prefix + '/' + name
        points = tf.concat(args, axis=-1, name=block_name + '/concat')
        block = conv_block(points, name=block_name, **kwargs)
        return block

def tf_check_tensor(prefix, name):
    """ Simple wrapper around `get_tensor_by_name`. """
    tensor_name = tf.get_variable_scope().name + '/' + prefix + '/' + name + '/_output:0'
    graph = tf.get_default_graph()
    tensor = graph.get_tensor_by_name(tensor_name)
    return tensor

def add_tokens(var_dict=None, postfix='__', module='tf',
               names=None, namespaces=None):
    """ Add tokens to passed namespace.

    Parameters
    ----------
    var_dict : dict
        Namespace to add names to. Default values is the namespace from which the function is called.
    postfix : str
        If the passed namespace already contains item with the same name, then
        postfix is appended to the name to avoid naming collision.
    module : str
        Can be 'np' (stands for `numpy`) or 'tf'(stands for `tensorflow`). Either choice binds tokens to
        correspondingly named operations from a module. For instance, token 'sin' for module 'np' stands for
        operation `np.sin`.
    names : str
        Names of function to be tokenized from the given module.

    Notes
    -----
    This function is also called when anything from this module is imported inside
    executable code (e.g. code where __name__ = __main__).
    """
    names = names or (MATH_TOKENS + CUSTOM_TOKENS)

    if not var_dict:
        frame = inspect.currentframe()
        try:
            var_dict = frame.f_back.f_locals
        finally:
            del frame

    for name in names:
        token = make_token(module=module, name=name, namespaces=namespaces)
        if name not in var_dict:
            name_ = name
        else:
            name_ = name + postfix
            msg = 'Name `{}` already present in current namespace. Added as {}'.format(name, name+postfix)
            print(msg)
        var_dict[name_] = token
