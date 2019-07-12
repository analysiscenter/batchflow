""" Contains node-class of a syntax tree, builder of mathematical tokens (`sin`, `cos` and others),
tree-parser and helper-functions for plotting of syntax-trees.
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

try:
    import networkx as nx
except ImportError:
    pass



MATH_TOKENS = ['sin', 'cos', 'tan',
               'asin', 'acos', 'atan',
               'sinh', 'cosh', 'tanh',
               'asinh', 'acosh', 'atanh',
               'exp', 'log', 'pow',
               'sqrt', 'sign',
               ]

CUSTOM_TOKENS = ['D', 'P', 'V', 'C', 'R']

LABELS_MAPPING = {
    '__sub__': '-', '__rsub__': '-',
    '__mul__': '*', '__rmul__': '*',
    '__div__': '/', '__rdiv__': '/',
    '__truediv__': '/', '__rtruediv__': '/',
    '__add__': '+', '__radd__': '+',
    '__pow__': '^', '__rpow__': '^'
}


def add_binary_magic(cls):
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
    operators = list(LABELS_MAPPING.keys())

    for magic_name in operators:
        def magic(self, other, magic_name=magic_name):
            return cls(LABELS_MAPPING.get(magic_name), self, other)

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
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self._args = args
        self._kwargs = kwargs

    def __len__(self):
        return len(self._args)

    def __repr__(self):
        return tuple((self.name, *self._args, self._kwargs)).__repr__()

def get_num_parameters(form):
    """ Get number of unique parameters (created via `P` letter) in the passed form."""
    n_args = len(inspect.signature(form).parameters)
    tree = form(*[SyntaxTreeNode('_' + str(i)) for i in range(n_args)])
    return len(get_unique_parameters(tree))

def get_unique_parameters(tree):
    """ Get unique names of parameters-variables (those containing 'P' in its name) from a parse-tree.
    """
    # pylint: disable=protected-access
    if isinstance(tree, (int, float, str, tf.Tensor, tf.Variable)):
        return []
    if tree.name == 'P':
        return [tree._args[0]]
    if len(tree) == 0:
        return []

    result = []
    for arg in tree._args:
        result += get_unique_parameters(arg)

    return list(set(result))


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
    # parse namespaces-arg
    if module in ['tensorflow', 'tf']:
        namespaces = namespaces or [tf.math, tf, tf.nn]
        d_func = lambda f, x: tf.gradients(f, x)[0]
        v_func = tf_v
        c_func = tf_c
        p_func = tf_p
        r_func = tf_r
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
               'P': p_func,
               'V': v_func,
               'C': c_func,
               'R': r_func}

    method_ = letters.get(name) or fetch_method(name, namespaces)
    method = (lambda *args, **kwargs: SyntaxTreeNode(name, *args, **kwargs)
              if isinstance(args[0], SyntaxTreeNode) else method_(*args, **kwargs))
    return method

def fetch_method(name, modules):
    """ Get function from list of modules. """
    for module in modules:
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError('Cannot find method ' + name + ' in ' + ', '.join([module.__name__ for module in modules]))


def add_tokens(var_dict=None, postfix='__', module='tf', names=None, namespaces=None):
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



# TF implementations of custom letters
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

def tf_p(*args, **kwargs):
    """ Tensorflow implementation of `R` letter: controllable from the outside perturbation. """
    _ = kwargs
    if len(args) != 1:
        raise ValueError('`P` is reserved to create exactly one perturbation at a time. ')
    return tf.identity(args[0])

def tf_r(*args, **kwargs):
    """ Tensorflow implementation of `E` letter: dynamically generated random noise. """
    if len(args) > 2:
        raise ValueError('`R`')
    if len(args) == 2:
        inputs, scale = args
        shape = tf.shape(inputs)
    else:
        scale = args[0] if len(args) == 1 else 1
        try:
            points = tf_check_tensor('inputs', 'concat', ':0')
            shape = (tf.shape(points)[0], 1)
        except KeyError:
            shape = ()

    distribution = kwargs.pop('distribution', 'normal')

    if distribution == 'normal':
        noise = tf.random.normal(shape=shape, stddev=scale)
    if distribution == 'uniform':
        noise = tf.random.uniform(shape=shape, minval=-scale, maxval=scale)
    return noise

def tf_check_tensor(prefix=None, name=None, postfix='/_output:0'):
    """ Simple wrapper around `get_tensor_by_name`. """
    tensor_name = tf.get_variable_scope().name + '/' + prefix + '/' + name + postfix
    graph = tf.get_default_graph()
    tensor = graph.get_tensor_by_name(tensor_name)
    return tensor


# Drawing functions. Require graphviz
def make_unique_node(graph, name):
    """ Add as much postfix-'_' to `name` as necessary to make unique name for new node in `graph`.

    Parameters
    ----------
    graph : nx.Graph
        graph, for which the node is created.
    name : str
        name of new node.

    Returns
    -------
    Resulting name. Composed from `name` and possibly several '_'-characters.
    """
    if name not in graph:
        return name
    ctr = 1
    while True:
        name_ = name + '_' * ctr
        if name_ not in graph:
            return name_
        ctr += 1

def _build_graph(tree, graph, parent_name, labels):
    """ Recursive graph-builder. Util-function.
    """
    #pylint: disable=protected-access
    if isinstance(tree, (float, int)):
        return
    if len(tree) == 0:
        return

    for child in tree._args:
        if isinstance(child, (float, int)):
            child_name = make_unique_node(graph, str(np.round(child, 2)))
            labels.update({child_name: str(np.round(child, 2))})
        else:
            child_name = make_unique_node(graph, child.name)
            labels.update({child_name: LABELS_MAPPING.get(child.name, child.name)})

        graph.add_edge(parent_name, child_name)
        _build_graph(child, graph, child_name, labels)

def build_graph(tree):
    """ Build graph from a syntax tree.
    """
    # boundary case: trees with no children
    graph = nx.DiGraph()
    if isinstance(tree, (float, int)):
        graph.add_node(str(np.round(tree, 2)))
        return graph

    parent_name = LABELS_MAPPING.get(tree.name, tree.name)
    graph.add_node(parent_name)
    if len(tree) == 0:
        return graph

    # process generic trees
    labels = {parent_name: parent_name}
    _build_graph(tree, graph, parent_name, labels)

    return graph, labels
