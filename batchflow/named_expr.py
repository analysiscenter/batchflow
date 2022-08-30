""" Contains named expression classes"""
import operator
from collections import defaultdict

from .config import Config
from .utils_random import make_rng


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.dataset = pipeline.dataset if pipeline is not None else None
        self.indices = self.dataset.indices if self.dataset is not None else None


def eval_expr(expr, no_eval=None, **kwargs):
    """ Evaluate a named expression recursively

    Parameters
    ----------
    expr
        an expression to evaluate

    no_eval : sequence of str
        a list of arguments not to evalute.
        Applicable only if expr is a dict which keys are checked against `no_eval` list.
    """
    no_eval = no_eval or []
    if isinstance(expr, NamedExpression):
        try:
            _expr = expr.get(**kwargs)
            if isinstance(expr, W):
                expr = _expr
            elif isinstance(_expr, (NamedExpression, list, tuple, dict, Config)):
                expr = eval_expr(_expr, **kwargs)
            else:
                expr = _expr
        except Exception as e:
            raise type(e)(f"Can't evaluate expression: {expr} because \n {str(e)}") from e
    elif isinstance(expr, (list, tuple)):
        _expr = []
        for val in expr:
            _expr.append(eval_expr(val, **kwargs))
        expr = type(expr)(_expr)
    elif isinstance(expr, (dict, Config)):
        if isinstance(expr, defaultdict):
            _expr = type(expr)(expr.default_factory)
        else:
            _expr = type(expr)()
        for key, val in expr.items():
            key = eval_expr(key, **kwargs)
            if key in no_eval:
                # save current params for later evaluation
                if isinstance(val, NamedExpression):
                    val.set_params(**kwargs)
            else:
                val = eval_expr(val, **kwargs)
            _expr.update({key: val})
        expr = _expr
    return expr


def swap(op):
    """ Swap args """
    def _op_(a, b):
        return op(b, a)
    return _op_


AN_EXPR = "#!__op__"

TERNARY_OPS = {
    '#slice': lambda a, b, c: slice(a, b, c),
    '#call': lambda a, b, c: a(*b, **c),
}

BINARY_OPS = {
    '__add__': operator.add, '__radd__': swap(operator.add), '__concat__': operator.concat,
    '__sub__': operator.sub, '__rsub__': swap(operator.sub),
    '__mul__': operator.mul, '__rmul__': swap(operator.mul),
    '__floordiv__': operator.floordiv, '__rfloordiv__': swap(operator.floordiv),
    '__truediv__': operator.truediv, '__rtruediv__': swap(operator.truediv),
    '__mod__': operator.mod, '__rmod__': swap(operator.mod),
    '__pow__': operator.pow, '__rpow__': swap(operator.pow),
    '__matmul__': operator.matmul, '__rmatmul__': swap(operator.matmul),
    '__lshift__': operator.lshift, '__rshift__': operator.rshift,
    '__and__': operator.and_, '__or__': operator.or_, '__xor__': operator.xor,
    '__lt__': operator.lt, '__le__': operator.le, '__gt__': operator.gt, '__ge__': operator.ge,
    '__eq__': operator.eq, '__ne__': operator.ne,
    '__is__': operator.is_, '__is_not__': operator.is_not,
    '#getitem': lambda a, b: a[b],
    '#item': lambda a, b: a[b],
    '#format': lambda a, b: b.format(a),
    '#attr': lambda a, b: getattr(a, b),
}

UNARY_OPS = {
    '__neg__': operator.neg, '__pos__': operator.pos, '__invert__': operator.inv, '__not__': operator.not_,
    '__abs__': operator.abs,
    '#str': str,
}

OPERATIONS = {**TERNARY_OPS, **BINARY_OPS, **UNARY_OPS}

OPERATIONS_SIGNS = {
    '__pos__': '+', '__neg__': '-', '__invert__': '~', '__concat__': '+',
    '__add__': '+', '__radd__': '+',
    '__sub__': '-', '__rsub__': '-',
    '__mul__': '*', '__rmul__': '*',
    '__floordiv__': '//', '__rfloordiv__': '//',
    '__truediv__': '/', '__rtruediv__': '/',
    '__mod__': '%', '__rmod__': '%',
    '__pow__': '**', '__rpow__': '**',
    '__matmul__': '@', '__rmatmul__': '@',
    '__lshift__': '>>', '__rshift__': '>>',
    '__and__': '&', '__or__': ' |', '__xor__': '^',
    '__lt__': '<', '__le__': '<=', '__gt__': '>', '__ge__': '>=',
    '__eq__': '==', '__ne__': '!=',
    '__is__': 'is', '__is_not__': 'is not', '__not__': 'not '
}

def add_ops(cls):
    """ Add arithmetic operations to a class.
    Allows to create and parse syntax trees using operations like '+', '-', '*', '/'.

    Parameters
    ----------
    op_cls : class
        The class which represents an arithmetics expression.
    """
    for op in OPERATIONS:
        if op[0] != '#':
            def _oper_(self, other=None, op=op):
                return AlgebraicNamedExpression(op=op, a=self, b=other)
            setattr(cls, op, _oper_)
    return cls


class MetaNamedExpression(type):
    """ Meta class to allow for easy instantiation through attribution

    Examples
    --------
    `B.images` is equal to B('images'), but requires fewer letters to type
    """
    def __getattr__(cls, name):
        return cls(name)

@add_ops
class NamedExpression(metaclass=MetaNamedExpression):
    """ Base class for a named expression

    Attributes
    ----------
    name : str
        a name
    mode : str
        a default assignment method: write, append, extend, update.
        Can be shrotened to jiust the first letter: w, a, e, u.

        - 'w' - overwrite with a new value. This is a default mode.
        - 'a' - append a new value
                (see list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
        - 'e' - extend with a new value
                (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists)
        - 'u' - update with a new value
                (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
                or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update)

    """
    __slots__ = ('__dict__', )

    def __init__(self, name=None, mode='w', **kwargs):
        self.name = name
        self.mode = mode
        self.params = None
        self.eval = eval
        self.set_params(**kwargs)

    def __getattr__(self, name):
        return AlgebraicNamedExpression(op='#attr', a=self, b=name)

    def __getitem__(self, key):
        if isinstance(key, slice) and any(isinstance(v, NamedExpression) for v in (key.start, key.stop, key.step)):
            key = AlgebraicNamedExpression(op='#slice', a=key.start, b=key.stop, c=key.step)
        return AlgebraicNamedExpression(op='#item', a=self, b=key)

    def __call__(self, *args, **kwargs):
        return AlgebraicNamedExpression(op='#call', a=self, b=args, c=kwargs)

    def str(self):
        """ Convert a named expression value to a string """
        return AlgebraicNamedExpression(op='#str', a=self)

    def format(self, string):
        """ Convert a value to a formatted representation, controlled by format spec.

        Examples
        --------
        Unlike Python built-in function, the usage is value.format(format_spec), for example:
        ::

            V('variable').format('Value of the variable is {:7.7}')
        """
        return AlgebraicNamedExpression(op='#format', a=self, b=string)

    def _get_params(self, **kwargs):
        """ Return parameters needed to evaluate the expression """
        if self.params is None:
            pkwargs = kwargs
        else:
            pkwargs = {}
            for arg in self.params.keys() | kwargs.keys():
                if self.params.get(arg) is None:
                    pkwargs[arg] = kwargs.get(arg)
                else:
                    # pre-set parameters should prevail
                    if isinstance(self.params.get(arg), NamedExpression):
                        pkwargs[arg] = self.params.get(arg).get(**kwargs)
                    else:
                        pkwargs[arg] = self.params.get(arg)

        if pkwargs.get('batch') is None:
            pkwargs['batch'] = _DummyBatch(pkwargs.get('pipeline'))

        name = self._get_name(**pkwargs)

        return name, pkwargs

    def set_params(self, **kwargs):
        self.params = kwargs

    def _get_name(self, **kwargs):
        if isinstance(self.name, NamedExpression):
            return eval_expr(self.name, **kwargs)
        return self.name

    def get(self, **kwargs):
        """ Return a value of a named expression

        Notes
        -----
        This method should be overriden in child classes.
        In the first line it should usually call `_get_params` method::

            name, kwargs = self._get_params(**kwargs)
        """
        raise NotImplementedError('Cannot get a value from an abstract named expression')

    def set(self, value, mode=None, eval=True, **kwargs):
        """ Set a value to a named expression

        Parameters
        ----------
        mode : str
            an assignment method: write, append, extend, update.
            A default mode may be specified when instantiating an expression.
        eval : bool
            whether to evaluate value before assigning it to the expression
            (as value might contain other named expressions,
            so it should be processed recursively)
        """
        params = self._get_params(**kwargs)
        kwargs = params[-1]
        mode = mode or self.mode

        if eval:
            value = eval_expr(value, **kwargs)
        if mode in ['a', 'append']:
            self.append(value, **kwargs)
        elif mode in ['e', 'extend']:
            self.extend(value, **kwargs)
        elif mode in ['u', 'update']:
            self.update(value, **kwargs)
        else:
            self.assign(value, **kwargs)

    def assign(self, value, **kwargs):
        """ Assign a value to a named expression """
        raise NotImplementedError("assign should be implemented in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression

        if a named expression is a dict or set, `update` is called, or `append` otherwise.

        See also
        --------
        list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
        dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update
        """
        var = self.get(*args, **kwargs)
        if var is None:
            self.assign(value, *args, **kwargs)
        elif isinstance(var, (set, dict)):
            var.update(value)
        else:
            var.append(value)

    def extend(self, value, *args, **kwargs):
        """ Extend a named expression with a new value
        (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        var = self.get(*args, **kwargs)
        if var is None:
            self.assign(value, *args, **kwargs)
        else:
            var.extend(value)

    def update(self, value, *args, **kwargs):
        """ Update a named expression with a new value
        (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update) """
        var = self.get(*args, **kwargs)
        if var is not None:
            var.update(value)
        else:
            self.assign(value, *args, **kwargs)

    def __repr__(self):
        return type(self).__name__ + '(' + str(self.name) + ')'

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getstate__(self):
        return self.__dict__

class AlgebraicNamedExpression(NamedExpression):
    """ Algebraic expression over named expressions """
    def __init__(self, op=None, a=None, b=None, c=None):
        super().__init__(AN_EXPR, mode='w')
        self.op = op
        self.a = a
        self.b = b
        self.c = c

    def get(self, **kwargs):
        """ Return a value of an algebraic expression """
        if self.op == '#call':
            kwargs.pop('_call', None)
            a = eval_expr(self.a, _call=False, **kwargs)
        else:
            a = eval_expr(self.a, **kwargs)
        b = eval_expr(self.b, **kwargs)
        c = eval_expr(self.c, **kwargs)
        if self.op in UNARY_OPS:
            return OPERATIONS[self.op](a)
        if self.op in BINARY_OPS:
            return OPERATIONS[self.op](a, b)
        return OPERATIONS[self.op](a, b, c)

    def assign(self, value, **kwargs):
        """ Assign a value to a named expression """
        if self.op not in ['#attr', '#item']:
            raise ValueError("Assigning a value to an arithmetic expression is not possible", self)

        _, kwargs = self._get_params(**kwargs)

        a = eval_expr(self.a, **kwargs)
        b = eval_expr(self.b, **kwargs)
        if self.op == '#attr':
            setattr(a, b, value)
        elif self.op == '#item':
            a[b] = value

    def __repr__(self):
        if self.op in OPERATIONS_SIGNS:
            if self.op in UNARY_OPS:
                return OPERATIONS_SIGNS[self.op] + repr(self.a)
            if self.op in BINARY_OPS:
                return repr(self.a) + ' ' + OPERATIONS_SIGNS[self.op] + ' ' + repr(self.b)

        if self.op == '__abs__':
            return '|' + repr(self.a) + '|'
        if self.op == '#str':
            return 'str(' + repr(self.a) + ')'
        if self.op == '#attr':
            return repr(self.a) + '.' + repr(self.b)[1:-1] # remove ''
        if self.op == '#item':
            return repr(self.a) + '[' + repr(self.b) +']'
        if self.op == '#format':
            a = repr(self.a) if self.a is not None else ''
            b = repr(self.b) if self.b is not None  else ''
            return 'f' + b + '.' + a
        if self.op == '#slice':
            a = repr(self.a) if self.a is not None else ''
            b = repr(self.b) if self.b is not None else ''
            c = ':' + repr(self.c) if self.c is not None else ''
            return a + ':' + b + c

        if self.op == '#call':
            args = ''
            if self.b is not None:
                args = repr(self.b)[1:-1]
                kwargs = ','.join([repr(k) + '=' + repr(v) for k,v in self.c.items()])
                args = args + ', ' + kwargs if args else kwargs
            return repr(self.a) + '(' + args + ')'

        return 'Unknown expression'


class B(NamedExpression):
    """ Batch component or attribute name

    Notes
    -----
    ``B()`` return the batch itself.

    To avoid unexpected data changes the copy of the batch may be returned, if ``copy=True``.

    Examples
    --------
    ::

        B('size')
        B('images_shape')
        B(copy=True)
    """
    def __init__(self, name=None, mode='w', copy=False):
        super().__init__(name, mode)
        self.copy = copy

    def _get_params(self, **kwargs):
        name, kwargs = super()._get_params(**kwargs)
        batch = kwargs['batch']
        return name, batch, kwargs

    def get(self, **kwargs):
        """ Return a value of a batch component """
        name, batch, _ = self._get_params(**kwargs)

        if isinstance(batch, _DummyBatch):
            raise ValueError(f"Batch expressions are not allowed in static models: B('{name}')")
        if name is None:
            return batch.copy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, **kwargs):
        """ Assign a value to a batch component """
        name, batch, _ = self._get_params(**kwargs)
        if name is not None:
            setattr(batch, name, value)


class L(B):
    """ List of objects or a batch component with a list of objects.

    Note
    ----
    ``L('comp').attr`` is equivalent to the list comprehension ``[val.attr for val in batch.comp]``.
    ``L('comp').func(*args, **kwargs)`` is equivalent to the list comprehension ``[val.func(*args, **kwargs)
                                                                                   for val in batch.comp]``.
    ``L('comp')[item]`` is equivalent to the list comprehension ``[val[item] for val in batch.comp]``.
    Any chains of consecutive calls of items or attribures like ``L('comp').attr[item].attr2 ... `` are also allowed.
    """
    def __init__(self, name=None, mode='w', **kwargs):
        super().__init__(name, mode)
        self.kwargs = kwargs

    def get(self, **kwargs):
        """ Returns an instance of the class that allows one to access attributes or items stored in the batch
        component or call a method from it.
        """
        name, batch, _ = self._get_params(**kwargs)

        # when given a component name, convert to a component data
        if isinstance(name, str):
            return L(getattr(batch, name))

        # expecting that name is a collection of items
        if 'attr' in self.kwargs:
            return [getattr(v, self.kwargs['attr']) for v in name]
        if 'item' in self.kwargs:
            return [v[eval_expr(self.kwargs['item'], **kwargs)] for v in name]
        if 'call' in self.kwargs:
            call_args, call_kwargs = self.kwargs['call']
            return [v(*eval_expr(call_args, **kwargs), **eval_expr(call_kwargs, **kwargs)) for v in name]
        return name

    def assign(self, value, **kwargs):
        """ Assign a value to batch component or item/attribute stored in the batch component """
        name, batch, _ = self._get_params(**kwargs)

        if 'attr' in self.kwargs:
            for n, v in zip(name ,value):
                setattr(n, self.kwargs['attr'], v)
        elif 'item' in self.kwargs:
            for n, v in zip(name, value):
                n[eval_expr(self.kwargs['item'], **kwargs)] = v
        else:
            # If value is assigned to the object itself it will be rewritten with `value`.
            setattr(batch, name, value)

    def __getattr__(self, name):
        return L(self, attr=name)

    def __getitem__(self, item):
        return L(self, item=item)

    def __call__(self, *args, **kwargs):
        return L(self, call=(args, kwargs))

    def __repr__(self):
        s = 'L(' + repr(self.name) + ')'
        if 'attr' in self.kwargs:
            return s + '.' + self.kwargs['attr']
        if 'item' in self.kwargs:
            return s + '[' + str(self.kwargs['item']) +']'
        if 'call' in self.kwargs:
            args, kwargs = self.kwargs['call']
            args = ', '.join(map(str, args)) if args else ''
            kwargs = ', '.join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ''
            args = args + ', ' + kwargs if kwargs else args
            return s + '(' + args + ')'
        return s


class PipelineNamedExpression(NamedExpression):
    #pylint: disable=abstract-method
    """ Base class for pipeline expressions """
    def _get_params(self, **kwargs):
        name, kwargs = super()._get_params(**kwargs)
        pipeline = kwargs.get('pipeline') if kwargs.get('pipeline') is not None else kwargs.get('batch').pipeline
        return name, pipeline, kwargs

class C(PipelineNamedExpression):
    """ A pipeline config option

    Notes
    -----
    ``C()`` return config itself.

    Examples
    --------
    Get a value from the current pipeline config::

        C('model_class', default=ResNet)
        C('GPU')

    Get the whole config from the current pipeline::

        C()

    Get a value from another pipeline config::

        C('model_class', pipeline=train_pipeline)
    """
    def __init__(self, name=None, mode='w', **kwargs):
        super().__init__(name, mode, **kwargs)
        self._has_default = 'default' in kwargs
        self.default = kwargs.get('default')

    def get(self, **kwargs):
        """ Return a value of a pipeline config """
        name, pipeline, _ = self._get_params(**kwargs)
        config = pipeline.config or Config()

        if name is None:
            return config
        try:
            if self._has_default:
                value = config.get(name, default=self.default)
            else:
                value = config[name]
        except KeyError:
            raise KeyError(f"Name is not found in the config: {name}") from None
        return value

    def assign(self, value, **kwargs):
        """ Assign a value to a pipeline config """
        name, pipeline, _ = self._get_params(**kwargs)
        pipeline.config[name] = value


class V(PipelineNamedExpression):
    """ Pipeline variable name

    Examples
    --------
    Get a variable value from the current pipeline::

        V('model_name')

    Get a variable value from another pipeline::

        V('loss_history', pipeline=train_pipeline)
    """
    def get(self, **kwargs):
        """ Return a value of a pipeline variable """
        name, pipeline, _ = self._get_params(**kwargs)
        value = pipeline.get_variable(name)
        return value

    def assign(self, value, **kwargs):
        """ Assign a value to a pipeline variable """
        name, pipeline, _ = self._get_params(**kwargs)
        pipeline.assign_variable(name, value)


class M(PipelineNamedExpression):
    """ Model name

    Examples
    --------
    Get a model from the current pipeline::

        M('model_name')

    Get a model from a given pipeline::

        M('model_name', pipeline=train_pipeline)

    Get a model from a pipeline specified in the current pipeline config::

        M('model_name', pipeline=C('train_pipeline'))

    Get a model from a pipeline specified in another pipeline config::

        M('model_name', pipeline=C('train_pipeline', pipeline=test_template))
    """
    def get(self, **kwargs):
        """ Return a model from a pipeline """
        name, pipeline, _ = self._get_params(**kwargs)
        value = pipeline.get_model_by_name(name)
        return value

    def assign(self, value, batch=None, pipeline=None):
        """ Assign a value to a model """
        _ = value, batch, pipeline
        raise ValueError('Assigning a value to a model is not possible.')


class I(PipelineNamedExpression):
    """ Iteration counter

    Parameters
    ----------
    name : str
        Determines returned value. One of:
            - 'current' or its substring - current iteration number, default.
            - 'maximum' or 'total' or their substring - total number of iterations to be performed.
              If total number is not defined, raises an error.
            - 'ratio' or its substring - current iteration divided by a total number of iterations.

    Raises
    ------
    ValueError
    If `name` is not valid.
    If `name` is 'm' or 'r' and total number of iterations is not defined.

    Examples
    --------
    ::

        I('current')
        I('max')
        I('max')
        R('normal', loc=0, scale=I('ratio')*100)
    """
    def __init__(self, name='c', mode='w', **kwargs):
        super().__init__(name, mode=None, **kwargs)

    def get(self, **kwargs):
        # pylint:disable=protected-access
        """ Return current or maximum iteration number or their ratio """
        name, pipeline, kwargs = self._get_params(**kwargs)

        current_iter = kwargs['batch'].iteration or pipeline.iter_params.get('_n_iters')

        if 'current'.startswith(name):
            return current_iter

        total = pipeline.iter_params.get('_total') # if pipeline.iter_params else None

        if 'maximum'.startswith(name) or 'total'.startswith(name):
            return total

        if total is None:
            raise ValueError('Total number of iterations is not defined!')

        if 'ratio'.startswith(name):
            ratio = current_iter / total
            return ratio

        raise ValueError(f'Unknown key for named expresssion I: {name}')

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to an iteration number is not supported")


class R(PipelineNamedExpression):
    """ A random value

    Parameters
    ----------
    name : str
        a distribution name

    seed : int, SeedSequence, Generator, BitGenerator, RandomState
        a random state (see :func:`~.make_rng`)

    args, kwargs
        distribution parameters

    Notes
    -----
    If `size` is needed, it should be specified as a named, not a positional argument.

    Examples
    --------
    ::

        R('normal', 0, 1)
        R('poisson', lam=5.5, seed=42, size=3)
        R(['metro', 'taxi', 'bike'], p=[.6, .1, .3], size=10)
    """
    def __init__(self, name, *args, seed=None, size=None, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs
        self.random = make_rng(seed)
        self.default_random = seed is None

        if not isinstance(size, (type(None), NamedExpression, int, tuple)):
            raise TypeError('size is expected to be int or tuple of int or a named expression')
        self.size = size

    def _get_params(self, **kwargs):
        name, pipeline, kwargs = super()._get_params(**kwargs)

        # if seed was explicitly set in R(...), use it
        # otherwise use the RNG from the pipeline if it exists
        if self.default_random and pipeline is not None and pipeline.random is not None:
            random = pipeline.random
        else:
            random = self.random

        return name, random, kwargs

    def get(self, size=None, **kwargs):
        """ Return a value of a random variable

        Parameters
        ----------
        size : int, tuple of int
            Output shape. If the given shape is (m, n, k), then m * n * k samples are drawn
            and returned as m x n x k array.
            If size was also specified at instance creation, then output shape is extended from the beginning.
            So `size` is treated like a batch size, while size specified at instantiation is an item size.

        Examples
        --------
        ::

            ne = R('normal', 0, 1, size=(10, 20)))
            value = ne.get(batch=batch)
            # value.shape will be (10, 20)

            value = ne.get(size=30, batch=batch)
            # value.shape will be (30, 10, 20)
            # so size is treated like a batch size
        """
        if not isinstance(size, (type(None), int, tuple)):
            raise TypeError('size is expected to be int or tuple of int')

        name, random, kwargs = self._get_params(**kwargs)
        args = self.args

        if not isinstance(name, str):
            args = (name,) + args
            name = 'choice'
        if isinstance(name, str) and hasattr(random, name):
            name = getattr(random, name)
        else:
            raise TypeError('An expression should be an int, an iterable or a numpy distribution name',
                            name, random)

        args = eval_expr(args, **kwargs)

        size, kwsize = eval_expr((self.size, size), **kwargs)
        if kwsize is not None:
            if size is None:
                size = kwsize
            else:
                if isinstance(size, int):
                    size = (size,)
                if isinstance(kwsize, int):
                    kwsize = (kwsize,)
                size = kwsize + size

        rkwargs = {**self.kwargs, 'size': size}
        rkwargs = eval_expr(rkwargs, **kwargs)

        return name(*args, **rkwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a random variable is not supported")

    def __repr__(self):
        repr_str = 'R(' + str(self.name)
        if self.args:
            repr_str += ', ' + ', '.join(str(a) for a in self.args)
        if self.kwargs:
            repr_str += ', ' + str(self.kwargs)
        return repr_str + (', size=' + str(self.size) + ')' if self.size else ')')



class F(NamedExpression):
    """ A function, method or any other callable that might take arguments

    Examples
    --------
    ::

        F(MyBatch.rotate)(B(), angle=30)
        F(make_data)
        F(prepare_data)(batch=B(), item=10)

    """
    def get(self, _call=True, **kwargs):
        """ Return a value from a callable

        Parameters
        ----------
        _call : bool
            Whether to call name-function while evaluating the expression.
            Sometimes we might not want calling the func, e.g. when evaluating an F-expr within a call-expression
            F(func)(1, arg2=10), since we want to evaluate the whole expression.
        """
        name, _ = self._get_params(**kwargs)
        return name() if _call else name

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a callable is not supported")


class D(NamedExpression):
    """ Dataset attribute or dataset itself

    Examples
    --------
    ::

        D()
        D('classes')
        D('organization')
    """
    def _get_params(self, **kwargs):
        name, kwargs = super()._get_params(**kwargs)
        batch = kwargs['batch']
        dataset = batch.dataset or kwargs['batch'].pipeline.dataset
        if dataset is None:
            raise ValueError("Dataset is not set", self)
        return name, dataset, kwargs

    def get(self, **kwargs):
        """ Return a value of a dataset attribute """
        name, dataset, _ = self._get_params(**kwargs)

        if name is None:
            value = dataset
        elif hasattr(dataset, name):
            value = getattr(dataset, name)
        else:
            raise KeyError("Attribute does not exist in the dataset", name)
        return value

    def assign(self, value, **kwargs):
        """ Assign a value to a dataset attribute """
        name, dataset, _ = self._get_params(**kwargs)
        if name is None:
            raise ValueError('Assigning a value to D() is not possible.')
        setattr(dataset, name, value)


class W(NamedExpression):
    """ A wrapper which returns the wrapped named expression without evaluating it

    Examples
    --------
    ::

        W(V('variable'))
        W(B(copy=True))
        W(R('normal', 0, 1, size=B('size')))
    """
    def get(self, **kwargs):
        """ Return a wrapped named expression """
        if not isinstance(self.name, NamedExpression):
            raise ValueError(f"Named expressions is expected, but given {self.name}")
        self.name.set_params(**kwargs)
        return self.name

    def assign(self, value, **kwargs):
        """ Assign a value """
        _ = kwargs
        self.name = value # pylint: disable=attribute-defined-outside-init


class P(W):
    """ A wrapper for values passed to actions parallelized with @inbatch_parallel

    Examples
    --------
    Each image in the batch will be rotated at its own angle::

        pipeline
            .rotate(angle=P(R('normal', 0, 1)))

    Without ``P`` all images in the batch will be rotated at the same angle,
    as an angle is randomized across batches only::

        pipeline
            .rotate(angle=R('normal', 0, 1))

    To put it simply, ``R(...)`` is evaluated as ``R(..., size=batch.size)``.

    Generate 3 categorical random samples for each batch item::

        pipeline
            .calc_route(P(R(['metro', 'taxi', 'bike'], p=[.6, 0.1, 0.3], size=3))

    Generate a random number of random samples for each batch item::

        pipeline
            .some_action(P(R('normal', 0, 1, size=R('randint', 3, 8))))

    ``P`` works with arbitrary iterables too::

        pipeline
            .do_something(n=P([1, 2, 3, 4, 5]))

    The first batch item will get ``n=1``, the second ``n=2`` and so on.

    See also
    --------
    :func:`~.inbatch_parallel`
    """
    def _get_name(self, **kwargs):
        return self.name

    def get(self, *args, parallel=False, **kwargs):   # pylint:disable=arguments-differ
        """ Calculate and return a value of the expression """
        _ = args

        name, kwargs = self._get_params(**kwargs)
        batch = kwargs['batch']

        # it's called from the decorator, so values were pre-calculated, just return them
        if parallel:
            # However, we can still have some R-expressions, e.g. for probabilities
            if isinstance(self.name, R):
                return self.name.get(**kwargs, size=batch.size)
            return self.name

        # pre-calculate values to pass them into decorator which takes them one by one
        if isinstance(name, (R, AlgebraicNamedExpression)):
            values = name.get(**kwargs, size=batch.size)
        elif isinstance(name, NamedExpression):
            values = name.get(**kwargs)
        else:
            values = name

        if len(values) != len(batch):
            msg = f"{self} returned a value (len={len(values)}) which does not fit the batch size (len={len(batch)})"
            raise ValueError(msg)

        # return P-expr to be recognized by the decorator
        return P(values)

    def assign(self, value, **kwargs):
        """ Assign a value """
        _ = kwargs
        self.name = value # pylint: disable=attribute-defined-outside-init


class PP(P):
    """ A wrapper for single-value expressions passed to actions parallelized with @inbatch_parallel
    `PP(expr)` is essentialy `P([expr for _ in batch.indices])`

    Examples
    --------
    Each image in the batch will be rotated at its own angle::

        pipeline
            .rotate(angle=PP(F(get_single_angle)))

    as ``get_single_angle`` will be called ``batch.size`` times.

    ``R(...)`` will be evaluated only once within ``P(...)``, but many times within ``PP(...)``::

        pipeline
            .rotate(angle=PP(R('normal', 0, 1)))

    That is why ``P(R(...))`` is much more efficient than ``PP(R(...))``.

    However, ``PP`` is indispensable for shape-specific operations like ``@`` or broadcasting::

        pipeline
            .rotate(angle=PP(R('normal', R('normal', 50, 15, size=3), 15)))

    Internal ``R`` specifies a 3D angle mean and thus defines the shape.
    External ``R`` knows nothing about that shape and will throw an exception within ``P``,
    but it'll work fine within ``PP``.

    See also
    --------
    :func:`~.inbatch_parallel`
    :class:`~.P`
    """

    def get(self, *_, **kwargs):   # pylint:disable=arguments-differ
        """ Calculate and return a value of the expression """

        name, kwargs = self._get_params(**kwargs)
        batch = kwargs['batch']

        # pre-calculate values to pass them into decorator which takes them one by one
        values = [name.get(**kwargs) if isinstance(name, NamedExpression) else name for _ in batch.indices]

        # return P-expr to be recognized by the decorator
        return P(values)
