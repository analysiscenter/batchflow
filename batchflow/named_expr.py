""" Contains named expression classes"""
import operator
from collections import OrderedDict
from functools import partial

import numpy as np


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline

def swap(op):
    """ Swap args """
    def _op_(a, b):
        return op(b, a)
    return _op_


AN_EXPR = "#!__op__"

BINARY_OPS = {
    '__add__': operator.add, '__radd__': swap(operator.add),
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
    '#slice': lambda a, b: a[b], '#format': lambda a, b: b.format(a),
    '#attr': lambda a, b: getattr(a, b), '#call': lambda a, b: a(*b[0], **b[1]),
}

UNARY_OPS = {
    '__neg__': operator.neg, '__pos__': operator.pos, '__abs__': operator.abs, '__invert__': operator.inv,
    '#str': str,
}


OPERATIONS = {**BINARY_OPS, **UNARY_OPS}


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
                return cls(AN_EXPR, op=op, a=self, b=other)
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
    param_names = 'batch', 'pipeline', 'model'

    def __init__(self, name, mode='w', op=None, a=None, b=None, op_class=None):
        self.name = name
        self.mode = mode
        self.op = op
        self.a = a
        self.b = b
        self.params = None
        self._call = False
        self.op_class = op_class or NamedExpression

    def __getattr__(self, name):
        return self.op_class(AN_EXPR, op='#attr', a=self, b=name)

    def __getitem__(self, key):
        return self.op_class(AN_EXPR, op='#slice', a=self, b=key)

    def __call__(self, *args, **kwargs):
        if isinstance(self, F):
            self._call = False
        return self.op_class(AN_EXPR, op='#call', a=self, b=(args, kwargs))

    def str(self):
        """ Convert a named expression value to a string """
        return self.op_class(AN_EXPR, op='#str', a=self)

    def format(self, string):
        """ Convert a value to a formatted representation, controlled by format spec.

        Examples
        --------
        Unlike Python built-in function, the usage is value.format(format_spec), for example:
        ::

            V('variable').format('Value of the variable is {:7.7}')
        """
        return NamedExpression(AN_EXPR, op='#format', a=self, b=string)

    @classmethod
    def default_kwargs(cls, **kwargs):
        return OrderedDict([(item, kwargs[item] if item in kwargs else None) for item in cls.param_names])

    def set_params(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("set_params don't work woth positional arguments")
        self.params = self.default_kwargs(**kwargs)

    def get(self, **kwargs):
        """ Return a value of a named expression

        Parameters
        ----------
        batch
            a batch which should be used to calculate a value
        pipeline
            a pipeline which should be used to calculate a value
            (might be omitted if batch is passed)
        model
            a model which should be used to calculate a value
            (usually omitted, but might be useful for F- and L-expressions)
        """
        params = self.params if self.params else self.default_kwargs(**kwargs)
        name = self._get_name(**params)
        if name == AN_EXPR:
            return self._get_value(**params)
        raise ValueError("Undefined value")

    def _get_name(self, **kwargs):
        if isinstance(self.name, NamedExpression):
            params = self.params if self.params else self.default_kwargs(**kwargs)
            return self.name.get(**params)
        return self.name

    def _get_value(self, **kwargs):
        kwargs = self.default_kwargs(**kwargs)
        if self.name == AN_EXPR:
            a = self.eval_expr(self.a, **kwargs)
            b = self.eval_expr(self.b, **kwargs)
            if self.op in UNARY_OPS:
                return OPERATIONS[self.op](a)
            return OPERATIONS[self.op](a, b)
        raise ValueError("Undefined value")

    @classmethod
    def _get_params(cls, batch=None, pipeline=None, model=None):
        batch = batch if batch is not None else _DummyBatch(pipeline)
        pipeline = pipeline if pipeline is not None else batch.pipeline
        return OrderedDict(batch=batch, pipeline=pipeline, model=model)

    @classmethod
    def eval_expr(cls, expr, *args, **kwargs):
        """ Evaluate a named expression recursively """
        if len(args) > 0 and len(kwargs) > 0:
            raise TypeError("All `eval_expr` parameters except `expr` must be positional or keywords.")
        if len(args) > 0:
            kwargs = dict(zip(cls.param_names, args))
        kwargs = cls.default_kwargs(**kwargs)
        args = cls._get_params(**kwargs)
        if isinstance(expr, NamedExpression):
            params_ = expr.params or {item: None for item in cls.param_names}
            args = {**params_, **args}

            _expr = expr.get(**args)
            if isinstance(expr, W):
                expr = _expr
            elif isinstance(_expr, NamedExpression):
                expr = cls.eval_expr(_expr, **args)
            else:
                expr = _expr
        elif isinstance(expr, (list, tuple)):
            _expr = []
            for val in expr:
                _expr.append(cls.eval_expr(val, **args))
            expr = type(expr)(_expr)
        elif isinstance(expr, dict):
            _expr = type(expr)()
            for key, val in expr.items():
                key = cls.eval_expr(key, **args)
                val = cls.eval_expr(val, **args)
                _expr.update({key: val})
            expr = _expr
        return expr

    def set(self, value, mode=None, eval=True, **kwargs):
        """ Set a value to a named expression

        Parameters
        ----------
        batch
            a batch which should be used to calculate a value
        pipeline
            a pipeline which should be used to calculate a value
            (might be omitted if batch is passed)
        model
            a model which should be used to calculate a value
            (usually omitted, but might be useful for F- and L-expressions)
        mode : str
            an assignment method: write, append, extend, update.
            A default mode may be specified when instantiating an expression.
        eval : bool
            whether to evaluate value before assigning it to the expression
            (as value might contain other named expressions,
            so it should be processed recursively)
        """
        params = self.params if self.params else self.default_kwargs(**kwargs)
        mode = mode or self.mode

        if eval:
            value = self.eval_expr(value, **params)
        if mode in ['a', 'append']:
            self.append(value, **params)
        elif mode in ['e', 'extend']:
            self.extend(value, **params)
        elif mode in ['u', 'update']:
            self.update(value, **params)
        else:
            self.assign(value, **params)

    def assign(self, value, **kwargs):
        """ Assign a value to a named expression """
        params = self.params if self.params else self.default_kwargs(**kwargs)
        a = self.eval_expr(self.a, **params)
        b = self.eval_expr(self.b, **params)

        if self.op == '#attr':
            setattr(a, b, value)
        elif self.op == '#slice':
            a[b] = value
        else:
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
        if isinstance(self.name, str) and self.name == AN_EXPR:
            val = "Arithmetic expression " + str(self.op) + " on " + repr(self.a)
            if self.op in BINARY_OPS:
                val += " and " + repr(self.b)
            return val
        return type(self).__name__ + '(' + str(self.name) + ')'

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __getstate__(self):
        return self.__dict__


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

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models: B('%s')" % name)
        if name is None:
            return batch.copy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)


class C(NamedExpression):
    """ A pipeline config option

    Notes
    -----
    ``C()`` return config itself.

    Examples
    --------
    ::

        C('model_class')
        C('GPU')
        C()
    """
    def __init__(self, name=None, mode='w'):
        super().__init__(name, mode)

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        if name is None:
            return config
        try:
            value = config[name]
        except KeyError:
            raise KeyError("Name is not found in the config: %s" % name) from None
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(NamedExpression):
    """ A function, method or any other callable that takes a batch or a pipeline and possibly other arguments

    Examples
    --------
    ::

        F(MyBatch.rotate)(angle=30)
        F(make_data)
        F(prepare_data)(115, item=10)

    Notes
    -----
    Take into account that the actual calls will look like `current_batch.rotate(angle=30)`,
    `make_data(current_batch)` and `prepare_data(current_batch, 115, item=10)`.
    """
    def __init__(self, name, mode='w', _pass=True, op_class=None):
        super().__init__(name, mode, op_class=op_class)
        self._pass = _pass
        self._call = True

    def get(self, batch=None, pipeline=None, model=None, *args, **kwargs):
        """ Return a value from a callable """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        args = []
        if self._pass:
            if isinstance(batch, _DummyBatch) or batch is None:
                _pipeline = batch.pipeline if batch is not None else pipeline
                args += [_pipeline]
            else:
                args += [batch]
            if model is not None:
                args += [model]
            name = partial(name, *args)
        return name() if self._call else name

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")


class L(F):
    """ A function, method or any other callable """
    def __init__(self, name, mode='w', op_class=None):
        super().__init__(name, mode=mode, _pass=False, op_class=op_class)


class V(NamedExpression):
    """ Pipeline variable name

    Examples
    --------
    ::

        V('model_name')
        V('loss_history')
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        value = pipeline.get_variable(name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.assign_variable(name, value, batch=batch)


class D(NamedExpression):
    """ Dataset attribute or dataset itself

    Examples
    --------
    ::

        D()
        D('classes')
        D('organization')
    """
    def __init__(self, name=None, mode='w'):
        super().__init__(name, mode)

    def _get_name_dataset(self, batch=None, pipeline=None, model=None):
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        pipeline = pipeline if pipeline is not None else batch.pipeline
        dataset = pipeline.dataset if pipeline is not None else None
        dataset = dataset or batch.dataset
        if dataset is None:
            raise ValueError("Dataset is not set", self)
        return name, dataset

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a dataset attribute """
        if self.params:
            batch, pipeline, model = self.params
        name, dataset = self._get_name_dataset(batch=batch, pipeline=pipeline, model=model)

        if name is None:
            value = dataset
        elif hasattr(dataset, name):
            value = getattr(dataset, name)
        else:
            raise KeyError("Attribute does not exist in the dataset", name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a dataset attribute """
        name, dataset = self._get_name_dataset(batch=batch, pipeline=pipeline, model=model)
        if name is None:
            raise ValueError('Assigning a value to D() is not possible.')
        setattr(dataset, name, value)


class R(NamedExpression):
    """ A random value

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
    def __init__(self, name, *args, state=None, seed=None, size=None, **kwargs):
        if not (callable(name) or isinstance(name, (str, NamedExpression))):
            args = (name,) + args
            name = 'choice'
        super().__init__(name)
        if isinstance(state, np.random.RandomState):
            self.random_state = state
        else:
            self.random_state = np.random.RandomState(seed)
        self.args = args
        self.kwargs = kwargs
        self.size = size

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a random variable """
        if self.params:
            batch, pipeline, model = self.params
        name = self._get_name(batch=batch, pipeline=pipeline, model=model)
        if callable(name):
            pass
        elif isinstance(name, str) and hasattr(self.random_state, name):
            name = getattr(self.random_state, name)
        else:
            raise TypeError('Random distribution should be a callable or a numpy distribution')
        args = self.eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        if self.size is not None:
            self.kwargs['size'] = self.size
        kwargs = self.eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)

        return name(*args, **kwargs)

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


class W(NamedExpression):
    """ A wrapper which returns the wrapped named expression without evaluating it

    Examples
    --------
    ::

        W(V('variable'))
        W(B(copy=True))
        W(R('normal', 0, 1, size=B('size')))
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a wrapped named expression """
        self.name.set_params(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a wrapper is not supported")


class P(W):
    """ A wrapper for actions parallelized with @inbatch_parallel

    Examples
    --------
    Each image in the batch will be rotated at its own angle::

        pipeline
            .rotate(angle=P(R('normal', 0, 1)))

    Without ``P`` all images in the batch will be rotated at the same angle,
    as an angle is randomized across batches only::

        pipeline
            .rotate(angle=R('normal', 0, 1))

    Generate 3 categorical random samples for each batch item::

        pipeline
            .calc_route(P(R(['metro', 'taxi', 'bike'], p=[.6, 0.1, 0.3], size=3))

    Generate random number of random samples for each batch item::

        pipeline
            .some_action(P(R('normal', 0, 1, size=R('randint', 3, 8))))

    ``P`` works with arbitrary iterables too::

        pipeline
            .do_something(n=P([1, 2, 3, 4, 5]))

    The first batch item will get ``n=1``, the second ``n=2`` and so on.

    See also
    --------
    :func:`~batchflow.inbatch_parallel`
    """
    def get(self, batch=None, pipeline=None, model=None, parallel=False):   # pylint:disable=arguments-differ
        """ Return a wrapped named expression """
        if parallel:
            if isinstance(self.name, R):
                val = np.array([self.name.get(batch=batch, pipeline=pipeline, model=model) for _ in batch])
            elif isinstance(self.name, NamedExpression):
                val = self.name.get(batch=batch, pipeline=pipeline, model=model)
            else:
                val = self.name
            if len(val) < len(batch):
                raise ValueError('%s returns a value (len=%d) which does not fit the batch size (len=%d)'
                                 % (self, len(val), len(batch)))
            return val
        return self

class I(NamedExpression):
    """ Iteration counter

    Parameters
    ----------
    name : str
        'current' or its substring - current iteration number, default.
        'maximum' or its substring - total number of iterations to be performed.
                                     If total number is not defined, raises an error.
        'ratio' or its substring - current iteration divided by a total number of iterations.

    Raises
    ------
    ValueError
        * If `name` is not valid.
        * If `name` is 'm' or 'r' and total number of iterations is not defined.
    Examples
    --------
    ::

        I('current')
        I('max')
        R('normal', loc=0, scale=I('ratio')*100)
    """
    def __init__(self, name='c'):
        super().__init__(name, mode=None)

    def get(self, batch=None, pipeline=None, model=None):    # pylint:disable=inconsistent-return-statements
        """ Return current or maximum iteration number or their ratio """
        name = self._get_name(batch, pipeline, model)

        pipeline = batch.pipeline if batch is not None else pipeline
        if 'current'.startswith(name):
            return pipeline._iter_params['_n_iters']    # pylint:disable=protected-access

        total = pipeline._iter_params.get('_total')    # pylint:disable=protected-access

        if 'maximum'.startswith(name):
            return total
        if 'ratio'.startswith(name):
            if total is None:
                raise ValueError('Total number of iterations is not defined!')
            ratio = pipeline._iter_params['_n_iters'] / total    # pylint:disable=protected-access
            return ratio

        raise ValueError('Unknown key for named expresssion I: %s' % name)

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to an iteration number is not supported")
