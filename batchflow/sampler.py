# pylint: disable=too-few-public-methods, method-hidden
""" Contains Sampler-classes. """

import warnings
from copy import copy
import numpy as np
try:
    import scipy.stats as ss
except ImportError:
    pass

from .utils_random import make_rng


# aliases for Numpy, Scipy-Stats, TensorFlow-samplers
ALIASES = {
    'n': {'np': 'normal', 'ss': 'norm', 'tf': 'Normal'},
    'u': {'np': 'uniform', 'ss': 'uniform', 'tf': 'Uniform'},
    'mvn': {'np': 'multivariate_normal', 'ss': 'multivariate_normal'},
    'e': {'np': 'exponential', 'ss': 'expon', 'tf': 'Exponential'},
    'g': {'np': 'gamma', 'ss': 'gamma', 'tf': 'Gamma'},
    'be' : {'np': 'beta', 'ss': 'beta', 'tf': 'Beta'},
    'mnm': {'np': 'multinomial', 'ss': 'multinomial', 'tf': 'Multinomial'},
    'f': {'np': 'f', 'ss': 'f'},
    'p': {'np': 'poisson', 'ss': 'poisson'},
    'w': {'np': 'weibull', 'ss': 'dweibull'},
    'ln': {'np': 'lognormal', 'ss': 'lognorm'},
    'b' : {'np': 'binomial', 'ss': 'binom'},
    'chi2': {'np': 'chisquare', 'ss': 'chi2'},
    'c': {'np': 'choice'}
}

def _get_method_by_alias(alias, module, tf_distributions=None):
    """ Fetch fullname of a randomizer from ``scipy.stats``, ``tensorflow`` or
    ``numpy`` by its alias or fullname.
    """
    rnd_submodules = {'np': np.random,
                      'tf': tf_distributions,
                      'ss': ss}
    # fetch fullname
    fullname = ALIASES.get(alias, {module: alias for module in ['np', 'tf', 'ss']}).get(module, None)
    if fullname is None:
        raise ValueError(f"Distribution {alias} has no implementaion in module {module}")

    # check that the randomizer is implemented in corresponding module
    if not hasattr(rnd_submodules[module], fullname):
        raise ValueError(f"Distribution {fullname} has no implementaion in module {module}")

    return fullname


def arithmetize(cls):
    """ Add arithmetic operations to Sampler-class.
    """
    for oper in ['__add__', '__mul__', '__truediv__', '__sub__', '__pow__', '__floordiv__', '__mod__',
                 '__radd__', '__rmul__', '__rtruediv__', '__rsub__', '__rpow__', '__rfloordiv__', '__rmod__']:
        def transform(self, other, fake=oper):
            """ Arithmetic operation on couple of Samplers.

            Implemented via corresponding operation in ndarrays.

            Parameters
            ----------
            other : Sampler
                second Sampler, the operation is applied to.

            Returns
            -------
            Sampler
                resulting sampler.
            """
            _class = classes[fake]
            return _class(self, other)
        setattr(cls, oper, transform)

    return cls

@arithmetize
class Sampler():
    """ Base class Sampler that implements algebra of Samplers.

    Attributes
    ----------
    weight : float
        weight of Sampler self in mixtures.
    """
    def __init__(self, *args, **kwargs):
        self.__array_priority__ = 100
        self.weight = 1.0

        # if dim is supplied, redefine sampling method
        if 'dim' in kwargs:
            # assemble stacked sampler
            dim = kwargs.pop('dim')
            stacked = type(self)(*args, **kwargs)
            for _ in range(dim - 1):
                stacked = type(self)(*args, **kwargs) & stacked

            # redefine sample of self
            self.sample = stacked.sample

    def sample(self, size):
        """ Sampling method of a sampler.

        Parameters
        ----------
        size : int
            lentgh of sample to be generated.

        Returns
        -------
        np.ndarray
            Array of size (len, Sampler's dimension).
        """
        raise NotImplementedError('The method should be implemented in child-classes!')

    def __or__(self, other):
        """ Implementation of '|' operation for two instances of Sampler-class.

        The result is the mixture of two samplers. Weights are taken from
        samplers' weight-attributes.

        Parameters
        ----------
        other : Sampler
            the sampler to be added to self.

        Returns
        -------
        Sampler
            resulting mixture of two samplers.
        """
        return OrSampler(self, other)

    def __and__(self, other):
        """ Implementation of '&' operation for instance of Sampler-class.

        Two cases are possible: if ``other`` is numeric, then "&"-operation changes
        the weight of a sampler. Otherwise, if ``other`` is also a Sampler, the resulting
        Sampler is a multidimensional sampler, with starting coordinates being sampled from
        ``self``, and trailing - from ``other``.

        Parameters
        ----------
        other : int or float or Sampler
            the sampler/weight for multiplication.

        Returns
        -------
        Sampler
            result of the multiplication.
        """
        if isinstance(other, (float, int)):
            self.weight *= other
            return self

        return AndSampler(self, other)

    def __rand__(self, other):
        """ Implementation of '&' operation on a weight for instance of Sampler-class.
        see docstring of Sampler.__and__.
        """
        return self & other

    def apply(self, transform):
        """ Apply a transformation to the sampler.
        Build new sampler, which sampling function is given by `transform(self.sample(size))``.

        Parameters
        ----------
        transform : callable
            function, that takes ndarray of shape (size, dim_sampler) and produces
            ndarray of shape (size, new_dim_sampler).

        Returns
        -------
        Sampler
            instance of class Sampler with redefined method `sample`.
        """
        return ApplySampler(self, transform)

    def truncate(self, high=None, low=None, expr=None, prob=0.5, max_iters=None, sample_anyways=False):
        """ Truncate a sampler. Resulting sampler produces points satisfying ``low <= pts <= high``.
        If ``expr`` is suplied, the condition is ``low <= expr(pts) <= high``.

        Uses while-loop to obtain a sample from the region of interest of needed size. The behaviour
        of the while loop is controlled by parameters ``max_iters`` and ``sample_anyways``-parameters.

        Parameters
        ----------
        high : ndarray, list, float
            upper truncation-bound.
        low : ndarray, list, float
            lower truncation-bound.
        expr : callable, optional.
            Some vectorized function. Accepts points of sampler, returns either bool or float.
            In case of float, either high or low should also be supplied.
        prob : float, optional
            estimate of P(truncation-condtion is satisfied). When supplied,
            can improve the performance of sampling-method of truncated sampler.
        max_iters : float, optional
            if the number of iterations needed for obtaining the sample exceeds this number,
            either a warning or error is raised. By default is set to 1e7 (constant of TruncateSampler-class).
        sample_anyways : bool, optional
            If set to True, when exceeding `self.max_iters` number of iterations the procedure throws a warning
            but continues. If set to False, the error is raised.

        Returns
        -------
        Sampler
            new Sampler-instance, truncated version of self.
        """
        return TruncateSampler(self, high, low, expr, prob, max_iters, sample_anyways)


class OrSampler(Sampler):
    """ Class for implementing `|` (mixture) operation on `Sampler`-instances.
    """
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bases = [left, right]

        # calculate probs of samplers in mixture
        weights = np.array([self.bases[0].weight, self.bases[1].weight])
        self.weight = np.sum(weights)
        self.normed = weights / np.sum(weights)

    def sample(self, size):
        """ Sampling procedure of a mixture of two samplers. Samples points with probabilities
        defined by weights (`self.weight`-attr) from two samplers invoked (`self.bases`-attr) and
        mixes them in one sample of needed size.
        """
        up_size = np.random.binomial(size, self.normed[0])
        low_size = size - up_size

        up_sample = self.bases[0].sample(size=up_size)
        low_sample  = self.bases[1].sample(size=low_size)
        sample_points = np.concatenate([up_sample, low_sample])
        sample_points = sample_points[np.random.permutation(size)]

        return sample_points

class AndSampler(Sampler):
    """ Class for implementing `&` (coordinates stacking) operation on `Sampler`-instances.
    """
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bases = [left, right]

    def sample(self, size):
        """ Sampling procedure of a product of two samplers. Check out the docstring of
        `Sampler.__and__` for more info.
        """
        left_sample = self.bases[0].sample(size)
        right_sample = self.bases[1].sample(size)
        return np.concatenate([left_sample, right_sample], axis=1)

class ApplySampler(Sampler):
    """ Class for implementing `apply` (adding transform) operation on `Sampler`-instances.
    """
    def __init__(self, sampler, transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bases = [sampler]
        self.transform = transform

    def sample(self, size):
        """ Sampling procedure of a sampler subjugated to a transform. Check out the docstring of
        `Sampler.apply` for more info.
        """
        return self.transform(self.bases[0].sample(size))

class TruncateSampler(Sampler):
    """ Class for implementing `truncate` (truncation by a condition) operation on `Sampler`-instances.
    """
    # Used when truncating a Sampler. If we cannot obtain a needed amount of points
    # from the region of interest using this number of iterations, we throw a Warning or ValueError
    max_iters = 1e7

    def __init__(self, sampler, high=None, low=None, expr=None, prob=0.5, max_iters=None,
                 sample_anyways=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bases = [sampler]
        self.high = high
        self.low = low
        self.expr = expr
        self.prob = prob
        self.sample_anyways = sample_anyways
        self.max_iters = max_iters or self.max_iters

    def sample(self, size):
        """ Sampling method of a sampler subjugated to truncation. Check out the docstring of
        `Sampler.truncation` for more info.
        """
        if size == 0:
            return self.bases[0].sample(size=0)

        high, low, expr, prob = self.high, self.low, self.expr, self.prob
        # set batch-size
        expectation = size / prob
        sigma = np.sqrt(size * (1 - prob) / (prob**2))
        batch_size = int(expectation + 2 * sigma)

        # sample, filter out, concat
        ctr = 0
        cumulated = 0
        samples = []
        while cumulated < size:
            # sample points and compute condition-vector
            sample = self.bases[0].sample(size=batch_size)
            cond = np.ones(shape=batch_size).astype(np.bool)
            if low is not None:
                if expr is not None:
                    cond &= np.greater_equal(expr(sample).reshape(batch_size, -1), low).all(axis=1)
                else:
                    cond &= np.greater_equal(sample, low).all(axis=1)

            if high is not None:
                if expr is not None:
                    cond &= np.less_equal(expr(sample).reshape(batch_size, -1), high).all(axis=1)
                else:
                    cond &= np.less_equal(sample, high).all(axis=1)

            if high is None and low is None:
                cond &= expr(sample).all(axis=1)

            # check if we reached max_iters-number of iterations
            if ctr > self.max_iters:
                if self.sample_anyways:
                    msg = f"Already took {self.max_iters} number of iteration to make a sample. "\
                          "Yet, `sample_anyways` is set to true, so going on. Kill the process manually if needed."
                    warnings.warn(msg)
                else:
                    msg = f"The number of iterations needed to obtain the sample exceeds {self.max_iters}. "\
                          "Stopping the process."
                    raise ValueError(msg)

            # get points from region of interest
            samples.append(sample[cond])
            cumulated += np.sum(cond)
            ctr += 1

        return np.concatenate(samples)[:size]


class BaseOperationSampler(Sampler):
    """ Base class for implementing all arithmetic operations on `Sampler`-instances.
    """
    operation = None
    def __init__(self, left, right, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bases = [left, right]

    def sample(self, size):
        if isinstance(self.bases[1], Sampler):
            return getattr(self.bases[0].sample(size), self.operation)(self.bases[1].sample(size))
        return getattr(self.bases[0].sample(size), self.operation)(np.array(self.bases[1]))


class AddSampler(BaseOperationSampler):
    operation = '__add__'

class MulSampler(BaseOperationSampler):
    operation = '__mul__'

class TruedivSampler(BaseOperationSampler):
    operation = '__truediv__'

class SubSampler(BaseOperationSampler):
    operation = '__sub__'

class PowSampler(BaseOperationSampler):
    operation = '__pow__'

class FloordivSampler(BaseOperationSampler):
    operation = '__floordiv__'

class ModSampler(BaseOperationSampler):
    operation = '__mod__'

class RAddSampler(BaseOperationSampler):
    operation = '__radd__'

class RMulSampler(BaseOperationSampler):
    operation = '__rmul__'

class RTruedivSampler(BaseOperationSampler):
    operation = '__rtruediv__'

class RSubSampler(BaseOperationSampler):
    operation = '__rsub__'

class RPowSampler(BaseOperationSampler):
    operation = '__rpow__'

class RFloordivSampler(BaseOperationSampler):
    operation = '__rfloordiv__'

class RModSampler(BaseOperationSampler):
    operation = '__rmod__'

classes = dict(zip(['__add__', '__mul__', '__truediv__', '__sub__', '__pow__', '__floordiv__', '__mod__',
                    '__radd__', '__rmul__', '__rtruediv__', '__rsub__', '__rpow__', '__rfloordiv__', '__rmod__'],
                   [AddSampler, MulSampler, TruedivSampler, SubSampler, PowSampler, FloordivSampler, ModSampler,
                    RAddSampler, RMulSampler, RTruedivSampler, RSubSampler, RPowSampler, RFloordivSampler,
                    RModSampler]))


class ConstantSampler(Sampler):
    """ Sampler of a constant.

    Parameters
    ----------
    constant : int, str, float, list
        constant, associated with the Sampler. Can be multidimensional,
        e.g. list or np.ndarray.

    Attributes
    ----------
    constant : np.array
        vectorized constant, associated with the Sampler.
    """
    def __init__(self, constant, **kwargs):
        self.constant = np.array(constant).reshape(1, -1)
        super().__init__(constant, **kwargs)

    def sample(self, size):
        """ Sampling method of ``ConstantSampler``.
        Repeats sampler's constant ``size`` times.

        Parameters
        ----------
        size : int
            the size of sample to be generated.

        Returns
        -------
        np.ndarray
            array of shape (size, 1) containing Sampler's constant.
        """
        return np.repeat(self.constant, repeats=size, axis=0)


class NumpySampler(Sampler):
    """ Sampler based on a distribution from `numpy random`.

    Parameters
    ----------
    name : str
        a distribution name (a method from `numpy random`) or its alias.
    seed : int
        random seed for setting up sampler's state (see :func:`~.make_rng`).
    **kwargs
        additional keyword-arguments defining properties of specific
        distribution (e.g. ``loc`` for 'normal').

    Attributes
    ----------
    name : str
        a distribution name (a method from `numpy random`).
    state : numpy.random.Generator
        a random number generator
    _params : dict
        dict of args for Sampler's distribution.
    """
    def __init__(self, name, seed=None, **kwargs):
        super().__init__(name, seed, **kwargs)
        name = _get_method_by_alias(name, 'np')
        self.name = name
        self._params = copy(kwargs)
        self.state = make_rng(seed)


    def sample(self, size):
        """ Generates random samples from distribution ``self.name``.

        Parameters
        ----------
        size : int
            the size of sample to be generated.

        Returns
        -------
        np.ndarray
            array of shape (size, Sampler's dimension).
        """
        sampler = getattr(self.state, self.name)
        sample = sampler(size=size, **self._params)
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1)
        return sample


class ScipySampler(Sampler):
    """ Sampler based on a distribution from `scipy.stats`.

    Parameters
    ----------
    name : str
        a distribution name, a class from `scipy.stats`, or its alias.
    seed : int
        random seed for setting up sampler's state (see :func:`~.make_rng`).
    **kwargs
        additional parameters for specification of the distribution.
        For instance, `scale` for name='gamma'.

    Attributes
    ----------
    name : str
        a distribution name (a class from `scipy.stats`).
    state : numpy.random.Generator
        a random number generator
    distr
        a distribution class
    """
    def __init__(self, name, seed=None, **kwargs):
        super().__init__(name, seed, **kwargs)
        name = _get_method_by_alias(name, 'ss')
        self.name = name
        self.state = make_rng(seed)
        self.distr = getattr(ss, self.name)(**kwargs)

    def sample(self, size):
        """ Sampling method of ``ScipySampler``.
        Generates random samples from distribution ``self.name``.

        Parameters
        ----------
        size : int
            the size of sample to be generated.

        Returns
        -------
        np.ndarray
            array of shape (size, Sampler's dimension).
        """
        sampler = self.distr.rvs
        sample = sampler(size=size, random_state=self.state)
        if len(sample.shape) == 1:
            sample = sample.reshape(-1, 1)
        return sample


class HistoSampler(Sampler):
    """ Sampler based on a histogram, output of `np.histogramdd`.

    Parameters
    ----------
    histo : tuple
        histogram, on which the sampler is based.
        Make sure that it is unnormalized (`normed=False` in `np.histogramdd`).
    edges : list
        list of len=histo_dimension, contains edges of bins along axes.
    seed : int
        random seed for setting up sampler's state (see :func:`~.make_rng`).

    Attributes
    ----------
    bins : np.ndarray
        bins of base-histogram (see `np.histogramdd`).
    edges : list
        edges of base-histogram.

    Notes
    -----
        The sampler should be based on unnormalized histogram.
        if `histo`-arg is supplied, it is used for histo-initilization.
        Otherwise, edges should be supplied. In this case all bins are empty.
    """
    def __init__(self, histo=None, edges=None, seed=None, **kwargs):
        super().__init__(histo, edges, seed, **kwargs)
        if histo is not None:
            self.bins = histo[0]
            self.edges = histo[1]
        elif edges is not None:
            self.edges = edges
            bins_shape = [len(axis_edge) - 1 for axis_edge in edges]
            self.bins = np.zeros(shape=bins_shape, dtype=np.float32)
        else:
            raise ValueError('Either `histo` or `edges` should be specified.')

        self.l_all = cart_prod(*(range_dim[:-1] for range_dim in self.edges))
        self.h_all = cart_prod(*(range_dim[1:] for range_dim in self.edges))

        self.probs = (self.bins / np.sum(self.bins)).reshape(-1)
        self.nonzero_probs_idx = np.asarray(self.probs != 0.0).nonzero()[0]
        self.nonzero_probs = self.probs[self.nonzero_probs_idx]

        self.state = make_rng(seed)
        self.state_sampler = self.state.uniform

    def sample(self, size):
        """ Sampling method of ``HistoSampler``.
        Generates random samples from distribution, represented by
        histogram (self.bins, self.edges).

        Parameters
        ----------
        size : int
            the size of sample to be generated.

        Returns
        -------
        np.ndarray
            array of shape (size, histo dimension).
        """
        # Choose bins to use according to non-zero probabilities
        bin_nums = self.state.choice(self.nonzero_probs_idx, p=self.nonzero_probs, size=size)

        # uniformly generate samples from selected boxes
        low, high = self.l_all[bin_nums], self.h_all[bin_nums]
        return self.state_sampler(low=low, high=high)

    def update(self, points):
        """ Update bins of sampler's histogram by throwing in additional points.

        Parameters
        ----------
        points : np.ndarray
            Array of points of shape (n_points, histo_dimension).
        """
        histo_update = np.histogramdd(sample=points, bins=self.edges)
        self.bins += histo_update[0]


def cart_prod(*arrs):
    """ Get array of cartesian tuples from arbitrary number of arrays.
    Faster version of itertools.product. The order of tuples is lexicographic.

    Parameters
    ----------
    arrs : tuple, list or ndarray.
        Any sequence of ndarrays.

    Returns
    -------
    ndarray
        2d-array with rows (arr[0][i], arr[2][j],...,arr[n][k]).
    """
    grids = np.meshgrid(*arrs, indexing='ij')
    return np.stack(grids, axis=-1).reshape(-1, len(arrs))
