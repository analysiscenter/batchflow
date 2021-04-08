""" contains data utils """
import warnings

import numpy as np


def make_rng(seed):
    """ Create a random number generator

    Parameters
    ----------
    seed : bool, int, Generator, BitGenerator, RandomState
        a random state

        - False - returns None
        - None or True - creates a new SFC64 generator with random entropy
        - int - creates a new SFC64 generator with the seed given
        - SeedSequence - creates a new SFC64 generator with the seed given
        - Generator - returns it
        - BitGenerator - creates a new generator
        - RandomState - returns it

    Notes
    -----
    Do not use a legacy RandomState unless for backward compatibility.

    Returns
    -------
    a numpy RNG
    """
    if seed is False:
        random_state = None
    elif seed is None or seed is True:
        random_state = np.random.default_rng(np.random.SFC64())
    elif isinstance(seed, np.random.SeedSequence):
        random_state = np.random.default_rng(np.random.SFC64(seed))
    elif isinstance(seed, int):
        random_state = np.random.default_rng(np.random.SFC64(seed))
    elif isinstance(seed, np.random.Generator):
        random_state = seed
    elif isinstance(seed, np.random.BitGenerator):
        random_state = np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        random_state = seed
    else:
        warnings.warn("Unknown seed type: %s" %  seed)
        random_state = None

    return random_state
