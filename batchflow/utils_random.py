""" contains data utils """
import warnings

import numpy as np


def make_rng(seed=None):
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
    numpy.random.Generator
    """
    if seed is False:
        rng = None
    elif seed is None or seed is True:
        rng = np.random.default_rng(np.random.SFC64())
    elif isinstance(seed, np.random.SeedSequence):
        rng = np.random.default_rng(np.random.SFC64(seed))
    elif isinstance(seed, (int, np.integer)):
        rng = np.random.default_rng(np.random.SFC64(seed))
    elif isinstance(seed, np.random.Generator):
        rng = seed
    elif isinstance(seed, np.random.BitGenerator):
        rng = np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        warnings.warn(f"Unknown seed type: {type(seed)}.")
        rng = None

    return rng


def make_seed_sequence(shuffle=False):
    """ Create a seed sequence for random number generation

    Parameters
    ----------
    shuffle : bool or int or object with a seed sequence attribute
        a random state

        - False or True - creates a new seed sequence with random entropy
        - int - creates a new seed sequence with the given entropy

    Returns
    -------
    numpy.random.SeedSequence
    """
    if isinstance(getattr(shuffle, 'random_seed', None), np.random.SeedSequence):
        return shuffle.random_seed
    if shuffle is None or isinstance(shuffle, bool):
        seed = np.random.SeedSequence()
    elif isinstance(shuffle, int):
        if shuffle >= 0:
            seed = np.random.SeedSequence(shuffle)
        else:
            # if shuffle is negative, do not shuffle the dataset, but use the seed for randomization
            seed = np.random.SeedSequence(-shuffle)
    else:
        raise TypeError('shuffle can be bool or int', shuffle)

    return seed


def spawn_seed_sequence(source):
    """ Return a new seed sequence or None

    Parameters
    ----------
    source : numpy.random.SeedSequence or Batch or Pipeline

    Returns
    -------
    numpy.random.SeedSequence
    """
    if isinstance(source, np.random.SeedSequence):
        pass
    elif isinstance(getattr(source, 'random_seed', None), np.random.SeedSequence):
        source = source.random_seed
    else:
        raise ValueError(f'source should be SeedSequence, Batch or Pipeline, but given {type(source)}')

    return source.spawn(1)[0]
