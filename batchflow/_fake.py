""" Contains fake classes for uninstalled libraries

This simplifies code for type checking and reduces strict requirements for batchflow import.
"""

class DataFrame:
    """ Fake DataFrame to use when pandas or dask are not installed """
    pass
