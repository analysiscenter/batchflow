""" Logger class """

import os
import traceback
import logging

from .named_expr import ResearchPath
from ..named_expr import eval_expr

class Logger:
    """ Basic logging class.

    Logger consists of one or few pairs of functions (info logging and error logging).
    """
    def __init__(self):
        self._loggers = []

    def info(self, message, **kwargs):
        """ Log some message

        Parameters
        ----------
        message : str

        kwargs : dict
            parameters for info function
        """
        for item in self._loggers:
            item['info'](message, **item['kwargs'], **kwargs)

    def error(self, exception, **kwargs):
        """ Log some exception

        Parameters
        ----------
        exception : Exception

        kwargs : dict
            parameters for error function
        """
        for item in self._loggers:
            if 'error' in item and item['error'] is not None:
                item['error'](exception, **item['kwargs'], **kwargs)
            else:
                item['info'](exception, **item['kwargs'], **kwargs)

    def append(self, info, error=None, **kwargs):
        self._loggers.append({
            'info': info,
            'error': error,
            'kwargs': kwargs
        })

    def eval_kwargs(self, **kwargs):
        for item in self._loggers:
            item['kwargs'] = eval_expr(item['kwargs'], **kwargs)

class BasicLogger(Logger):
    """ Basic logging class """
    def __init__(self):
        super().__init__()
        self._loggers = [{'info': log_info, 'error': log_error, 'kwargs': dict(path=ResearchPath())}]

def log_info(message, path):
    """ Write message into log. """
    filename = os.path.join(path, 'research.log')
    logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
    logging.info(message)

def log_error(exception, path):
    """ Write error message into log. """
    filename = os.path.join(path, 'research.log')
    logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
    ex_traceback = exception.__traceback__
    tb_lines = ''.join(traceback.format_exception(exception.__class__, exception, ex_traceback))
    logging.info(tb_lines)
