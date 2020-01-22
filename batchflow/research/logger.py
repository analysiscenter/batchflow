""" Logger class """

import os
import traceback
import logging
import requests

from .named_expr import RD
from ..named_expr import eval_expr

def _get_traceback(exception):
    ex_traceback = exception.__traceback__
    return ''.join(traceback.format_exception(exception.__class__, exception, ex_traceback))

def log_info(message, path):
    """ Write message into log. """
    filename = os.path.join(path, 'research.log')
    logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.INFO)
    logging.info(message)

def log_error(exception, path):
    """ Write error message into log. """
    filename = os.path.join(path, 'research.log')
    logging.basicConfig(format='%(levelname)-8s [%(asctime)s] %(message)s', filename=filename, level=logging.ERROR)
    logging.info(_get_traceback(exception))

class BaseLogger:
    """ Basic logging class.

    BaseLogger consists of one or few pairs of functions (info logging and error logging).
    """
    def __init__(self, loggers=None):
        if loggers is None:
            loggers = []
        self._loggers = loggers

    def info(self, message, **kwargs):
        """ Log some message

        Parameters
        ----------
        message : str

        kwargs : dict
            parameters for info function
        """
        for item in self._loggers:
            if 'info' in item and item['info'] is not None:
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

    def append(self, info, error=None, **kwargs):
        self._loggers.append({
            'info': info,
            'error': error,
            'kwargs': kwargs
        })

    def eval_kwargs(self, **kwargs):
        for item in self._loggers:
            item['kwargs'] = eval_expr(item['kwargs'], **kwargs)

    def __add__(self, other):
        # pylint: disable=protected-access
        if isinstance(other, BaseLogger):
            return BaseLogger(self._loggers + other._loggers)
        raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))

class FileLogger(BaseLogger):
    """ Basic logging class """
    def __init__(self):
        super().__init__()
        self._loggers = [{'info': log_info, 'error': log_error, 'kwargs': dict(path=RD())}]

class PrintLogger(BaseLogger):
    """ Logging by print """
    def __init__(self):
        super().__init__()
        self._loggers = [{'info': print, 'error': print, 'kwargs': dict()}]

class TelegramLogger(BaseLogger):
    """ Telegram Logger """
    def __init__(self, bot_token, chat_id):
        """ Initialize Logger

        Parameters
        ----------
        bot_token : str
            telegram bot token

        chat_id : int or str

        **How to get token and chat id**
            See https://github.com/datagym-ru/tg_tqdm/
        """
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self._loggers = [{'info': self._info, 'error': self._info, 'kwargs': dict()}]

    def _info(self, message):
        send_text = ('https://api.telegram.org/bot' + self.bot_token + '/sendMessage?chat_id='
                     + self.chat_id + '&parse_mode=Markdown&text=' + message)
        response = requests.get(send_text)
        return response.json()

    def _error(self, exception):
        return self._info(_get_traceback(exception))
