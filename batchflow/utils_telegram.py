""" A class to send updateable content messages to Telegram. """
import os
import json
import imghdr
from uuid import uuid4
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

from urllib3 import PoolManager



class TelegramMessage:
    """ Class to send a message with text or image (either a matplotlib figure or path to photo) to a Telegram bot.
    On subsequent updates, the same message is edited to avoid clutter in notifications.
    By default, all messages are silent (with no notifications).

    All of Telegram updates are send with `urllib3`, with no usage of official Telegram Python API.
    Message updates are performed in a separate thread to avoid IO constraints.

    One must supply telegram `token` and `chat_id` either by passing directly or
    setting environment variables `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`. To get them:
        - create a bot <https://core.telegram.org/bots#6-botfather> and copy its `{token}`
        - add the bot to a chat and send it a message such as `/start`
        - go to <https://api.telegram.org/bot`{token}`/getUpdates> to find out the `{chat_id}`
    """
    API = 'https://api.telegram.org/bot'
    USER_AGENT = 'Python Telegram Bot (https://github.com/python-telegram-bot/python-telegram-bot)'

    def __init__(self, token=None, chat_id=None, silent=True, content=None):
        # Connection
        self.token = token or os.getenv('TELEGRAM_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        if self.token is None or self.chat_id is None:
            raise ValueError('Supply `token` and `chat_id` or '
                             'set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` environment variables!')

        self.connection = PoolManager()

        # State
        self.deleted = False
        self.silent = silent

        # Keep track of the message, its type and previous contents
        self.message_id = None
        self.message_type = None
        self.message_content_hash = None

        # A separate worker to send requests
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.queue = []

        if content is not None:
            self.send(content)

    def submit(self, function, *args, **kwargs):
        """ Run tasks in a separate thread, keeping at most two tasks in a queue: one running, one waiting. """
        if len(self.queue) == 2:
            if self.queue[0].done():
                # The first task is already done, means that the second is running
                # Remove the first, make running the first, put new task as the second
                self.queue.pop(0)
            else:
                # The first task is running, so we can replace the waiting task with the new one
                self.queue.pop(1).cancel()
        try:
            waiting = self.pool.submit(function, *args, **kwargs)
            self.queue.append(waiting)
        except Exception as e: #pylint: disable=broad-except
            print(e)


    def post(self, action, **kwargs):
        """ Send a `POST` request with `action` to Telegram API.
        We use the same headers, as used in official Telegram Python API implementation, see
        <https://github.com/python-telegram-bot/python-telegram-bot/blob/master/telegram/utils/request.py>`_".
        """
        url = f'{self.API}{self.token}/{action}'
        fields = {'chat_id': self.chat_id, **kwargs}
        headers = {'connection': 'keep-alive', 'user-agent': self.USER_AGENT}
        return self.connection.request('POST', url, fields=fields, headers=headers)


    def send(self, content, force_update=False):
        """ Send content to a message in a separate thread. """
        if self.deleted:
            raise TypeError('Sending contents to a deleted message is not allowed!')

        self.submit(function=self._send, content=content, force_update=force_update)

        # Wait for the first message
        if self.message_id is None:
            self.queue[0].result()

    def _send(self, content, force_update):
        # No need to re-send the same content
        if not force_update and hash(content) == self.message_content_hash:
            return None

        # Assert the same type of contents to send, as in initialization
        message_type = 'text' if isinstance(content, str) else 'media'
        if self.message_type and self.message_type != message_type:
            raise TypeError('Sending different types of content in the same message is not allowed!')

        # First message: initialization
        if self.message_id is None:
            if message_type == 'text':
                response = self.post('sendMessage', text=content,
                                     disable_notification=self.silent, parse_mode='MarkdownV2')
            else:
                data = self.content_to_dict(content)
                response = self.post('sendMediaGroup', **data,
                                     disable_notification=self.silent)

            response_ = json.loads(response.data.decode())['result']
            response_ = response_[0] if isinstance(response_, list) else response_
            self.message_id = response_['message_id']
            self.message_type = message_type

        # Subsequent messages: updating contents by editing the message
        else:
            if message_type == 'text':
                response = self.post('editMessageText', text=content,
                                     message_id=self.message_id, parse_mode='MarkdownV2')
            else:
                data = self.content_to_dict(content, group=False)
                response = self.post('editMessageMedia', **data,
                                     message_id=self.message_id)
        self.message_content_hash = hash(content)
        return response


    def delete(self):
        """ Cancel pending message updates and remove it. """
        for task in self.queue:
            if not task.running():
                task.cancel()
        self.pool.shutdown(wait=True)

        if self.message_id is not None:
            self.post('deleteMessage', message_id=self.message_id)
            self.deleted = True


    @staticmethod
    def content_to_dict(content, group=True, **kwargs):
        """ Convert a content (either a path to photo or matplotlib figure) to a telegram-acceptable dictionary. """
        # Convert content to a bytestring
        if isinstance(content, str):
            bytes_ = TelegramMessage.path_to_bytes(content)
        else:
            bytes_ = TelegramMessage.figure_to_bytes(content, **kwargs)

        # Create unique id of attachment, pack entities in json
        attach_id = f'attached{uuid4().hex}'
        attach_fmt = imghdr.what(None, bytes_)

        media_dict = {'media': f'attach://{attach_id}', 'type': 'photo'}
        media_dict = [media_dict] if group else media_dict
        media_dict = json.dumps(media_dict)

        # Resulting dictionary: `media` references the attachment, which is stored as a separate key
        return {'media': media_dict,
                attach_id: ('image', bytes_, attach_fmt)}

    @staticmethod
    def path_to_bytes(path, **_):
        """ Read image file contents as a bytestring. """
        with open(path, 'rb') as file:
            return file.read()

    @staticmethod
    def figure_to_bytes(figure, **kwargs):
        """ Save figure to a buffer in memory, then read from it. """
        kwargs = {'format': 'png', 'dpi': 100,
                  'bbox_inches': 'tight', 'pad_inches': 0,
                  **kwargs}
        file = BytesIO()
        figure.savefig(file, **kwargs)
        file.seek(0)
        return file.read()
