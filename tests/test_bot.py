"""Unit tests for app/bot.py.

These tests never contact the real Telegram API and never use the real token.
Run them from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Chat, Message, MessageEntity, Update
from telegram.ext import CommandHandler, MessageHandler

from app.bot import build_application, echo, load_bot_token, start

# A fake token with the same shape as a real one. It is NOT a real secret.
FAKE_TOKEN = "123456789:FAKE-TOKEN-FOR-UNIT-TESTS"


def make_mock_update(text: str) -> MagicMock:
    """Create a fake Update whose message.reply_text we can inspect."""
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    return update


def make_real_update(text: str, is_command: bool = False) -> Update:
    """Create a real telegram Update object, like the one Telegram would send."""
    entities = []
    if is_command:
        # Telegram marks commands like "/start" with a "bot_command" entity.
        entities = [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=len(text))]

    chat = Chat(id=1, type=Chat.PRIVATE)
    message = Message(
        message_id=1,
        date=datetime.now(),
        chat=chat,
        text=text,
        entities=entities,
    )
    # CommandHandler reads the bot's username (to support "/start@BotName"),
    # so we attach a fake bot. No network call is made.
    message.set_bot(MagicMock(username="DocuBotTestBot"))
    return Update(update_id=1, message=message)


def build_test_application():
    """Build the real Application, but with a fake token instead of .env."""
    with patch("app.bot.load_bot_token", return_value=FAKE_TOKEN):
        return build_application()


def find_handler(application, handler_type):
    """Return the first registered handler of the given type."""
    for handler in application.handlers[0]:
        if isinstance(handler, handler_type):
            return handler
    raise AssertionError(f"No {handler_type.__name__} registered")


class LoadBotTokenTests(unittest.TestCase):
    # patch load_dotenv so the real .env file is never read during tests.
    @patch("app.bot.load_dotenv")
    def test_raises_clear_error_when_token_is_missing(self, _mock_load_dotenv):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as error:
                load_bot_token()

        self.assertIn("TELEGRAM_BOT_TOKEN is not set", str(error.exception))

    @patch("app.bot.load_dotenv")
    def test_returns_token_from_environment(self, _mock_load_dotenv):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": FAKE_TOKEN}, clear=True):
            self.assertEqual(load_bot_token(), FAKE_TOKEN)


class CallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_sends_welcome_message(self):
        update = make_mock_update("/start")

        await start(update, MagicMock())

        update.message.reply_text.assert_awaited_once_with("Welcome to DocuBot AI!")

    async def test_echo_repeats_english_text(self):
        update = make_mock_update("Hello")

        await echo(update, MagicMock())

        update.message.reply_text.assert_awaited_once_with("You said: Hello")

    async def test_echo_repeats_chinese_text(self):
        update = make_mock_update("你好 DocuBot")

        await echo(update, MagicMock())

        update.message.reply_text.assert_awaited_once_with("You said: 你好 DocuBot")

    async def test_echo_ignores_edited_messages(self):
        # Edited messages arrive with update.message = None.
        update = MagicMock()
        update.message = None

        await echo(update, MagicMock())  # must not raise


class HandlerRoutingTests(unittest.TestCase):
    def setUp(self):
        self.application = build_test_application()

    def test_start_command_handler_is_registered(self):
        handler = find_handler(self.application, CommandHandler)

        self.assertEqual(handler.commands, frozenset({"start"}))
        self.assertIs(handler.callback, start)

    def test_echo_message_handler_is_registered(self):
        handler = find_handler(self.application, MessageHandler)

        self.assertIs(handler.callback, echo)

    def test_echo_filter_accepts_normal_text(self):
        handler = find_handler(self.application, MessageHandler)

        for text in ["Hello", "How are you?", "你好 DocuBot", "12345"]:
            with self.subTest(text=text):
                self.assertTrue(handler.check_update(make_real_update(text)))

    def test_echo_filter_rejects_commands(self):
        handler = find_handler(self.application, MessageHandler)

        update = make_real_update("/start", is_command=True)

        self.assertFalse(handler.check_update(update))

    def test_start_command_is_routed_to_start_handler(self):
        handler = find_handler(self.application, CommandHandler)

        update = make_real_update("/start", is_command=True)

        self.assertTrue(handler.check_update(update))


if __name__ == "__main__":
    unittest.main()
