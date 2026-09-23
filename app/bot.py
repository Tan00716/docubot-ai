"""Minimal Telegram Bot application entry point.

This module prepares the Application object, registers the /start command
and the echo handler for normal text messages, and starts polling.
"""

import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


def load_bot_token() -> str:
    """Load TELEGRAM_BOT_TOKEN from the environment (.env file).

    Raises:
        RuntimeError: if the token is missing.
    """
    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. "
            "Add it to your .env file before starting the bot."
        )

    return token


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to the /start command with a welcome message."""
    await update.message.reply_text("Welcome to DocuBot AI!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to a normal text message by repeating it back to the user."""
    # When a user edits an old message, Telegram sends an "edited_message"
    # update and update.message is None. We simply ignore those updates.
    if update.message is None:
        return

    await update.message.reply_text(f"You said: {update.message.text}")


def build_application() -> Application:
    """Create the Telegram Bot Application and register its handlers."""
    token = load_bot_token()
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
    return application


def main() -> None:
    """Start the bot using polling."""
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
