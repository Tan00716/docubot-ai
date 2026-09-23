# docubot-ai
Production-oriented RAG Telegram knowledge assistant

> **Current status: Telegram Bot MVP v0.1.**
> RAG, document upload, and AI features are planned but **not implemented yet**.

## Current Features

- `/start` command — replies `Welcome to DocuBot AI!`
- Normal-text Echo — replies `You said: <your message>`
- Environment-variable based token configuration (`TELEGRAM_BOT_TOKEN`)
- Local development using polling

## How It Works

```
Telegram User
→ Telegram
→ Python Bot (python-telegram-bot, polling)
→ Handler
→ Response
→ Telegram User
```

The bot has two handlers in [app/bot.py](app/bot.py):

| User sends | Handler | Callback | Reply |
|---|---|---|---|
| `/start` | `CommandHandler("start")` | `start()` | `Welcome to DocuBot AI!` |
| Normal text (not a command) | `MessageHandler(filters.TEXT & ~filters.COMMAND)` | `echo()` | `You said: <text>` |

## Local Setup

1. Python environment uses a local virtual environment in `.venv`.
2. Dependencies are listed in `requirements.txt`:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. The Telegram token is stored in a `.env` file in the project root.
   Copy `.env.example` to `.env` and put your own token from BotFather in it:

   ```
   TELEGRAM_BOT_TOKEN=your-token-here
   ```

4. `.env` must **never** be committed. It is already listed in `.gitignore`.

## Run

From the project root:

```powershell
.venv\Scripts\python.exe -m app.bot
```

The bot keeps running and polls Telegram for new messages. Press `Ctrl+C` to stop it.

## Tests

Unit tests use Python's built-in `unittest` and do not contact Telegram:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
