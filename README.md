# docubot-ai
Production-oriented RAG Telegram knowledge assistant

> **Current status: Telegram Bot MVP + FastAPI file upload MVP.**
> Uploaded documents are only **stored**. Text extraction, RAG, and AI features
> are planned but **not implemented yet**.

## Current Features

**Telegram bot** ([app/bot.py](app/bot.py))

- `/start` command — replies `Welcome to DocuBot AI!`
- Normal-text Echo — replies `You said: <your message>`
- Environment-variable based token configuration (`TELEGRAM_BOT_TOKEN`)
- Local development using polling

**FastAPI backend** ([app/api.py](app/api.py))

- `GET /health` — health check
- `POST /upload` — upload one document
- Supported file types: PDF, DOCX, TXT (max 10 MB)
- Local development storage in `storage/uploads/`

The Telegram bot and the web API are two **separate programs**. They are not
connected to each other yet.

## How It Works

Telegram bot:

```
Telegram User
→ Telegram
→ Python Bot (python-telegram-bot, polling)
→ Handler
→ Response
→ Telegram User
```

| User sends | Handler | Callback | Reply |
|---|---|---|---|
| `/start` | `CommandHandler("start")` | `start()` | `Welcome to DocuBot AI!` |
| Normal text (not a command) | `MessageHandler(filters.TEXT & ~filters.COMMAND)` | `echo()` | `You said: <text>` |

Web API upload:

```
Client (browser / script)
→ HTTP POST /upload (multipart/form-data, field name "file")
→ FastAPI validates filename, extension, size
→ File saved as storage/uploads/<random-id>.<ext>
→ JSON metadata returned to the client
```

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /upload`

Send one file as `multipart/form-data` in a field named `file`.

Success — `201 Created`:

```json
{
  "file_id": "8adc0251f64d4cf983e29ce470705707",
  "filename": "handbook.pdf",
  "stored_filename": "8adc0251f64d4cf983e29ce470705707.pdf",
  "extension": ".pdf",
  "content_type": "application/pdf",
  "size_bytes": 12345,
  "status": "stored"
}
```

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| `400 Bad Request` | Filename missing, too long, contains a path (`/` or `\`), or the file is empty |
| `413 Content Too Large` | File is larger than 10 MB |
| `415 Unsupported Media Type` | Extension is not `.pdf`, `.docx`, or `.txt` |
| `422 Unprocessable Content` | No `file` field was sent (FastAPI's built-in validation) |
| `500 Internal Server Error` | The file could not be saved, or an unexpected error happened (no internal details are returned) |

Interactive API docs are available at `http://127.0.0.1:8000/docs` while the server runs.

## File Upload Rules

- Supported: **PDF**, **DOCX**, **TXT** (checked by file extension, case-insensitive)
- Maximum size: **10 MB** (`MAX_UPLOAD_SIZE_BYTES` in `app/api.py`)
- The server generates the stored filename (`<random-id>.<ext>`). The user's
  original filename is only returned as metadata; it is never used as a file path.
- Files are stored locally in `storage/uploads/`. Uploaded files are ignored by Git.
- Documents are currently **not parsed, not indexed, and not searchable**.

### Security limitations (development only)

This upload endpoint has basic validation, but it is **not production-secure** yet:

- The file type is checked only by **extension**. File contents are not
  inspected, so a renamed file would be accepted.
- There is **no authentication** — anyone who can reach the server can upload.
- There is **no virus/malware scanning**.
- The whole request is received before the 10 MB check runs. A production
  deployment should also limit request size at a reverse proxy.
- There is no rate limiting and no storage quota.
- Uploaded files are only stored, never executed.

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

The web API does not need the Telegram token.

## Local Run

Run these from the project root.

Telegram bot:

```powershell
.venv\Scripts\python.exe -m app.bot
```

Web API (development server with auto-reload):

```powershell
.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

The API listens on `http://127.0.0.1:8000`. Press `Ctrl+C` to stop either program.

## Testing

Tests use Python's built-in `unittest`. They do not contact Telegram, do not
use the real `.env`, and write uploads only into temporary folders:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
