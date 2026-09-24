# docubot-ai
Production-oriented RAG Telegram knowledge assistant

> **Current status: Telegram Bot MVP + FastAPI file upload + document processing.**
> Uploaded documents can be turned into normalized text with source locations.
> Chunking, search, RAG, and AI features are planned but **not implemented yet**.

## Current Features

**Telegram bot** ([app/bot.py](app/bot.py))

- `/start` command — replies `Welcome to DocuBot AI!`
- Normal-text Echo — replies `You said: <your message>`
- Environment-variable based token configuration (`TELEGRAM_BOT_TOKEN`)
- Local development using polling

**FastAPI backend** ([app/api.py](app/api.py))

- `GET /health` — health check
- `POST /upload` — upload one document
- `POST /documents/{document_id}/process` — extract and normalize its text
- `GET /documents/{document_id}` — processing status and metadata
- Supported file types: PDF, DOCX, TXT (max 10 MB)
- Local development storage in `storage/` (uploads, processed output, SQLite metadata)

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
→ Metadata row saved in SQLite (status "uploaded")
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

The `file_id` is the document's ID for the document endpoints below.

### `POST /documents/{document_id}/process`

Extracts and normalizes the text of an uploaded document and saves it as JSON
in `storage/processed/<document_id>.json`. Processing is **synchronous**: the
response arrives when processing is finished. Processing a document again is
allowed and produces the same output.

Success — `200 OK`:

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "original_filename": "handbook.pdf",
  "file_type": ".pdf",
  "content_type": "application/pdf",
  "size_bytes": 12345,
  "status": "completed",
  "created_at": "2026-09-24T01:30:00.123456+00:00",
  "updated_at": "2026-09-24T01:30:01.654321+00:00",
  "processed_path": "8adc0251f64d4cf983e29ce470705707.json",
  "error_message": null,
  "section_count": 12
}
```

`processed_path` is a filename inside `storage/processed/`. The extracted text
itself is never returned by the API.

Errors return `{"detail": "..."}`. When processing starts but fails, the
document's status becomes `failed` and the same safe message is stored in
`error_message`:

| Status | When |
|---|---|
| `400 Bad Request` | The document ID is not a valid ID (32 lower-case hex characters) |
| `404 Not Found` | No document with this ID, or its stored file is missing |
| `409 Conflict` | The document is already being processed |
| `415 Unsupported Media Type` | The file type has no parser |
| `422 Unprocessable Content` | Corrupted PDF, invalid DOCX, DOCX that unpacks to more than 50 MB, TXT not valid UTF-8, encrypted PDF, or no text found |
| `500 Internal Server Error` | Processed output or the database could not be written, or an unexpected error happened (no internal details are returned) |

### `GET /documents/{document_id}`

Returns the same metadata fields as above (without `section_count`):
`status` is one of `uploaded`, `processing`, `completed`, `failed`.
`processed_path` is set only when completed; `error_message` only when failed.
It returns `400` for an invalid ID and `404` for an unknown document.

Interactive API docs are available at `http://127.0.0.1:8000/docs` while the server runs.

## File Upload Rules

- Supported: **PDF**, **DOCX**, **TXT** (checked by file extension, case-insensitive)
- Maximum size: **10 MB** (`MAX_UPLOAD_SIZE_BYTES` in `app/api.py`)
- The server generates the stored filename (`<random-id>.<ext>`). The user's
  original filename is only returned as metadata; it is never used as a file path.
- Files are stored locally in `storage/uploads/`. Uploaded files are ignored by Git.
- Uploading only stores the file. Text is extracted when the process endpoint
  is called. Documents are **not chunked, indexed, or searchable** yet.

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

## Document Processing

Code: [app/document_processing/](app/document_processing/)

```
Upload (POST /upload)
→ Process (POST /documents/{id}/process)
→ Extract text with the parser for the file type
→ Normalize whitespace
→ Store the processed representation (storage/processed/<id>.json)
→ Track status in SQLite (storage/metadata/documents.db)
```

| File | Responsibility |
|---|---|
| `models.py` | Data shapes: `Section`, `ProcessedDocument`, `DocumentStatus`, error types |
| `parsers.py` | Text extraction per file type + normalization (bytes in, sections out) |
| `database.py` | SQLite table `documents` and status changes |
| `processor.py` | The pipeline: find → mark processing → parse → save JSON → mark completed/failed |

### Supported formats

| Format | Library | How text is split | Source location |
|---|---|---|---|
| PDF | `pypdf` | One section per page that has text | `{"page": 3}` (real page number) |
| DOCX | `python-docx` | One section per non-empty body paragraph, in document order | `{"paragraph": 12}` |
| TXT | Python standard library | One section per block of lines separated by blank lines | `{"block": 2, "line_start": 5, "line_end": 9}` |

TXT files must be UTF-8 (a UTF-8 BOM is allowed and removed).

### Processed representation

Every format produces the same JSON shape, so a later stage can read any
document the same way:

```json
{
  "schema_version": 1,
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "source_filename": "handbook.pdf",
  "file_type": ".pdf",
  "sections": [
    {"source_location": {"page": 1}, "text": "Welcome to the handbook."},
    {"source_location": {"page": 3}, "text": "Page three: rules."}
  ]
}
```

The output is deterministic: processing the same file again produces
byte-for-byte the same JSON (no timestamps inside, fixed key order).

### Normalization

Only whitespace is cleaned; the text is never rewritten or summarized:

- line endings become `\n`; null characters are removed
- runs of spaces/tabs become one space; each line is trimmed
- paragraph breaks are kept (more than one blank line becomes one blank line)
- letters, digits, and punctuation are never changed

### Processing status

```
uploaded → processing → completed
                      ↘ failed
```

Statuses are defined once in `DocumentStatus` and enforced by a SQLite
`CHECK` constraint. If processing fails, the status becomes `failed` with a
short, safe `error_message` (never a stack trace or file path). A `completed`
or `failed` document can be processed again. If the server is killed during
processing, the document can be processed again after 10 minutes instead of
staying stuck in `processing`.

Files uploaded before metadata was recorded (older uploads) are registered
automatically the first time their ID is used; their original filename is
unknown, so the stored filename is shown instead.

### Limitations

- **No OCR**: scanned or image-only PDFs have no machine-readable text. They
  fail with a clear "No machine-readable text was found" message.
- **DOCX page numbers are not invented**: Word files have no fixed pages
  without rendering them, so DOCX locations are paragraph numbers.
- DOCX tables, headers, footers, and footnotes are not extracted yet.
- Encrypted PDFs are not supported.
- A DOCX file is a ZIP archive. DOCX files whose contents unpack to more than
  50 MB are refused, to protect the server from "zip bomb" files.
- Processing is **synchronous** in this MVP: the request waits until it finishes.
- SQLite is **local development** metadata storage (one file, one machine).
- All files (uploads, processed JSON, database) are stored **locally** and
  ignored by Git.
- Uploaded files are treated as untrusted data: they are only read and parsed,
  never executed, and extracted text is never treated as instructions.

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
use the real `.env`, and write uploads, processed output, and SQLite databases
only into temporary folders. Test PDFs and DOCX files are generated in code
([tests/sample_documents.py](tests/sample_documents.py)):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
