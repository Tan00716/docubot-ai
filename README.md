# docubot-ai
Production-oriented RAG Telegram knowledge assistant

> **Current status: Telegram Bot MVP + FastAPI file upload + document processing + chunking.**
> Uploaded documents can be turned into normalized text with source locations,
> and then split into deterministic chunks stored in SQLite.
> Embeddings, vector search, retrieval, reranking, RAG, and AI features are
> planned but **not implemented yet**.

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
- `POST /documents/{document_id}/chunk` — split the processed text into chunks
- `GET /documents/{document_id}/chunks` — list the stored chunks and their source locations
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

### `POST /documents/{document_id}/chunk`

Splits a **processed** (`completed`) document into chunks and stores them in
SQLite (see [Chunking](#chunking)). Optional query parameters:

| Parameter | Default | Rule |
|---|---|---|
| `chunk_size` | `1200` | 1 to 10000 characters |
| `chunk_overlap` | `200` | 0 or more, and at most half of `chunk_size` |

```
POST /documents/8adc0251f64d4cf983e29ce470705707/chunk?chunk_size=1200&chunk_overlap=200
```

Success — `200 OK` (a summary only; the chunk text is not returned):

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "chunking_status": "chunked",
  "chunk_count": 5,
  "chunking_version": 1,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "chunked_at": "2026-09-24T02:10:00.123456+00:00"
}
```

Chunking the same document again **replaces** its previous chunks; it never
creates duplicates. Chunking does not change the document's processing `status`.

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| `400 Bad Request` | The document ID is not valid |
| `404 Not Found` | No document with this ID, or its processed output file is missing |
| `409 Conflict` | The document is not processed yet (`uploaded`, `processing`, `failed`), or it was re-processed while being chunked |
| `422 Unprocessable Content` | Invalid `chunk_size` / `chunk_overlap`, or the processed document contains no text |
| `500 Internal Server Error` | The processed output is corrupted, the database could not be written, or an unexpected error happened (no internal details are returned) |

If chunking fails, the previously stored chunks (if any) are left unchanged.

### `GET /documents/{document_id}/chunks`

Lists a document's stored chunks in order. Add `?include_text=true` to also
return each chunk's text; by default `text` is `null` to keep the response small.

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "chunking_status": "chunked",
  "chunk_count": 5,
  "chunking_version": 1,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "chunked_at": "2026-09-24T02:10:00.123456+00:00",
  "source_filename": "handbook.pdf",
  "file_type": ".pdf",
  "chunks": [
    {
      "chunk_id": "8adc0251f64d4cf983e29ce470705707_v1_s1200_o200_00000",
      "chunk_index": 0,
      "char_count": 1143,
      "source_locations": [
        {"page": 1, "char_start": 0, "char_end": 820},
        {"page": 2, "char_start": 0, "char_end": 321}
      ],
      "text": null
    }
  ]
}
```

A document without chunks returns `"chunking_status": "not_chunked"`,
`"chunk_count": 0` and an empty `chunks` list. It returns `400` for an invalid
ID and `404` for an unknown document.

Interactive API docs are available at `http://127.0.0.1:8000/docs` while the server runs.

## File Upload Rules

- Supported: **PDF**, **DOCX**, **TXT** (checked by file extension, case-insensitive)
- Maximum size: **10 MB** (`MAX_UPLOAD_SIZE_BYTES` in `app/api.py`)
- The server generates the stored filename (`<random-id>.<ext>`). The user's
  original filename is only returned as metadata; it is never used as a file path.
- Files are stored locally in `storage/uploads/`. Uploaded files are ignored by Git.
- Uploading only stores the file. Text is extracted when the process endpoint
  is called, and chunks are created when the chunk endpoint is called.
  Documents are **not embedded, indexed, or searchable** yet.

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
| `models.py` | Data shapes: `Section`, `ProcessedDocument`, `Chunk`, `ChunkSet`, statuses, error types |
| `parsers.py` | Text extraction per file type + normalization (bytes in, sections out) |
| `chunking.py` | Chunking algorithm and its configuration (sections in, chunks out; no I/O) |
| `database.py` | SQLite tables `documents` and `chunks`, status changes, chunk storage |
| `processor.py` | The pipelines: processing (parse → save JSON) and chunking (load JSON → chunk → store) |

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
staying stuck in `processing`. Processing a document again deletes its chunks
(they were made from the old processed output); chunk it again afterwards.

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

## Chunking

Code: [app/document_processing/chunking.py](app/document_processing/chunking.py)

A **chunk** is a small piece of a processed document. Later stages
(Embedding → Vector Search → RAG) will work on chunks, not whole documents:
an embedding model can only take a limited amount of text, and a small,
focused piece of text is easier to find and to cite than a whole book.
This project only **creates and stores** chunks so far; nothing embeds or
searches them yet.

```
Processed document (storage/processed/<id>.json, status "completed")
→ POST /documents/{id}/chunk
→ Load and validate the processed JSON
→ Build chunks (deterministic, no LLM, no network)
→ Replace the document's chunks in SQLite (one transaction)
```

### How documents are split

Sizes are measured in **characters**. The defaults are defined once in
`chunking.py` (`DEFAULT_CHUNK_SIZE = 1200`, `DEFAULT_CHUNK_OVERLAP = 200`) and
can be overridden per request.

1. The **blocks** are the Batch 3 sections: PDF pages, DOCX paragraphs, TXT
   blocks. Whitespace-only blocks are skipped.
2. **Small blocks are packed** into one chunk (joined by a blank line, `\n\n`)
   until the next block would make the chunk longer than `chunk_size`. Such
   blocks are never cut.
3. **Overlap between packed chunks:** a new chunk starts with the last whole
   block(s) of the previous chunk that fit into `chunk_overlap` characters.
   A block longer than the overlap is not repeated, so the real overlap can be
   smaller than `chunk_overlap`, or zero.
4. **A block longer than `chunk_size`** (e.g. a long PDF page) is split on its
   own into windows of at most `chunk_size` characters. A window ends at the
   last space or line break in its second half, so words stay whole (only a
   single "word" longer than that, like a very long URL, is cut in the middle).
   The next window starts up to `chunk_overlap` characters earlier, moved to
   the start of a word. Windows are never mixed with other blocks.
5. No chunk is ever empty, longer than `chunk_size`, or starts/ends with
   whitespace. Text is never rewritten or summarized: every chunk is made of
   exact slices of the processed text, and every non-whitespace character of
   the document is in at least one chunk.

The loops have explicit safety guards: every window must move forward, and a
chunk may never consist only of text the previous chunk already had.

**Why overlap is limited to half of `chunk_size`:** overlapping text is stored
twice. With an overlap close to `chunk_size`, every chunk would add only a few
new characters, and one document could turn into thousands of nearly identical
chunks (a denial-of-service risk found in security review). With the limit,
and because every window moves forward by at least a quarter of `chunk_size`,
the stored chunk text stays within a few times the document size, and
chunking time grows linearly with the document length.

**Limitation:** characters are **not** tokens. Embedding models measure input
in tokens; 1200 characters of English is very roughly 300 tokens, while
Chinese text is often about one token per character. A token-based size limit
can replace this when an embedding model is chosen.

### Source metadata

Every chunk lists **every** block it contains, in order, using the Batch 3
source location plus the exact character range used from that block's text
(`char_end` is exclusive, like Python slicing):

| Format | Example `source_locations` entry |
|---|---|
| PDF | `{"page": 3, "char_start": 0, "char_end": 812}` |
| DOCX | `{"paragraph": 12, "char_start": 0, "char_end": 240}` |
| TXT | `{"block": 2, "line_start": 5, "line_end": 9, "char_start": 0, "char_end": 318}` |

A chunk that spans pages lists each page. DOCX chunks never get page numbers
(Word files have no fixed pages). Together with `source_filename` and
`file_type`, this is what a later citation feature will use.

### Deterministic chunks and chunk IDs

The same processed document with the same `chunk_size` and `chunk_overlap`
always produces exactly the same chunks, in the same order, with the same IDs.
Nothing random, no timestamps and no LLM are involved. Chunk IDs are readable:

```
8adc0251f64d4cf983e29ce470705707_v1_s1200_o200_00003
```

= `<document_id>_v<chunking_version>_s<chunk_size>_o<chunk_overlap>_<chunk_index>`
(`chunk_index` starts at 0 and is padded to 5 digits).

- Different documents never share chunk IDs (the document ID comes first).
- Different settings give different IDs, so chunks made with different
  settings can never be mistaken for each other later.
- `CHUNKING_VERSION` is increased whenever the algorithm changes.
- `created_at` is stored as metadata only; it is **not** part of the ID.

### Storage and idempotency

Chunks are stored in the same SQLite database, in a `chunks` table:

| Column | Meaning |
|---|---|
| `chunk_id` | Deterministic ID (primary key) |
| `document_id` | Owning document (foreign key to `documents`) |
| `chunk_index` | Position in the document; unique per document |
| `text` | Exactly the text a later stage will embed |
| `char_count` | Length of `text` in characters |
| `source_locations` | JSON list, see above |
| `chunking_version`, `chunk_size`, `chunk_overlap` | Settings that produced the chunk |
| `created_at` | When the chunks were stored |

- The file name and type are not copied into every chunk; they are read from
  the `documents` table.
- Chunking again **replaces** the document's chunks in one transaction, so
  repeated calls never create duplicates (this is called being idempotent).
  `UNIQUE(document_id, chunk_index)` also makes duplicates impossible in the
  database itself.
- Chunking status is **separate** from processing status and is not stored as
  its own column: a document is `chunked` exactly when it has rows in `chunks`.
- If the document is re-processed while it is being chunked, the chunks are
  not saved (`409`), so stored chunks always match the current processed output.

### Current limitations

- Chunk size is counted in characters, not model tokens (see above).
- Chunking is synchronous and runs only when the endpoint is called. Normal
  documents take well under a second, but an extreme file (e.g. millions of
  one-character TXT blocks) can take tens of seconds.
- The whole chunk list is returned at once (no pagination yet).
- There are **no embeddings, no vector database, no semantic retrieval, no
  reranking, and no RAG answer generation** yet. Chunks are stored only.

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
