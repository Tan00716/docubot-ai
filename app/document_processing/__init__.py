"""Document processing: turn an uploaded PDF, DOCX or TXT into normalized text,
then split that text into chunks.

Modules:
    models.py    - data shapes (Section, ProcessedDocument, Chunk, status, errors)
    parsers.py   - text extraction and normalization per file type
    chunking.py  - the chunking algorithm and its configuration (no I/O)
    database.py  - SQLite metadata, processing status and chunks
    processor.py - the pipelines that tie them together
"""
