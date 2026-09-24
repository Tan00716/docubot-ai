"""Document processing: turn an uploaded PDF, DOCX or TXT into normalized text.

Modules:
    models.py    - data shapes (Section, ProcessedDocument, status, errors)
    parsers.py   - text extraction and normalization per file type
    database.py  - SQLite metadata and processing status
    processor.py - the pipeline that ties them together
"""
