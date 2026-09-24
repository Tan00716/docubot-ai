"""Turn raw file bytes (PDF, DOCX, TXT) into normalized Sections.

Parsers only receive bytes and return data. They never touch file paths,
SQLite or HTTP, so they are easy to test and cannot write anywhere.

Extracted text is treated as plain data. It is never executed or interpreted.
"""

import io
import logging
import re
import zipfile

import docx
from pypdf import PdfReader

from app.document_processing.models import (
    DocumentContentError,
    Section,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

# Spaces, tabs, form feeds, vertical tabs and non-breaking spaces.
HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v\u00a0]+")
# Three or more line breaks = more than one blank line in a row.
EXTRA_BLANK_LINES = re.compile(r"\n{3,}")

# A DOCX file is a ZIP archive. A tiny ZIP can unpack to gigabytes (a "zip
# bomb"), so refuse archives whose contents would be larger than this.
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB


# --- Normalization ------------------------------------------------------------


def normalize_line_endings(text: str) -> str:
    """Convert Windows (\\r\\n) and old Mac (\\r) line endings to \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_text(text: str) -> str:
    """Clean up whitespace without changing the meaning of the text.

    - removes null characters
    - normalizes line endings to \\n
    - turns runs of spaces/tabs into one space and trims each line
    - keeps paragraph breaks (one blank line), but collapses extra blank lines
    - never touches letters, digits or punctuation
    """
    text = normalize_line_endings(text.replace("\x00", ""))
    lines = [HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# --- TXT ----------------------------------------------------------------------


def parse_txt(data: bytes) -> list[Section]:
    """Split a UTF-8 text file into blocks separated by blank lines."""
    try:
        # "utf-8-sig" is UTF-8 that also removes an optional BOM at the start.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DocumentContentError("The text file is not valid UTF-8.")

    lines = normalize_line_endings(text.replace("\x00", "")).split("\n")

    sections: list[Section] = []
    block_lines: list[str] = []
    block_start = 0

    def finish_block(last_line: int) -> None:
        location = {
            "block": len(sections) + 1,
            "line_start": block_start,
            "line_end": last_line,
        }
        sections.append(Section(location, normalize_text("\n".join(block_lines))))
        block_lines.clear()

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not block_lines:
                block_start = line_number
            block_lines.append(line)
        elif block_lines:
            finish_block(line_number - 1)

    if block_lines:
        finish_block(len(lines))

    return sections


# --- PDF ----------------------------------------------------------------------


def parse_pdf(data: bytes) -> list[Section]:
    """Extract text page by page. Page numbers are real (1 = first page).

    Pages without extractable text (for example scanned images) are skipped,
    so the remaining page numbers still match the original PDF. No OCR is done.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentContentError("Encrypted PDFs are not supported.")

        sections = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                sections.append(Section({"page": page_number}, text))
        return sections
    except DocumentContentError:
        raise
    except Exception as error:  # pypdf raises many different error types
        logger.warning("PDF parsing failed: %s", type(error).__name__)
        raise DocumentContentError("The PDF file is corrupted or cannot be read.")


# --- DOCX ---------------------------------------------------------------------


def parse_docx(data: bytes) -> list[Section]:
    """Extract body paragraphs in document order.

    DOCX files have no reliable page numbers without rendering them, so the
    location is the paragraph position (1 = first paragraph, empty ones
    included). Tables, headers and footers are not extracted yet.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unpacked_size = sum(entry.file_size for entry in archive.infolist())
    except zipfile.BadZipFile:
        raise DocumentContentError("The DOCX file is invalid or cannot be read.")
    if unpacked_size > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise DocumentContentError("The DOCX file is too large to process.")

    try:
        document = docx.Document(io.BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
    except Exception as error:  # bad zip, missing parts, broken XML, ...
        logger.warning("DOCX parsing failed: %s", type(error).__name__)
        raise DocumentContentError("The DOCX file is invalid or cannot be read.")

    sections = []
    for paragraph_number, raw_text in enumerate(paragraphs, start=1):
        text = normalize_text(raw_text)
        if text:
            sections.append(Section({"paragraph": paragraph_number}, text))
    return sections


# --- Dispatcher ---------------------------------------------------------------

PARSERS = {
    ".txt": parse_txt,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}

EMPTY_DOCUMENT_MESSAGES = {
    ".txt": "The text file contains no text.",
    ".pdf": (
        "No machine-readable text was found in this PDF. "
        "Scanned or image-only PDFs are not supported (no OCR)."
    ),
    ".docx": "The DOCX file contains no paragraph text.",
}


def extract_sections(data: bytes, extension: str) -> list[Section]:
    """Pick the parser for the file type and make sure some text was found."""
    parser = PARSERS.get(extension)
    if parser is None:
        raise UnsupportedFileTypeError()

    sections = parser(data)
    if not sections:
        raise DocumentContentError(EMPTY_DOCUMENT_MESSAGES[extension])
    return sections
