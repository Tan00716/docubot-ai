"""Build small, real PDF and DOCX files in memory for tests.

Generating files in code (instead of committing binary fixtures) keeps the
test data visible and easy to change.
"""

import io
import zipfile

import docx
from pypdf import PdfReader, PdfWriter


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(page_texts: list[str | None]) -> bytes:
    """Build a valid PDF with one page per item.

    Each item is the text of that page ("\\n" starts a new line), or None
    for a page that has no text at all (like a scanned image page).
    Only ASCII text is supported (standard Helvetica font).
    """
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_numbers = []
    next_number = 4
    for text in page_texts:
        page_number, content_number = next_number, next_number + 1
        next_number += 2
        page_numbers.append(page_number)

        commands = ""
        if text is not None:
            lines = " 0 -14 Td ".join(f"({_pdf_escape(line)}) Tj" for line in text.split("\n"))
            commands = f"BT /F1 12 Tf 72 720 Td {lines} ET"
        stream = commands.encode("latin-1")

        objects[page_number] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_number} 0 R >>"
        ).encode()
        objects[content_number] = (
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode()

    output = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"

    xref_offset = len(output)
    size = max(objects) + 1
    output += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for number in range(1, size):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(output)


def make_encrypted_pdf(page_texts: list[str | None]) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(page_texts))))
    writer.encrypt("secret-password", algorithm="RC4-128")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx(paragraphs: list[str]) -> bytes:
    """Build a valid DOCX with one body paragraph per item."""
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip_without_word_document() -> bytes:
    """A valid ZIP file that is NOT a Word document."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a word document")
    return buffer.getvalue()
