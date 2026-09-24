"""Chunking: split a processed document into smaller pieces called chunks.

Why: later stages (Embedding, Vector Search, RAG) work on short pieces of
text, not whole documents. A chunk must be small enough to embed, and must
remember exactly where its text came from so answers can cite the source.

The chunker is pure and deterministic: it only receives a ProcessedDocument
and a ChunkingConfig and returns data. It never reads files, touches SQLite,
calls an LLM, uses random numbers or looks at the clock. The same document
and the same configuration always give exactly the same chunks.

Algorithm (sizes are counted in characters):

1. Blocks are the Batch 3 sections (PDF pages, DOCX paragraphs, TXT blocks).
   Whitespace-only blocks carry no text and are skipped.
2. Small blocks are packed into one chunk, joined by BLOCK_SEPARATOR, until
   the next block would make the chunk longer than chunk_size. Blocks are
   never cut in this case.
3. When a chunk is full, the next chunk starts with the last whole blocks of
   the previous chunk that fit into chunk_overlap characters (possibly none).
4. A block longer than chunk_size is split on its own into windows of at most
   chunk_size characters. Each window ends at a space or line break when there
   is one in its later part (so words stay whole), otherwise exactly after
   chunk_size characters. The next window starts up to chunk_overlap
   characters before the previous one ended, moved forward to a word start.
   Windows are never packed together with other blocks.

chunk_overlap may be at most half of chunk_size, and every window moves
forward by at least a quarter of chunk_size. So the stored chunk text stays
within a few times the size of the document, and chunking runs in time
proportional to the document length.

Text is never rewritten: every chunk is made of exact slices of section text.
"""

import re
from dataclasses import dataclass

from app.document_processing.models import (
    Chunk,
    DocumentContentError,
    InvalidChunkingConfigError,
    ProcessedDocument,
    Section,
)

# --- Configuration ------------------------------------------------------------
#
# This is the one place to change the defaults. Sizes are measured in
# characters (Python len()), NOT in model tokens. For English text one token is
# roughly 4 characters, so 1200 characters is very roughly 300 tokens; for
# Chinese text one character is often about one token.

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# Chunks larger than this would be too big for typical embedding models.
MAX_CHUNK_SIZE = 10_000

# Bump this whenever the algorithm changes in a way that makes the same
# document + configuration produce different chunks. It is part of chunk_id.
CHUNKING_VERSION = 1

# Placed between two blocks that share one chunk (a paragraph break).
BLOCK_SEPARATOR = "\n\n"

EMPTY_DOCUMENT_MESSAGE = "The processed document contains no text to chunk."

WHITESPACE = re.compile(r"\s")


def _is_whole_number(value: object) -> bool:
    # bool is a subclass of int in Python; True must not count as 1.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ChunkingConfig:
    """How big chunks are and how much neighbouring chunks overlap.

    Invalid values are rejected when the object is created, so a
    ChunkingConfig that exists is always valid.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def __post_init__(self) -> None:
        if not _is_whole_number(self.chunk_size) or self.chunk_size <= 0:
            raise InvalidChunkingConfigError("chunk_size must be a positive whole number.")
        if self.chunk_size > MAX_CHUNK_SIZE:
            raise InvalidChunkingConfigError(
                f"chunk_size must be at most {MAX_CHUNK_SIZE} characters."
            )
        if not _is_whole_number(self.chunk_overlap) or self.chunk_overlap < 0:
            raise InvalidChunkingConfigError(
                "chunk_overlap must be zero or a positive whole number."
            )
        # The overlap is repeated in two chunks. With an overlap close to
        # chunk_size, every chunk would add only a few new characters, and one
        # document could turn into thousands of nearly identical chunks. At most
        # half of chunk_size keeps the stored text within a few times the
        # document size (it also guarantees overlap < chunk_size).
        if self.chunk_overlap > self.chunk_size // 2:
            raise InvalidChunkingConfigError("chunk_overlap must be at most half of chunk_size.")


# --- Chunk IDs ----------------------------------------------------------------


def make_chunk_id(document_id: str, chunk_index: int, config: ChunkingConfig) -> str:
    """Build a readable, deterministic chunk ID.

    Example: "8adc0251f64d4cf983e29ce470705707_v1_s1200_o200_00003"
             document_id _ version _ size _ overlap _ chunk_index

    - Same document + same configuration -> same IDs, every time.
    - Different documents never collide, because document_id is unique and
      is the first part of the ID.
    - A different configuration gives different IDs, so chunks made with
      different settings can never be mistaken for each other later.
    """
    return (
        f"{document_id}_v{CHUNKING_VERSION}"
        f"_s{config.chunk_size}_o{config.chunk_overlap}_{chunk_index:05d}"
    )


# --- Pieces -------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    """A continuous slice of one section's text: section.text[start:end]."""

    section: Section
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.section.text[self.start:self.end]

    @property
    def location(self) -> dict[str, int]:
        return {
            **self.section.source_location,
            "char_start": self.start,
            "char_end": self.end,
        }


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink text[start:end] so it neither starts nor ends with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _joined_length(pieces: list[_Piece]) -> int:
    """Length of the pieces' text once joined with BLOCK_SEPARATOR."""
    if not pieces:
        return 0
    text_length = sum(piece.end - piece.start for piece in pieces)
    return text_length + len(BLOCK_SEPARATOR) * (len(pieces) - 1)


# --- Splitting one oversized block --------------------------------------------


def _choose_cut(text: str, start: int, stop: int, config: ChunkingConfig) -> int:
    """Where the window that begins at `start` should end.

    Prefer the last space or line break inside the window, so words stay
    whole. Only the later part of the window is searched, so windows do not
    get too short, and the next window (which starts chunk_overlap characters
    before the cut) moves forward by at least min_step characters. Without
    such whitespace (e.g. one very long URL), cut exactly after chunk_size
    characters.
    """
    hard_end = start + config.chunk_size
    if hard_end >= stop:
        return stop
    min_step = max(1, (config.chunk_size - config.chunk_overlap) // 2)
    earliest = start + max(config.chunk_size // 2, config.chunk_overlap + min_step)
    # A whitespace right after the window (index hard_end) is also a clean cut.
    cut = max(
        text.rfind(" ", earliest, hard_end + 1),
        text.rfind("\n", earliest, hard_end + 1),
    )
    return cut if cut != -1 else hard_end


def _choose_next_start(text: str, cut: int, overlap: int) -> int:
    """Start of the next window: `overlap` characters before `cut`.

    The start is moved forward to the beginning of a word, so a chunk does not
    begin with half a word. The real overlap is therefore at most `overlap`.
    """
    if overlap == 0:
        return cut
    start = cut - overlap
    if text[start - 1].isspace():
        return start
    whitespace = WHITESPACE.search(text, start, cut)
    if whitespace:
        return whitespace.end()
    return start  # the overlap is inside one long word: keep it as it is


def _split_oversized(section: Section, start: int, stop: int, config: ChunkingConfig) -> list[_Piece]:
    """Split section.text[start:stop] (longer than chunk_size) into windows."""
    text = section.text
    pieces: list[_Piece] = []
    while start < stop:
        cut = _choose_cut(text, start, stop, config)
        piece_start, piece_end = _trimmed_bounds(text, start, cut)
        if piece_start < piece_end:
            pieces.append(_Piece(section, piece_start, piece_end))
        if cut >= stop:
            break

        next_start, _ = _trimmed_bounds(text, _choose_next_start(text, cut, config.chunk_overlap), stop)
        # Safety guard: every window must move forward, so this loop always ends.
        if next_start <= start:
            raise RuntimeError("Chunking made no progress.")
        start = next_start
    return pieces


# --- Grouping pieces into chunks ----------------------------------------------


def _overlap_tail(pieces: list[_Piece], limit: int) -> list[_Piece]:
    """The last whole pieces whose joined length is at most `limit` characters."""
    count = 0
    length = 0
    for piece in reversed(pieces):
        added = (piece.end - piece.start) + (len(BLOCK_SEPARATOR) if count else 0)
        if length + added > limit:
            break
        length += added
        count += 1
    return pieces[len(pieces) - count:]


def _add_group(groups: list[list[_Piece]], group: list[_Piece]) -> None:
    """Append a finished chunk, refusing one that adds no new text."""
    # Safety guard: overlap may repeat pieces, but a chunk must never consist
    # only of pieces that the previous chunk already contained. Carried-over
    # pieces are the very same objects, so compare identity (id), not value.
    if groups:
        previous = {id(piece) for piece in groups[-1]}
        if all(id(piece) in previous for piece in group):
            raise RuntimeError("Chunking produced a duplicate-only chunk.")
    groups.append(group)


def group_pieces(sections: tuple[Section, ...], config: ChunkingConfig) -> list[list[_Piece]]:
    """Decide which pieces of text form each chunk, in document order."""
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_length = 0  # length of the current chunk's text, kept up to date

    for section in sections:
        start, end = _trimmed_bounds(section.text, 0, len(section.text))
        if start == end:
            continue  # whitespace-only block: there is no text to keep

        if end - start > config.chunk_size:
            # Rule 4: an oversized block is split on its own.
            if current:
                _add_group(groups, current)
                current, current_length = [], 0
            for piece in _split_oversized(section, start, end, config):
                _add_group(groups, [piece])
            continue

        piece = _Piece(section, start, end)
        piece_length = end - start
        separator = len(BLOCK_SEPARATOR)
        if current and current_length + separator + piece_length > config.chunk_size:
            _add_group(groups, current)
            # Carry over the last whole pieces that fit into chunk_overlap AND
            # still leave room for the new piece.
            room_left = config.chunk_size - separator - piece_length
            current = _overlap_tail(current, min(config.chunk_overlap, room_left))
            current_length = _joined_length(current)
        current_length += (separator if current else 0) + piece_length
        current.append(piece)

    if current:
        _add_group(groups, current)
    return groups


def build_chunks(document: ProcessedDocument, config: ChunkingConfig) -> list[Chunk]:
    """Split a processed document into chunks.

    Raises DocumentContentError if the document has no text at all.
    """
    groups = group_pieces(document.sections, config)
    if not groups:
        raise DocumentContentError(EMPTY_DOCUMENT_MESSAGE)

    chunks = []
    for chunk_index, pieces in enumerate(groups):
        text = BLOCK_SEPARATOR.join(piece.text for piece in pieces)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(document.document_id, chunk_index, config),
                document_id=document.document_id,
                chunk_index=chunk_index,
                text=text,
                char_count=len(text),
                source_filename=document.source_filename,
                file_type=document.file_type,
                source_locations=tuple(piece.location for piece in pieces),
                chunking_version=CHUNKING_VERSION,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        )
    return chunks
