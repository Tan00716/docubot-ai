"""Tests for the chunking algorithm and chunk IDs (pure functions, no storage).

Storage and pipeline tests are in test_chunk_storage.py.

No files, no SQLite, no Telegram, no .env, no network, no LLM.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import json
import random
import time
import unittest

from app.document_processing.chunking import (
    BLOCK_SEPARATOR,
    CHUNKING_VERSION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMPTY_DOCUMENT_MESSAGE,
    MAX_CHUNK_SIZE,
    ChunkingConfig,
    build_chunks,
    make_chunk_id,
)
from app.document_processing.models import (
    DocumentContentError,
    InvalidChunkingConfigError,
    ProcessedDocument,
    Section,
)

DOC_A = "a" * 32
DOC_B = "b" * 32


def make_document(texts, document_id=DOC_A, file_type=".txt"):
    """A ProcessedDocument with one TXT-style block per text."""
    sections = tuple(
        Section({"block": number, "line_start": number, "line_end": number}, text)
        for number, text in enumerate(texts, start=1)
    )
    return ProcessedDocument(document_id, f"sample{file_type}", file_type, sections)


def words(count, prefix="word"):
    """Deterministic filler text: "word0 word1 word2 ..."."""
    return " ".join(f"{prefix}{number}" for number in range(count))


def spans(chunks):
    """(char_start, char_end) of every source location, in order."""
    return [
        (location["char_start"], location["char_end"])
        for chunk in chunks
        for location in chunk.source_locations
    ]


def block_numbers(chunk):
    return [location["block"] for location in chunk.source_locations]


def location_key(location):
    """The Batch 3 part of a chunk location (without char_start/char_end)."""
    return tuple(sorted((k, v) for k, v in location.items() if not k.startswith("char_")))


class ChunkInvariantsMixin:
    """Checks that must hold for ANY document and ANY valid configuration."""

    def assert_valid_chunks(self, document, chunks, config):
        sections = {location_key(s.source_location): s for s in document.sections}
        covered = {key: [False] * len(s.text) for key, s in sections.items()}

        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        for chunk in chunks:
            # never empty, never longer than chunk_size, no outer whitespace
            self.assertTrue(chunk.text)
            self.assertEqual(chunk.char_count, len(chunk.text))
            self.assertLessEqual(chunk.char_count, config.chunk_size)
            self.assertEqual(chunk.text, chunk.text.strip())

            # every chunk is exactly its source slices joined together
            pieces = []
            for location in chunk.source_locations:
                section = sections[location_key(location)]
                start, end = location["char_start"], location["char_end"]
                self.assertLess(start, end)
                pieces.append(section.text[start:end])
                covered[location_key(location)][start:end] = [True] * (end - start)
            self.assertEqual(BLOCK_SEPARATOR.join(pieces), chunk.text)

        # no text loss: every non-whitespace character is in some chunk
        for key, section in sections.items():
            for position, character in enumerate(section.text):
                if not character.isspace():
                    self.assertTrue(covered[key][position], f"lost {character!r} at {position}")

        # overlap never produces a chunk that only repeats the previous one:
        # each chunk uses at least one source span the previous chunk did not.
        # (Texts may still look alike, e.g. two windows over "xxxx...".)
        def used_spans(chunk):
            return {(location_key(loc), loc["char_start"], loc["char_end"])
                    for loc in chunk.source_locations}

        for previous, current in zip(chunks, chunks[1:]):
            self.assertFalse(used_spans(current) <= used_spans(previous), "duplicate-only chunk")


# --- Configuration ------------------------------------------------------------


class ChunkingConfigTests(unittest.TestCase):
    def test_defaults_are_1200_and_200_characters(self):
        config = ChunkingConfig()

        self.assertEqual((config.chunk_size, config.chunk_overlap), (1200, 200))
        self.assertEqual((DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP), (1200, 200))

    def test_custom_values_are_accepted(self):
        for size, overlap in [(1, 0), (500, 0), (500, 250), (7, 3), (MAX_CHUNK_SIZE, 100)]:
            with self.subTest(size=size, overlap=overlap):
                config = ChunkingConfig(chunk_size=size, chunk_overlap=overlap)

                self.assertEqual((config.chunk_size, config.chunk_overlap), (size, overlap))

    def test_invalid_values_are_rejected_with_clear_messages(self):
        cases = [
            (0, 0, "chunk_size must be a positive whole number."),
            (-5, 0, "chunk_size must be a positive whole number."),
            (MAX_CHUNK_SIZE + 1, 0, f"chunk_size must be at most {MAX_CHUNK_SIZE} characters."),
            (100, -1, "chunk_overlap must be zero or a positive whole number."),
            (100, 51, "chunk_overlap must be at most half of chunk_size."),
            (100, 100, "chunk_overlap must be at most half of chunk_size."),
            (100, 150, "chunk_overlap must be at most half of chunk_size."),
            (1, 1, "chunk_overlap must be at most half of chunk_size."),
            (1.5, 0, "chunk_size must be a positive whole number."),
            ("100", 0, "chunk_size must be a positive whole number."),
            (True, 0, "chunk_size must be a positive whole number."),
            (100, 2.5, "chunk_overlap must be zero or a positive whole number."),
        ]
        for size, overlap, message in cases:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(InvalidChunkingConfigError) as caught:
                    ChunkingConfig(chunk_size=size, chunk_overlap=overlap)

                self.assertEqual(caught.exception.safe_message, message)


# --- Algorithm ----------------------------------------------------------------


class BasicChunkingTests(ChunkInvariantsMixin, unittest.TestCase):
    def test_small_document_becomes_one_chunk(self):
        document = make_document(["Hello DocuBot.", "Second block."])

        chunks = build_chunks(document, ChunkingConfig())

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello DocuBot.\n\nSecond block.")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].char_count, len("Hello DocuBot.\n\nSecond block."))

    def test_large_document_becomes_several_chunks(self):
        config = ChunkingConfig()
        document = make_document([words(50, f"b{n}w") for n in range(20)])

        chunks = build_chunks(document, config)

        self.assertGreater(len(chunks), 3)
        self.assert_valid_chunks(document, chunks, config)

    def test_smaller_chunk_size_gives_more_and_smaller_chunks(self):
        document = make_document([words(30, f"b{n}w") for n in range(10)])
        large = build_chunks(document, ChunkingConfig(1200, 0))
        small = build_chunks(document, ChunkingConfig(400, 0))

        self.assertGreater(len(small), len(large))
        self.assertTrue(all(chunk.char_count <= 400 for chunk in small))
        self.assertTrue(any(chunk.char_count > 400 for chunk in large))

    def test_chunk_carries_document_and_configuration_metadata(self):
        document = make_document(["Some text."], file_type=".txt")
        config = ChunkingConfig(800, 100)

        chunk = build_chunks(document, config)[0]

        self.assertEqual(chunk.document_id, DOC_A)
        self.assertEqual(chunk.source_filename, "sample.txt")
        self.assertEqual(chunk.file_type, ".txt")
        self.assertEqual(chunk.chunking_version, CHUNKING_VERSION)
        self.assertEqual((chunk.chunk_size, chunk.chunk_overlap), (800, 100))


class BoundaryTests(unittest.TestCase):
    def test_blocks_are_packed_until_the_next_would_not_fit(self):
        blocks = ["a" * 500, "b" * 500, "c" * 500]

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1200, 0))

        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1, 2], [3]])
        self.assertEqual(chunks[0].text, "a" * 500 + "\n\n" + "b" * 500)

    def test_chunk_exactly_at_chunk_size_is_allowed(self):
        # 599 + 2 (separator) + 599 = 1200 characters exactly
        blocks = ["a" * 599, "b" * 599, "c" * 10]

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1200, 0))

        self.assertEqual(chunks[0].char_count, 1200)
        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1, 2], [3]])

    def test_one_character_too_many_starts_a_new_chunk(self):
        blocks = ["a" * 600, "b" * 599]  # 600 + 2 + 599 = 1201

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1200, 0))

        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1], [2]])

    def test_small_blocks_are_never_cut(self):
        document = make_document([words(40, f"b{n}w") for n in range(12)])

        chunks = build_chunks(document, ChunkingConfig(1200, 200))

        for chunk in chunks:
            for location in chunk.source_locations:
                section = document.sections[location["block"] - 1]
                self.assertEqual((location["char_start"], location["char_end"]),
                                 (0, len(section.text)))


class OverlapTests(unittest.TestCase):
    def test_no_overlap_means_no_repeated_blocks(self):
        document = make_document([f"Block {n:02d} has a few words." for n in range(30)])

        chunks = build_chunks(document, ChunkingConfig(100, 0))

        all_blocks = [n for chunk in chunks for n in block_numbers(chunk)]
        self.assertEqual(all_blocks, list(range(1, 31)))

    def test_overlap_repeats_last_whole_blocks_that_fit(self):
        document = make_document([f"Block {n:02d} has a few words." for n in range(30)])
        config = ChunkingConfig(100, 50)

        chunks = build_chunks(document, config)

        for previous, current in zip(chunks, chunks[1:]):
            shared = set(block_numbers(previous)) & set(block_numbers(current))
            self.assertTrue(shared, "neighbouring chunks should share a block")
            # the shared blocks are the end of one chunk and the start of the next
            self.assertEqual(block_numbers(previous)[-len(shared):], sorted(shared))
            self.assertEqual(block_numbers(current)[:len(shared)], sorted(shared))
            shared_text = BLOCK_SEPARATOR.join(document.sections[n - 1].text for n in sorted(shared))
            self.assertLessEqual(len(shared_text), config.chunk_overlap)

    def test_large_overlap_repeats_a_whole_block(self):
        blocks = ["a" * 500, "b" * 500, "c" * 500]

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1200, 600))

        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1, 2], [2, 3]])

    def test_carried_overlap_never_pushes_a_chunk_over_chunk_size(self):
        # After [1, 2] is full, block 2 (300) fits into the overlap (500), but
        # 300 + 2 + 800 > 1000: block 2 must be dropped so block 3 fits.
        blocks = ["a" * 600, "b" * 300, "c" * 800]

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1000, 500))

        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1, 2], [3]])
        self.assertTrue(all(chunk.char_count <= 1000 for chunk in chunks))

    def test_block_bigger_than_overlap_is_not_repeated(self):
        blocks = ["a" * 500, "b" * 500, "c" * 500]

        chunks = build_chunks(make_document(blocks), ChunkingConfig(1200, 200))

        self.assertEqual([block_numbers(chunk) for chunk in chunks], [[1, 2], [3]])


class OversizedBlockTests(ChunkInvariantsMixin, unittest.TestCase):
    def test_block_without_whitespace_is_cut_into_exact_windows(self):
        text = "0123456789" * 1000  # 10,000 characters, no spaces at all

        chunks = build_chunks(make_document([text]), ChunkingConfig(1200, 200))

        # each window is 1200 characters and starts 1000 after the previous one
        expected = [(start, min(start + 1200, 10_000)) for start in range(0, 10_000, 1000)]
        self.assertEqual(spans(chunks), expected)
        self.assertEqual(chunks[1].text, text[1000:2200])

    def test_long_block_is_cut_between_words_with_overlap(self):
        text = words(700)  # about 5,000 characters
        config = ChunkingConfig(1200, 200)
        document = make_document([text])

        chunks = build_chunks(document, config)

        self.assertGreater(len(chunks), 3)
        self.assert_valid_chunks(document, chunks, config)
        for start, end in spans(chunks):
            self.assertTrue(start == 0 or text[start - 1] == " ", "window starts mid-word")
            self.assertTrue(end == len(text) or text[end] == " ", "window ends mid-word")
        for (_, previous_end), (next_start, _) in zip(spans(chunks), spans(chunks)[1:]):
            overlap = previous_end - next_start
            self.assertGreater(overlap, 0)
            self.assertLessEqual(overlap, config.chunk_overlap)

    def test_long_block_without_overlap_has_no_repeated_text(self):
        text = words(700)

        chunks = build_chunks(make_document([text]), ChunkingConfig(1200, 0))

        for (_, previous_end), (next_start, _) in zip(spans(chunks), spans(chunks)[1:]):
            self.assertGreaterEqual(next_start, previous_end)
        self.assertEqual(" ".join(chunk.text for chunk in chunks), text)

    def test_oversized_block_is_not_mixed_with_neighbouring_blocks(self):
        document = make_document(["Intro.", words(400), "Outro."])

        chunks = build_chunks(document, ChunkingConfig(1200, 200))

        self.assertEqual(block_numbers(chunks[0]), [1])
        self.assertEqual(block_numbers(chunks[-1]), [3])
        for chunk in chunks[1:-1]:
            self.assertEqual(block_numbers(chunk), [2])

    def test_very_large_block_is_chunked_completely(self):
        config = ChunkingConfig()
        document = make_document([words(40_000)])  # about 300,000 characters

        chunks = build_chunks(document, config)

        self.assertGreater(len(chunks), 250)
        self.assert_valid_chunks(document, chunks, config)

    def test_tiny_chunk_size_still_terminates(self):
        document = make_document(["abc def", "g"])
        config = ChunkingConfig(1, 0)

        chunks = build_chunks(document, config)

        self.assertEqual([chunk.text for chunk in chunks], list("abcdefg"))


class WhitespaceAndEmptyBlockTests(unittest.TestCase):
    def test_empty_and_whitespace_only_blocks_are_skipped(self):
        document = make_document(["", "First.", "   \n\t ", "Second."])

        chunks = build_chunks(document, ChunkingConfig())

        self.assertEqual(chunks[0].text, "First.\n\nSecond.")
        self.assertEqual(block_numbers(chunks[0]), [2, 4])

    def test_document_without_any_text_is_rejected(self):
        for texts in [[], [""], ["  ", "\n\n"]]:
            with self.subTest(texts=texts):
                with self.assertRaises(DocumentContentError) as caught:
                    build_chunks(make_document(texts), ChunkingConfig())

                self.assertEqual(caught.exception.safe_message, EMPTY_DOCUMENT_MESSAGE)

    def test_whitespace_inside_a_block_is_kept_exactly(self):
        text = "line one\nline two\n\nnext paragraph, with punctuation: 100% & 你好。"

        chunks = build_chunks(make_document([text]), ChunkingConfig())

        self.assertEqual(chunks[0].text, text)

    def test_outer_whitespace_is_trimmed_and_offsets_stay_correct(self):
        chunks = build_chunks(make_document(["  padded  "]), ChunkingConfig())

        self.assertEqual(chunks[0].text, "padded")
        self.assertEqual(spans(chunks), [(2, 8)])

    def test_no_chunk_is_empty_or_starts_or_ends_with_whitespace(self):
        document = make_document([words(300), " x ", words(5), "\n" + words(200) + "\n"])

        for config in [ChunkingConfig(1200, 200), ChunkingConfig(50, 20), ChunkingConfig(7, 3)]:
            with self.subTest(config=config):
                for chunk in build_chunks(document, config):
                    self.assertTrue(chunk.text.strip())
                    self.assertEqual(chunk.text, chunk.text.strip())


class NoTextLossTests(ChunkInvariantsMixin, unittest.TestCase):
    def test_without_overlap_chunks_rebuild_the_document_exactly(self):
        texts = [words(n, f"s{n}w") for n in range(1, 30)]  # every block <= 300 chars
        document = make_document(texts)

        chunks = build_chunks(document, ChunkingConfig(300, 0))

        self.assertTrue(all(len(text) <= 300 for text in texts))
        self.assertEqual(BLOCK_SEPARATOR.join(c.text for c in chunks), BLOCK_SEPARATOR.join(texts))

    def test_random_documents_keep_all_invariants(self):
        # Seeded random input: different shapes each seed, same result every run.
        for seed in range(150):
            rng = random.Random(seed)
            texts = []
            for _ in range(rng.randint(1, 8)):
                tokens = [
                    "x" * rng.randint(1, 3000) if rng.random() < 0.02
                    else "w" * rng.randint(1, 12)
                    # short, medium and long blocks relative to chunk_size
                    for _ in range(rng.randint(0, rng.choice([5, 40, 150, 400])))
                ]
                separators = [rng.choice([" ", " ", " ", "\n", "\n\n"]) for _ in tokens]
                texts.append("".join(t + s for t, s in zip(tokens, separators)).strip())
            size = rng.randint(1, 1500)
            config = ChunkingConfig(size, rng.randint(0, size // 2))
            document = make_document(texts)
            if not any(text.strip() for text in texts):
                continue

            with self.subTest(seed=seed, config=config):
                self.assert_valid_chunks(document, build_chunks(document, config), config)


class DeterminismTests(unittest.TestCase):
    def test_same_input_and_configuration_give_identical_chunks(self):
        document = make_document([words(300), "Short block.", words(90)])
        copy = ProcessedDocument.from_dict(json.loads(json.dumps(document.to_dict())))

        first = build_chunks(document, ChunkingConfig())
        second = build_chunks(copy, ChunkingConfig())

        self.assertEqual(first, second)

    def test_chunk_ids_follow_the_documented_format(self):
        chunks = build_chunks(make_document([words(400)]), ChunkingConfig(1200, 200))

        self.assertEqual(chunks[0].chunk_id, f"{DOC_A}_v{CHUNKING_VERSION}_s1200_o200_00000")
        self.assertEqual(chunks[2].chunk_id, f"{DOC_A}_v{CHUNKING_VERSION}_s1200_o200_00002")
        self.assertEqual(make_chunk_id(DOC_B, 7, ChunkingConfig(500, 50)),
                         f"{DOC_B}_v{CHUNKING_VERSION}_s500_o50_00007")

    def test_chunk_ids_are_stable_and_unique_within_a_document(self):
        document = make_document([words(1000)])

        first = [c.chunk_id for c in build_chunks(document, ChunkingConfig())]
        second = [c.chunk_id for c in build_chunks(document, ChunkingConfig())]

        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_different_configuration_gives_different_ids(self):
        document = make_document([words(1000)])

        default_ids = {c.chunk_id for c in build_chunks(document, ChunkingConfig(1200, 200))}
        other_ids = {c.chunk_id for c in build_chunks(document, ChunkingConfig(1200, 100))}

        self.assertFalse(default_ids & other_ids)

    def test_same_text_in_different_documents_never_collides(self):
        texts = [words(1000)]

        ids_a = {c.chunk_id for c in build_chunks(make_document(texts, DOC_A), ChunkingConfig())}
        ids_b = {c.chunk_id for c in build_chunks(make_document(texts, DOC_B), ChunkingConfig())}

        self.assertFalse(ids_a & ids_b)


class SourceLocationTests(unittest.TestCase):
    def test_chunk_spanning_blocks_lists_every_block_in_order(self):
        document = make_document(["One.", "Two.", "Three."])

        chunk = build_chunks(document, ChunkingConfig())[0]

        self.assertEqual(
            list(chunk.source_locations),
            [
                {"block": 1, "line_start": 1, "line_end": 1, "char_start": 0, "char_end": 4},
                {"block": 2, "line_start": 2, "line_end": 2, "char_start": 0, "char_end": 4},
                {"block": 3, "line_start": 3, "line_end": 3, "char_start": 0, "char_end": 6},
            ],
        )

    def test_pdf_pages_are_preserved_and_a_chunk_can_span_pages(self):
        document = ProcessedDocument(DOC_A, "guide.pdf", ".pdf", (
            Section({"page": 2}, "Page two text."),
            Section({"page": 3}, "Page three text."),
        ))

        chunk = build_chunks(document, ChunkingConfig())[0]

        self.assertEqual([loc["page"] for loc in chunk.source_locations], [2, 3])
        for location in chunk.source_locations:
            self.assertNotIn("paragraph", location)

    def test_docx_paragraphs_are_preserved_without_page_numbers(self):
        document = ProcessedDocument(DOC_A, "memo.docx", ".docx", (
            Section({"paragraph": 1}, "Intro."),
            Section({"paragraph": 4}, "Body."),
        ))

        chunk = build_chunks(document, ChunkingConfig())[0]

        self.assertEqual([loc["paragraph"] for loc in chunk.source_locations], [1, 4])
        for location in chunk.source_locations:
            self.assertNotIn("page", location)


class ResourceLimitTests(ChunkInvariantsMixin, unittest.TestCase):
    """Chunking must stay fast and must not blow up the stored text size.

    Regression tests for a denial-of-service found in security review: with an
    overlap close to chunk_size, every chunk added only ~1 new character.
    """

    def assert_bounded(self, document, config, seconds=10):
        started = time.perf_counter()
        chunks = build_chunks(document, config)
        elapsed = time.perf_counter() - started

        # The document's text as one long string, blocks joined by BLOCK_SEPARATOR
        # (with many tiny blocks, most of a chunk can be separators).
        texts = [section.text for section in document.sections]
        document_length = len(BLOCK_SEPARATOR.join(texts))
        stored_length = sum(chunk.char_count for chunk in chunks)
        # Overlap repeats text, but never more than a few times the document.
        self.assertLessEqual(stored_length, 4 * document_length + config.chunk_size)
        self.assertLess(elapsed, seconds, "chunking took too long")
        return chunks

    def test_huge_block_without_whitespace_and_maximum_overlap(self):
        config = ChunkingConfig(MAX_CHUNK_SIZE, MAX_CHUNK_SIZE // 2)
        document = make_document(["x" * 2_000_000])

        chunks = self.assert_bounded(document, config)

        self.assertLessEqual(len(chunks), 2_000_000 // (MAX_CHUNK_SIZE // 4) + 1)

    def test_many_tiny_blocks_with_maximum_overlap(self):
        config = ChunkingConfig(MAX_CHUNK_SIZE, MAX_CHUNK_SIZE // 2)
        document = make_document(["x"] * 300_000)

        self.assert_bounded(document, config)

    def test_long_words_text_with_maximum_overlap_keeps_invariants(self):
        config = ChunkingConfig(1000, 500)
        document = make_document([words(20_000), "y" * 50_000])

        chunks = self.assert_bounded(document, config)

        self.assert_valid_chunks(document, chunks, config)

    def test_every_window_moves_forward_by_at_least_a_quarter_of_chunk_size(self):
        for config in [ChunkingConfig(1000, 500), ChunkingConfig(1200, 200), ChunkingConfig(9, 4)]:
            with self.subTest(config=config):
                text = words(3000)
                chunks = build_chunks(make_document([text]), config)

                starts = [start for start, _ in spans(chunks)]
                for previous, current in zip(starts, starts[1:]):
                    self.assertGreaterEqual(current - previous, config.chunk_size // 4)


if __name__ == "__main__":
    unittest.main()
