"""Versioned, checksum-verified retrieval evaluation datasets.

The JSON artifacts are the reviewable source of truth for evaluations. Python
records remain the execution representation; the loader refuses a changed
artifact until its embedded SHA-256 is deliberately refreshed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.corpus import PASSAGES, Passage
from app.evaluation.expansion_v2 import HARD_NEGATIVES_V2, PASSAGES_V2, QUERIES_V2
from app.evaluation.golden import QUERIES, Evidence, GoldenQuery, direction_of

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
VERSIONS = ("retrieval_eval_v1", "retrieval_eval_v2")


@dataclass(frozen=True)
class Dataset:
    version: str
    passages: tuple[Passage, ...]
    queries: tuple[GoldenQuery, ...]
    sha256: str
    hard_negatives: dict[str, tuple[str, ...]]


def _payload(version: str, passages: tuple[Passage, ...],
             queries: tuple[GoldenQuery, ...]) -> dict[str, Any]:
    return {
        "version": version,
        "corpus": [{"key": p.key, "language": p.language, "text": p.text} for p in passages],
        "queries": [{
            "query_id": q.query_id, "language": q.language, "text": q.text,
            "target_language": direction_of(q, passages).split("->", 1)[1],
            "relevant": [{"passage_key": e.passage_key, "evidence": e.text}
                         for e in q.relevant],
            "secondary": [{"passage_key": e.passage_key, "evidence": e.text}
                          for e in q.secondary],
        } for q in queries],
        "hard_negatives": ({} if version == "retrieval_eval_v1" else
                           {key: list(value) for key, value in HARD_NEGATIVES_V2.items()}),
    }


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def expected_dataset(version: str) -> Dataset:
    if version == "retrieval_eval_v1":
        passages, queries = tuple(PASSAGES), tuple(QUERIES)
    elif version == "retrieval_eval_v2":
        passages, queries = tuple(PASSAGES) + tuple(PASSAGES_V2), tuple(QUERIES) + tuple(QUERIES_V2)
    else:
        raise ValueError(f"Unknown dataset version {version!r}; expected one of {VERSIONS}.")
    payload = _payload(version, passages, queries)
    hard_negatives = {} if version == "retrieval_eval_v1" else HARD_NEGATIVES_V2
    return Dataset(version, passages, queries, canonical_sha256(payload), hard_negatives)


def verify_dataset(version: str, directory: Path = DATASET_DIR) -> Dataset:
    """Verify artifact integrity and that Python execution records match it."""
    expected = expected_dataset(version)
    path = directory / f"{version}.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    actual_hash = artifact.get("sha256")
    payload = {key: value for key, value in artifact.items() if key != "sha256"}
    if artifact.get("version") != version or actual_hash != canonical_sha256(payload):
        raise ValueError(f"{version}: artifact checksum mismatch; update the version explicitly.")
    if payload != _payload(version, expected.passages, expected.queries):
        raise ValueError(f"{version}: artifact differs from the execution records.")
    if actual_hash != expected.sha256:
        raise ValueError(f"{version}: source records changed; update the version explicitly.")
    return Dataset(version, expected.passages, expected.queries, actual_hash,
                   expected.hard_negatives)


def write_artifact(version: str, directory: Path = DATASET_DIR, *, force: bool = False) -> Path:
    """Explicit maintainer operation used only when establishing a new version."""
    expected = expected_dataset(version)
    payload = _payload(version, expected.passages, expected.queries)
    payload["sha256"] = canonical_sha256(payload)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.json"
    if path.exists() and not force:
        raise FileExistsError(f"{path.name} is immutable; choose a new dataset version.")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def from_payload(payload: dict[str, Any]) -> Dataset:
    """Decode a checked artifact representation without accepting unknown fields."""
    passages = tuple(Passage(item["key"], item["language"], item["text"])
                     for item in payload["corpus"])
    queries = tuple(GoldenQuery(
        item["query_id"], item["language"], item["text"],
        tuple(Evidence(e["passage_key"], e["evidence"]) for e in item["relevant"]),
        tuple(Evidence(e["passage_key"], e["evidence"]) for e in item["secondary"]),
    ) for item in payload["queries"])
    return Dataset(payload["version"], passages, queries, payload["sha256"],
                   {key: tuple(value) for key, value in payload.get("hard_negatives", {}).items()})
