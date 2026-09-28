"""Build an assessor pack without model identity, scores, rankings, or gold labels."""

import hashlib
import json
from pathlib import Path

from app.evaluation.dataset_versions import Dataset, verify_dataset

RUBRIC = (
    "RELEVANT: directly supports the answer to the query. "
    "PARTIALLY_RELEVANT: related context but insufficient to answer. "
    "NOT_RELEVANT: does not support the answer. Judge only the query and passage text."
)


class ReviewError(ValueError):
    """A blind review is incomplete or cannot support a stable golden label."""


def finalize_blind_reviews(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep reviewed, answerable questions; remove explicitly ambiguous queries.

    Only RELEVANT judgments become positive for binary Recall@K/MRR. Partial
    relevance is retained in the review record but does not count as positive.
    """
    accepted: list[dict] = []
    removed: list[str] = []
    allowed = {"RELEVANT", "PARTIALLY_RELEVANT", "NOT_RELEVANT"}
    for record in records:
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ReviewError("Every reviewed query needs an ID.")
        if record.get("ambiguous") is True:
            removed.append(query_id)
            continue
        candidate_ids = [candidate["candidate_id"] for candidate in record.get("candidates", [])]
        candidates = set(candidate_ids)
        judgments = record.get("judgments", [])
        judgment_ids = [item.get("candidate_id") for item in judgments]
        if (len(candidates) != len(candidate_ids) or len(judgments) != len(candidates)
                or set(judgment_ids) != candidates or len(set(judgment_ids)) != len(judgment_ids)):
            raise ReviewError(f"{query_id}: every candidate must be judged exactly once.")
        labels = [item.get("label") for item in judgments]
        if any(label not in allowed for label in labels):
            raise ReviewError(f"{query_id}: unknown relevance judgment.")
        if "RELEVANT" not in labels:
            raise ReviewError(f"{query_id}: no answerable relevant evidence remains.")
        accepted.append(record)
    return accepted, removed


def blinded_review_pack(dataset: Dataset) -> dict:
    corpus = {passage.key: passage for passage in dataset.passages}
    records = []
    for query in dataset.queries[33:]:
        candidate_keys = {e.passage_key for e in query.relevant}
        candidate_keys.update(dataset.hard_negatives.get(query.query_id, ()))
        # Add one more same-language distractor when available to discourage a
        # two-choice answer-key guessing strategy.
        target = query.relevant[0].passage_key
        target_language = corpus[target].language
        extra = next((p.key for p in dataset.passages
                      if p.key not in candidate_keys and p.language == target_language), None)
        if extra:
            candidate_keys.add(extra)
        ordered = sorted(candidate_keys,
                         key=lambda key: hashlib.sha256(f"{query.query_id}:{key}".encode()).hexdigest())
        candidates = []
        for index, key in enumerate(ordered, start=1):
            passage = corpus[key]
            candidates.append({"candidate_id": f"{query.query_id}-c{index:02d}",
                               "language": passage.language, "text": passage.text})
        review_id = hashlib.sha256(f"review:{query.query_id}".encode()).hexdigest()[:12]
        records.append({"query_id": f"review-{review_id}", "query": query.text,
                        "query_language": query.language, "ambiguous": False,
                        "candidates": candidates, "judgments": []})
    return {"dataset_version": dataset.version, "review_mode": "blind", "rubric": RUBRIC,
            "records": records}


def write_blinded_review_pack(path: Path) -> Path:
    dataset = verify_dataset("retrieval_eval_v2")
    pack = blinded_review_pack(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "docs" / "retrieval_eval_v2_blind_review.json"
    print(write_blinded_review_pack(target))
