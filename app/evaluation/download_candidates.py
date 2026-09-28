"""Download ONE allowlisted comparison model into the local cache (explicit, online).

    .venv\\Scripts\\python.exe -m app.evaluation.download_candidates e5-base

This is the only step of the model comparison that uses the network. The
comparison itself (compare_models.py) runs with HF_HUB_OFFLINE=1 and never
downloads. The download reuses the application's own secure mechanism
(app.embeddings.provider.download_model_files):

    - only names on the candidate allowlist (candidates.py), never a URL
    - the exact pinned revision (a commit hash), never "main"
    - only the listed tokenizer / config / ONNX files: no Python code, no
      pickle or PyTorch files, so no code from the model repository runs
    - into storage/model_cache/ (git-ignored), one folder per repo + revision

Afterwards the downloaded files are checked against the spec.
"""

import os
import sys
from pathlib import Path

from app.embeddings.provider import download_model_files, model_files
from app.evaluation.candidates import UnknownCandidateError, get_candidate, verify_official_files
from app.evaluation.resources import files_size

MODEL_CACHE_DIR = Path(__file__).resolve().parents[2] / "storage" / "model_cache"
GIGABYTE = 1024**3


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m app.evaluation.download_candidates <candidate>", file=sys.stderr)
        return 2
    try:
        candidate = get_candidate(argv[0])
    except UnknownCandidateError as error:
        print(error, file=sys.stderr)
        return 2
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        print("HF_HUB_OFFLINE=1 is set; this command needs the network.", file=sys.stderr)
        return 1
    spec = candidate.spec
    print(f"Downloading {spec.name}@{spec.revision} "
          f"({candidate.download_bytes / GIGABYTE:.2f} GB) into {MODEL_CACHE_DIR} ...")
    folder = download_model_files(spec, MODEL_CACHE_DIR)
    if folder.name != spec.revision:
        print(f"Unexpected snapshot folder {folder.name}.", file=sys.stderr)
        return 1
    problems = verify_official_files(candidate, folder)
    if problems:
        print("The downloaded files do not match the spec: " + "; ".join(problems),
              file=sys.stderr)
        return 1
    size = files_size(folder, model_files(spec))
    print(f"OK: {len(model_files(spec))} files, {size / GIGABYTE:.2f} GB, spec verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
