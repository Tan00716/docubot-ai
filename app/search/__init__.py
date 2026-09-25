"""Vector search: find the stored chunks whose meaning is closest to a question.

Modules:
    models.py     - limits (top_k, query length), result shapes, errors, validation
    similarity.py - cosine similarity of unit vectors with NumPy (checks every vector)
    storage.py    - read-only SQLite query for chunks with a VALID stored vector
    service.py    - the pipeline: validate -> candidates -> embed query -> score -> rank

The search is exact brute force (every candidate is compared), meant for
local development and small collections. There is NO reranking, no hybrid
or keyword search, no vector database and no RAG answer generation yet.
"""
