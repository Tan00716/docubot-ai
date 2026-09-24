"""Embeddings: turn chunk text into vectors with a local model and store them.

Modules:
    config.py   - the one place that names the embedding model and its settings
    models.py   - data shapes (EmbeddingContract, records, summaries, errors)
    vectors.py  - text hashing, normalization and float32 (de)serialization
    provider.py - the local FastEmbed model (the only module that imports it)
    storage.py  - SQLite reads/writes for the "embeddings" table
    service.py  - the pipeline: chunks -> provider -> SQLite

There is NO vector search, retrieval or RAG yet: vectors are only stored.
"""
