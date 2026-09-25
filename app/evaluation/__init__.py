"""Retrieval evaluation: a small development test set for the CURRENT search.

Modules:
    corpus.py  - hand-written passages (English, Chinese, mixed), one document each
    golden.py  - evaluation questions + evidence-based relevance labels
    metrics.py - Recall@K and MRR (pure functions)
    runner.py  - runs the questions through the real pipeline
                 (process -> chunk -> embed -> app.search.service.search)
    report.py  - command line report: python -m app.evaluation.report

This is a development evaluation set, not a general benchmark: about forty
passages and thirty questions written for DocuBot's own domain. It measures
whether the right evidence is found and how high it is ranked. It does not
generate or judge answers.
"""
