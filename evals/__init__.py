"""Retrieval evals: a synthetic golden set, hand-written metrics and a regression baseline.

- ``make_golden.py``  generate ``golden.jsonl`` from the active app's corpus with the aux LLM
- ``metrics.py``      recall@k, precision@k, MRR, hit rate, nDCG@k (binary relevance)
- ``run_retrieval.py`` compare dense / lexical / hybrid / hybrid+rerank; ``make eval``
- ``baseline.json``   committed numbers; ``tests/test_eval_regression.py`` fails below them
"""
