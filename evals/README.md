# Evals

```bash
HACKSTACK_PROFILE=local make eval     # compare retrieval configs; exits 1 below baseline.json
HACKSTACK_PROFILE=local make golden   # regenerate golden.jsonl with the aux LLM, then review it
uv run python -m evals.run_retrieval --offline   # fake embedder + in-memory store, no services
```

| File | What it is |
|---|---|
| `metrics.py` | recall@k, precision@k, MRR, hit@k, nDCG@k by hand (binary relevance) |
| `make_golden.py` | one question per sampled chunk, lookalikes, a realism filter |
| `handwritten.yaml` | realistic questions on the planted moments, unanswerable and ACL-forbidden ones |
| `golden.jsonl` | the committed golden set (labels are chunk ids, chunker version included) |
| `run_retrieval.py` | dense / lexical / hybrid / hybrid+rerank on the same index; runs saved to `runs/` |
| `baseline.json` | recall@5 and MRR per profile; `tests/test_eval_regression.py` and `make eval` gate on them |

## Results: local profile, 4 November 2026

`qwen3-embedding:0.6b` + Postgres + the CPU MiniLM-L6 cross-encoder, over the Northwind corpus
(30 docs, 265 chunks). 51 questions: 46 answerable (24 synthetic, 11 lookalike, 11 hand-written),
3 not in the corpus, 2 ACL-forbidden.

| variant | recall@5 | recall@10 | MRR | hit@1 | recall@5 hand-written | p50 latency |
|---|---|---|---|---|---|---|
| dense | 0.928 | 0.957 | 0.845 | 0.761 | 0.879 | ~100 ms |
| lexical | 0.699 | 0.808 | 0.548 | 0.391 | 0.561 | ~7 ms |
| hybrid | 0.928 | 0.949 | 0.809 | 0.696 | 0.879 | ~100 ms |
| **hybrid + rerank** | **0.906** | 0.957 | **0.909** | **0.870** | 0.879 | ~1.6 s |

- **The headline:** hybrid + rerank puts a right chunk in the top 5 for 91% of answerable
  questions, and in first place for 87%. It is 88% on the 11 hand-written questions.
- **The reranker pays off at the top of the list, not lower down.** It raises MRR from 0.81 to
  0.91 and hit@1 from 0.70 to 0.87, which is what an 8-passage answer context needs. Recall@5
  dips by one question and recall@10 is the same. Its cost is about 1.6 s a query on CPU.
- **Not-found threshold:** the cross-encoder separates the 5 no-answer questions cleanly. Their
  best top score is below 97.8% of the answerable ones, and refusing below a score of about
  -3.1 is right on 98% of all 51 questions. That's only 5 negatives, so treat it as a hint, not
  a calibrated threshold.
- **No leaks:** the 2 ACL-forbidden questions never returned a restricted document, in any
  variant.

**Caveats.**
- Synthetic questions reuse their chunk's wording, which flatters full-text search. Lexical
  scores 0.667 on synthetic questions and 0.561 on hand-written ones. Compare configs with these
  numbers rather than quoting them as absolute quality.
- The labels are one chunk per synthetic question. A neighbouring chunk that also answers it
  counts as a miss, which is why `doc hit@5` (0.98) sits above recall@5.
- The questions and the realism ratings both come from the same 4B model. The lookalike labels
  were spot-checked against the corpus text, but not all 51 were.
- n = 46. One question is worth about 0.02 of recall.
