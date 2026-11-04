# Contextual enrichment on the local model: what it costs

Measured 2026-11-04 on the desktop (RTX 3070 Laptop 8 GB, 32 GB RAM). Model: `hackstack-chat`
(qwen3.5:4b Q4_K_M, 16k context, thinking off) through Ollama's OpenAI-compatible API.
Embedder `qwen3-embedding:0.6b`. Settings: `enrich.max_doc_tokens: 8000`, `concurrency: 1`.
Ingested into a throwaway Postgres database, not the shared dev index.

| Run | Docs | Chunks | Enrichment time | Per chunk | Input tokens | Output tokens |
|---|---|---|---|---|---|---|
| Northwind-authored docs only (`northwind/`) | 13 | 30 | 43.7 s | 1.46 s | 19,061 | 2,266 |
| Full sample corpus (Northwind + PostHog handbook) | 30 | 265 | 591.4 s | 2.23 s | 1,068,478 | 16,927 |

The whole ingest took 654 s, against about 13 s without enrichment. GPU memory was 7.1 of 8 GiB with
the chat model and embedder loaded together. All 265 chunks were enriched, with none failed or
over budget, and the blurbs averaged about 300 characters.

**How to read it.**
- **Input tokens:** each call resends the whole document, so input grows as chunks × document
  length. The long handbook pages average about 4,000 input tokens a call. On Claude the document
  block is marked `cache_control`, so every call after the first should read it from the cache.
  The enricher logs a warning if `cache_read_input_tokens` stays at 0 for a document above
  512 tokens. On Ollama the identical prefix lets the server reuse its KV cache. Ollama reports no
  cache counters, so that reuse can't be measured from the API.
- **Re-runs:** about 10 minutes for 265 chunks is a one-time cost. The sync manager hashes the
  enrichment settings into each document's fingerprint, so later runs re-enrich only the
  documents that changed.
- **Quality:** the blurbs name the document, the section, the owner and the policies involved.
  For example, a compensation-bands chunk got "Document: Compensation bands 2026; Section: Base
  salary table; People/Owner: Maria Santos (People Ops) …". Whether they improve recall is for the
  eval tickets to show. Anthropic's 35–49% failure reduction dates from 2024 and hasn't been
  replicated here.

**Choice.**
- **Default:** `enrich.contextual` stays **off** in every profile. With it off, ingest of the
  sample corpus is about 50× faster. It costs nothing to leave off: an index built without it
  stays valid.
- **Turning it on:** set it for the final demo index, as the research doc's knob table suggests,
  and measure it with the eval suite before keeping it on.

**Not measured.**
- The Anthropic path (Claude Haiku 5.5 with prompt caching) wasn't run live, because there's no
  key. Its request shape is covered offline only.
- Recall with enrichment against recall without it wasn't measured.
