# Local models for an 8 GB GPU

Decision record for the `local` profile (Ollama). Measured on 2026-11-04 on the demo machine: an RTX 3070 Laptop GPU (8192 MiB), 32 GB RAM, Windows 11, Ollama 0.40.14, NVIDIA driver 616.56. With no models loaded, Windows and the desktop already hold about **1.1 GiB of VRAM**, so roughly 7 GiB is usable.

## The choice

| Role | Model | Quantization | VRAM measured here | Fallback |
|---|---|---|---|---|
| Chat / agent | `qwen3.5:4b` (4.2B, vision + tools + thinking) | Q4_K_M (3.3 GB on disk) | 3.7 GB at `num_ctx` 16384, 4.1 GB at 32768, 100% GPU | `qwen3:4b-instruct-2507-q4_K_M` (pulled) |
| Embedder | `qwen3-embedding:0.6b` (596M), 1024 dims native | Q8_0 (639 MB on disk) | 1.2 GB, 100% GPU | `bge-m3` (567M, 1024 dims, fp16, 1.2 GB; not pulled) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` (22.7M, Apache-2.0) via sentence-transformers **on CPU** | fp32 | 0 (CPU) | `noop` |

Chat and embedder together at a 16k context: **6.5–6.7 GiB total GPU memory in use** (including the ~1.1 GiB desktop baseline). Both stay at "100% GPU" in `ollama ps`. Alternating chat and embed calls caused no reloads (chat 0.09–0.14 s for a 5-token reply, embed 0.015–0.03 s once warm).

The numbers behind each choice are below. The settings this implies for `hackstack.yaml` are at the end.

## Chat model

### Candidates and what fit

All three candidates passed the tool-calling smoke test (next section), so the deciding factor was **memory at the context length the agent needs**. Ollama defaults to a 4k context on GPUs under 24 GiB ([context-length docs](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/docs/context-length.mdx)). An agent turn holds a system prompt, tool schemas, and up to 8 reranked chunks of ~400 tokens per tool call over several iterations, so **16k is the working target and 32k is the stretch**.

Measured with the embedder co-resident (GPU total = `nvidia-smi` memory.used; per-model size and split from `ollama ps`):

| Model | Context | `ollama ps` size | Split | GPU total in use |
|---|---|---|---|---|
| `qwen3.5:4b` Q4_K_M | 16384 | 3.7 GB | 100% GPU | 6686 MiB |
| `qwen3.5:4b` Q4_K_M | 32768 | 4.1 GB | 100% GPU | 7252 MiB |
| `qwen3:4b-instruct-2507` Q4_K_M | 16384 | 5.1 GB | 100% GPU | 7509 MiB |
| `qwen3:4b-instruct-2507` Q4_K_M | 32768 | 7.9 GB | 19%/81% CPU/GPU | spills |
| `qwen3:8b` Q4_K_M | 8192 | 6.2 GB | 100% GPU | 7789 MiB (no headroom) |
| `qwen3:8b` Q4_K_M | 16384 | 7.8 GB | 20%/80% CPU/GPU | spills, evicted the embedder |

`qwen3.5:4b` grows only 0.4 GB from 16k to 32k. That fits its hybrid Gated-DeltaNet architecture ([Ollama model page](https://ollama.com/library/qwen3.5)), where most layers keep no per-token KV cache. Same-size Qwen3 models grow several GB over the same range. That is why it wins on this card. `qwen3:8b` is the stronger text model on paper, but it cannot hold 16k plus the embedder in 8 GB, so it was dropped (and removed from Ollama to save disk).

The newest Qwen generations in the library (`qwen3.6`, `qwen3.8`) start at 27B, about 17–18 GB at Q4 ([qwen3.6 tags](https://ollama.com/library/qwen3.6/tags), [qwen3.8 tags](https://ollama.com/library/qwen3.8/tags)), so they are out of reach here. `qwen3.5:9b` is 6.6 GB at Q4_K_M ([tags](https://ollama.com/library/qwen3.5/tags)), which leaves no room for the embedder or KV cache.

### Tool-calling smoke test

Script: a 2-tool agent prompt ("what did we decide about the billing retry policy, **and** who is the Postgres expert?") with `search_knowledge(query, source)` and `find_experts(topic)`, sent to Ollama's **OpenAI-compatible** `/v1/chat/completions`.

A run passes only if:

- one assistant turn contains **both** tool calls, with JSON-parseable arguments (`parallel_ok`);
- after fake tool results are fed back, the final answer uses both of them (`grounded_final`).

Each configuration ran 5 times.

| Model | Thinking | Parallel calls | Grounded final | Warm latency, tool turn / final turn |
|---|---|---|---|---|
| `qwen3.5:4b` | default (on) | 5/5 | 5/5 | 4.1–4.9 s / 5.1–6.8 s |
| `qwen3.5:4b` | `reasoning_effort: "none"` | 5/5 | 5/5 | 3.0–3.6 s / 3.1–3.5 s |
| `qwen3.5:4b` at `num_ctx` 16384 (Modelfile) | `"none"` | 5/5 | 5/5 | similar |
| `qwen3:4b-instruct-2507` | n/a (non-thinking) | 5/5 | 5/5 | 2.9 s / 3.4–3.5 s |
| `qwen3:8b` | default | 5/5 | 5/5 | 6.6–8.6 s / 8.6–9.8 s |

The first-ever request to `qwen3.5:4b` took 186 s. That was the cold load from disk, while a CPU reranker benchmark was also running. Warm-up belongs in `make dev`/`make up`.

The test also hit Ollama's **Anthropic-compatible `/v1/messages`** route, which exists in the current server ([routes.go](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/server/routes.go)). With the same two tools it returned `stop_reason: tool_use` with two parallel `tool_use` blocks. It **rejects `search_result` content blocks** (`cannot unmarshal ... anthropic.ImageSource`), so it cannot stand in for Claude's native citations. It is still useful for exercising the Anthropic adapter's tool loop live without a key.

The smoke test is easy: one obvious 2-way split, small context. It shows the plumbing works and that parallel calls come through the OpenAI-compatible layer. It does **not** measure agent quality on long contexts; the eval tickets do that.

### Constraints the adapter must respect

From Ollama's [OpenAI-compatibility docs](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/docs/api/openai-compatibility.mdx):

- **`tool_choice` is not supported.** The loop cannot force a tool call; it has to rely on the prompt.
- **The OpenAI API cannot set the context size.** Use a Modelfile (`FROM qwen3.5:4b` + `PARAMETER num_ctx 16384`, then `ollama create`) or start the server with `OLLAMA_CONTEXT_LENGTH=16384`. Verified: the Modelfile variant loaded at 16384 and passed 5/5.
- `reasoning_effort` (or `reasoning.effort`) is supported, and `"none"` turns thinking off. Thinking off was ~30% faster with no loss on this test, so use it as the agent-loop default.
- `stream_options.include_usage` is supported (needed for token accounting over SSE).

Ollama's own docs show parallel tool calling with `qwen3` ([tool-calling docs](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/docs/capabilities/tool-calling.mdx)).

### Fallback

Use `qwen3:4b-instruct-2507-q4_K_M` (pulled). It is text-only, non-thinking and slightly faster, and it also scored 5/5. It costs 1.4 GB more VRAM at 16k (5.1 vs 3.7 GB) and spills at 32k. Its model card reports BFCL-v3 61.9 versus 57.6 for the original Qwen3-4B in non-thinking mode ([model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)). Switch if `qwen3.5:4b` misbehaves, for example thinking leaking into answers or vision weights wasting memory.

## Embedder

`qwen3-embedding:0.6b`, as the research doc proposed. Verified here:

- `/api/embed` returns **1024-dimensional, unit-normalized** vectors (norm 1.0000).
- `/v1/embeddings` honours `dimensions` (asked 512, got 512): Matryoshka truncation works through the OpenAI-compatible layer too.
- Model metadata: Q8_0, 595.78M params, `embedding_length` 1024, 32k native context ([Ollama tags](https://ollama.com/library/qwen3-embedding/tags)). Ollama loads it at a 4096 context, which is plenty for 400-token chunks.
- Throughput: a batch of 32 ~254-token chunks (8,128 tokens) embedded in 2.9 s on the GPU, about **2,800 tokens/s**.
- It is **instruction-aware**. Queries should be prefixed `Instruct: <task>\nQuery: <q>`, and documents get no prefix. The model card reports 1–5% gains from instructions, with instructions written in English ([model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)). This is the asymmetry `embed_query()` vs `embed_documents()` exists for.
- License: Apache-2.0.

**Fallback:** `bge-m3`, which is also 1024-d native, 1.2 GB fp16 ([tags](https://ollama.com/library/bge-m3/tags)) and needs no query instruction. Not pulled. It needs a re-embed because vectors from two models are not interchangeable.

## Reranker

**Ollama has no rerank endpoint.** The current server registers `/api/embed`, `/api/chat`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/messages` and others, but nothing for reranking ([routes.go @ e9fde36](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/server/routes.go)). The feature request [#3368 "Reranking models"](https://github.com/ollama/ollama/issues/3368) is still open, and PR [#14172 "Add reranking support"](https://github.com/ollama/ollama/pull/14172) is unmerged. A local reranker therefore has to run in-process through sentence-transformers' `CrossEncoder`.

CPU benchmark: 50 candidates (the kit's shortlist), ~250-word passages, one relevant passage planted mid-list, 16 threads, sentence-transformers 6.2.0, torch 2.15.0+cpu. Every model ranked the planted passage first.

| Model | Params | Candidates × max tokens | Rerank time (3 runs) |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | 568M | 50 × 512 | 50.3–52.8 s |
| `BAAI/bge-reranker-v2-m3` | 568M | 20 × 256 | 11.3–11.7 s |
| `BAAI/bge-reranker-base` | 278M | 50 × 512 | 14.8–16.7 s |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | 50 × 512 | 2.5–3.1 s |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | 30 × 512 | 1.15–1.29 s |

The research doc's pick, **`bge-reranker-v2-m3`, is not usable on CPU here**: almost a minute per tool call, paid on every agent iteration. On the GPU it would compete with the chat model for the ~1 GiB of headroom left at 16k. GPU runs were not measured, since the CUDA torch wheel is a multi-GB install. It remains the right model for an offline eval comparison, or on a bigger GPU.

**Choice: `cross-encoder/ms-marco-MiniLM-L6-v2` on CPU, shortlist 30, `top_n` 8.** That costs about 1.2 s per tool call and no VRAM. It is English-only and trained on MS MARCO web queries, a fine fit for the English Northwind corpus but not for multilingual data. It needs `sentence-transformers` plus CPU torch as an optional dependency group, so `make test` never needs it.

**Fallback: `noop`.** It is the instant cut the research doc already requires, and the right setting whenever the optional dependency group isn't installed. Whether MiniLM's rerank actually beats `noop` on recall is for the eval slice to show, not this ticket.

The same Qwen3 family has `Qwen3-Reranker-0.6B` (Apache-2.0, 596M), but it is an LLM-style yes/no reranker with the same size and CPU-cost problem as bge-v2-m3, and Ollama cannot serve it as a reranker either.

## Operating notes

- **Keep exactly one chat model plus the embedder loaded.** During testing, leftover chat models from earlier runs made Ollama evict the embedder ("predicted to exceed available memory, evicting" in `server.log`). Set `OLLAMA_KEEP_ALIVE=30m` (or `-1`) for demo day. Don't switch chat models mid-demo.
- `OLLAMA_MAX_LOADED_MODELS` defaults to 3 per GPU, and memory is the real limit ([FAQ](https://github.com/ollama/ollama/blob/e9fde36d75fea66579d84099956ce1430792737a/docs/faq.mdx)). `OLLAMA_NUM_PARALLEL` defaults to 1. Raising it multiplies KV memory by the parallel count, so leave it at 1 on this card.
- Close GPU-heavy apps (browsers with hardware acceleration, games) before the demo. The ~1.1 GiB desktop baseline varied between 1.08 and 2.4 GiB during testing.

## Settings this implies

This snippet is for whoever owns the `local` profile in `hackstack.yaml`; this ticket does not edit it. The `answer_model` placeholder there is currently `qwen3:8b`, which does not fit a 16k context.

```yaml
profiles:
  local:
    llm:
      provider: openai_compat
      base_url: http://localhost:11434/v1
      answer_model: hackstack-chat   # = qwen3.5:4b + PARAMETER num_ctx 16384 (Modelfile, created by `make setup`)
      aux_model: hackstack-chat
      reasoning_effort: none
    embed:
      provider: ollama
      model: qwen3-embedding:0.6b
      dim: 1024
      query_instruction: "Given a question, retrieve passages from company documents that answer it"
    rerank:
      provider: cross_encoder        # or noop when the optional deps aren't installed
      model: cross-encoder/ms-marco-MiniLM-L6-v2
      window: 30
      top_n: 8
```

Pulled into Ollama: `qwen3.5:4b`, `qwen3:4b-instruct-2507-q4_K_M`, `qwen3-embedding:0.6b`. The MiniLM and bge reranker weights are in the HF cache from the benchmark.
