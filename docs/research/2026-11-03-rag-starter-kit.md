# Hand-build the brain, borrow the bricks

Jin should a small Python RAG core and buy only the commodity parts. The core is a FastAPI service that runs about 60 lines of Claude tool-use loop over one Postgres + pgvector database, doing hybrid BM25 + vector search fused with RRF, a reranker, and native `search_result` citations. A thin Next.js chat UI talks to it over a server-sent-events (SSE) contract he owns. Parsing (Docling/MarkItDown), embeddings (Voyage 4), reranking (Voyage rerank-3) and source SDKs are imported libraries, not frameworks. The research supports this for three reasons. First, every component of a company brain (connectors, ACL-aware hybrid retrieval, reranking, cited generation) is simple enough to write and explain in a day. Second, the frameworks that would hide it are churning or partly dead: LlamaIndex.TS was archived in April 2026, Verba and Cognita are abandoned, and Vercel's AI SDK has reportedly moved through v5, v6 and v7 in about 18 months. Third, Anthropic's own guidance is to "start by using LLM APIs directly." The plan for tomorrow works because Python owns the parsing and retrieval ecosystem, Postgres holds vectors, full text, ACLs, chat history and eval results in one database he can read with SQL, and Claude's `search_result` blocks give validated citations without any prompt engineering. A plain "chat with docs" bot is what everyone else will bring. To win, the kit needs visible, agentic, enterprise-shaped behaviour: answers that change when you switch users, "who should I ask about X", "what changed this week", conflict flags with dated citations, and a live sources panel. Its own eval number backs this up. The single most important logistics point: many hackathons score only work done during the event, so whatever he pre-builds tonight must be declared, or limited to credentials, data and design if the rules forbid pre-written code ([TinyFish judging](https://sg-hackathon.tinyfish.ai/judging)).

## Python core, Postgres memory, Claude voice: the stack layer by layer

The language choice is clear. The parsing and retrieval ecosystem is consolidating in Python: LlamaIndex deprecated its TypeScript port in March 2026 and archived it on April 30, 2026, leaving its orchestration Python-only ([DevelopersDigest](https://www.developersdigest.tech/blog/llamaindex-workflows-agents-whats-new-2026)). Docling, Unstructured, sentence-transformers rerankers and every vector-DB client are Python-first. Python is also Jin's strongest language. The frontend is where TypeScript earns its place. A Next.js chat with a citations panel reads as a product rather than a notebook demo, and an SSE contract that Jin owns keeps the two halves swappable. A TS-only stack (Mastra, Vercel AI SDK) is quick for UI but thin for ingestion. Mastra is TS-only with "no Python port planned" ([DEV Community](https://dev.to/jangwook_kim_e31e7291ad98/mastra-ai-10-the-typescript-agent-framework-developers-are-actually-shipping-2lan), single secondary source). A Python-only UI is the fallback, but Chainlit's original team "stepped back" in May 2025 and it is now community-maintained ([PyPI](https://pypi.org/project/chainlit/)).

| Layer | Pick for tomorrow | Runner-up | Why the pick wins |
|---|---|---|---|
| Backend language / API | Python 3.11+ with FastAPI, SSE via `StreamingResponse` | TypeScript (Mastra / AI SDK) | Parsing, rerankers and vector clients are Python-first; it is Jin's strongest language |
| Frontend | Next.js + shadcn/ui, hand-rolled `fetch` + `ReadableStream` SSE reader (~40 lines) | Streamlit/Gradio client on the same `/chat` endpoint | Looks like a product; owning the event schema avoids AI SDK version churn |
| LLM (default) | Claude via the official `anthropic` SDK. `claude-sonnet-5-5` ($2/$10 per MTok) for answers, `claude-haiku-5-5` (from $0.10/$0.50) for rewrite, contextualization and judges, `claude-opus-5-5` ($4/$20, Anthropic's recommended default) when quality matters more than latency ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)) | OpenAI-compatible adapter (covers OpenAI, Ollama, vLLM, Groq, OpenRouter) | Native `search_result` citations, prompt caching and `pause_turn` need a native adapter |
| Provider abstraction | Own `LLMProvider` protocol with two adapters (~150 lines) | Pydantic AI | Least magic. Note that LiteLLM PyPI 1.82.7/1.82.8 were compromised on March 24, 2026 ([Arthur.ai](https://www.arthur.ai/column/litellm-supply-chain-attack-pypi-compromise-2026)) |
| Embeddings | `voyage-4` at 1024 dims ($0.06/M, 200M free tokens) ([Voyage pricing](https://docs.voyageai.com/docs/pricing)) | Local `qwen3-embedding:0.6b` via Ollama (1024-d, 639 MB, Apache 2.0) ([Morph](https://www.morphllm.com/ollama-embedding-models)) | Free at hackathon scale. Matryoshka sizes 2048/1024/512/256 and a shared space across the voyage-4 family ([Voyage blog](https://hn.jakechampion.name/reader/https://blog.voyageai.com/2026/01/15/voyage-4)) |
| Vector + lexical store | Postgres with pgvector 0.8.x (HNSW, iterative scans) + built-in `tsvector` | Qdrant (native sparse+dense with server-side RRF, filter-aware HNSW, Apache 2.0) | One database for vectors, full text, ACLs, history and evals. pgvector alone is recommended below ~5M vectors ([eCorpIT](https://ecorpit.com/postgres-vector-search-pgvector-pgvectorscale-paradedb-lantern-2026/)) |
| Zero-Docker fallback | LanceDB embedded (hybrid + Tantivy FTS) | Chroma (single-writer, RAM-resident HNSW) | `pip install` only ([Actian](https://www.actian.com/blog/developer/comparing-embedded-vector-databases/)) |
| Fusion | RRF, k=60, top-50 dense + top-50 lexical | Weighted RRF / tuned score normalization | Needs no score calibration. Tuned normalization beat it by only ~3.9% NDCG@10 in one study ([BigData Boutique](https://bigdataboutique.com/blog/reciprocal-rank-fusion-how-it-works-and-when-to-use-it)) |
| Reranker | Voyage `rerank-3` ($0.05/M, 200M free) or `rerank-3-lite` ($0.02/M) | Local `bge-reranker-v2-m3` cross-encoder; Cohere Rerank 4 | Same API key as embeddings; one HTTP call |
| Parsing | MarkItDown for Office/HTML, Docling for PDFs with tables | `pymupdf4llm`/PyMuPDF (fastest, but AGPL) | Both MIT; Docling ranked first on PDFs in an independent 12-tool survey ([yage.ai](https://yage.ai/share/markitdown-survey-en-20260412.html)) |
| Connectors | Thin wrappers on official SDKs: `notion-client`, `google-api-python-client`, `slack_sdk`, PyGithub/`httpx` | LlamaHub / LangChain loaders | Full metadata and ACL control; avoids loaders broken by the Notion 2025-09-03 API change |
| Live tools | MCP client adapter and `search_kb` MCP server via `mcp>=1.27,<2` over stdio | Python SDK v2 (`MCPServer`), beta as of June 30, 2026 ([release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0b1)) | Lowest-risk path to a Claude Desktop demo |
| Tracing | Arize Phoenix, one container, OTel/OpenInference | Langfuse Cloud Hobby (free, 50k units/month) ([Langfuse pricing](https://langfuse.com/pricing)) | Self-hosted Langfuse needs 6 containers and ~16 GiB RAM ([testmuai](https://www.testmuai.com/blog/langfuse-vs-phoenix/)) |
| Evals | Own retrieval metrics (~30 lines) and binary Claude judges | DeepEval 4.x for pytest-style CI gates | Fully explainable. RAGAS appears stalled with an unpatched CVE ([ecorpit](https://ecorpit.com/rag-evaluation-tools-ragas-deepeval-braintrust-giskard-2026/), vendor-affiliated) |
| Config / infra | pydantic-settings + `brain.yaml` + Docker Compose (api, db, optional phoenix) | Plain env vars | Typed config and a hashable `config_hash` per trace |

Two choices deserve a steelman. **Qdrant** is the better pure retrieval engine: dense and sparse prefetches fused server-side, BM25 as a sparse encoder, and `is_tenant` for per-tenant filtering ([Qdrant docs](https://qdrant.tech/documentation/concepts/indexing/)). Postgres wins tomorrow because a company brain is mostly *relational* (documents, owners, timestamps, ACLs, sync cursors, conversations), and Jason Liu's advice is to keep full-text and vector search in one database to avoid index drift ([jxnl.co](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/)). The second steelman is to **skip RAG entirely**. Anthropic says that under ~200K tokens you should just put the whole knowledge base in the prompt with caching ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)), and current Claude models have 1M-token windows ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)). For a toy corpus that is the honest baseline. RAG still earns its keep here because of per-user ACL filtering, multi-source scale, freshness, and citations to specific chunks. Saying exactly this in Q&A shows judgment.

One connection to Jin's existing work matters (from his memory notes). His Augur project already specifies FastAPI + SSE, Next.js + TypeScript, pgvector, Anthropic, hand-built RAG through Phase 2 and LangGraph in Phase 3. This kit is effectively Augur's Phase 2 generalized into a library. The Augur streaming and UI code is reusable, but it counts as pre-existing work to declare. Because Augur already covers LangGraph, the kit's "framework mode" comparison should use Pydantic AI so the portfolio shows two different frameworks.

## Hand-build the loop, borrow the bricks

The verdict is a **framework-optional core**. Hand-write the agent loop, provider adapters, retrieval and fusion, which come to about 300 lines of code Jin will own completely. Import libraries only for components that are not the agent itself: parsers, the database driver, rerankers and SDKs. Then add a short `docs/frameworks.md` with a ~30-line Pydantic AI version of the same agent for comparison. The evidence against heavy frameworks for this goal is consistent. Anthropic warns that frameworks "often create extra layers of abstraction that can obscure the underlying prompts and responses," and that "incorrect assumptions about what's under the hood are a common source of customer error" ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)). Hamel Husain intercepted framework traffic and found LangChain's SmartLLMChain making 4 calls and a DSPy "minimal" example running for 30+ minutes with hundreds of calls ([hamel.dev](https://hamel.dev/blog/posts/prompt/)). Octomind removed LangChain once debugging it took as much time as building features ([Octomind](https://octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents)).

The frameworks themselves are not bad. They are wrong for *this* goal, which is deep understanding plus a reusable portfolio piece. **LangGraph 1.0** (October 22, 2025) has the most job-market recognition, durable checkpointers and `interrupt()` for human-in-the-loop, plus a no-breaking-changes promise until 2.0 ([LangChain](https://www.langchain.com/blog/langchain-langgraph-1dot0)). It is overkill for a single-agent retrieval loop. **Pydantic AI** has the lowest magic among Python frameworks: typed outputs, dozens of providers, and a `VercelAIAdapter` that serves Next.js `useChat` straight from FastAPI ([Pydantic AI](https://pydantic.dev/docs/ai/integrations/ui/vercel-ai/)). Nobody has verified that it passes Anthropic `search_result` blocks and `search_result_location` citations through end to end, and those are the kit's best feature. **LlamaIndex** is strongest at ingestion (LlamaParse, LiteParse) but has many small packages to pin ([DevelopersDigest](https://www.developersdigest.tech/blog/llamaindex-workflows-agents-whats-new-2026)). **The Claude Agent SDK** "runs the Claude Code binary" with file and shell tools, and Anthropic points API builders to the Client SDK instead, where "you write the tool loop yourself, or let the client SDK's beta tool runner drive it" ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)). That makes it a poor fit for a pluggable web RAG backend. Haystack, DSPy and current Pydantic AI versions were not researched in depth.

The SDK's beta `tool_runner` is a legitimate middle ground. It derives schemas from `@beta_tool` type hints, iterates until no tool use or `max_iterations`, and returns exceptions as `is_error` ([tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)). Anthropic's own docs recommend the manual loop when you need custom logging, approvals or conditional execution, which is exactly what a sources panel and approve-before-acting buttons require. Write the manual loop.

## One question, end to end: the architecture

The system has two paths that share one schema. On the **ingest path**, each connector yields a `RawDoc`: content bytes or text, MIME type, URL, title, authors, created and updated timestamps, parent container, and an ACL mapped to principals such as `user:alice@acme.com`, `group:eng` or `public`. A MIME router sends the content to a parser that returns Markdown with a section tree. A chunker chosen by `source_type` splits it: headings for docs, threads for Slack, messages for email, functions for code. An optional enricher asks Claude Haiku for 50–100 tokens of situating context per chunk. The embedder vectorizes `embed_text`, which is the breadcrumb plus context plus chunk text. The store upserts the chunk under a deterministic `chunk_id = f"{doc_id}#{ordinal}"`, with a generated `tsvector` column from the same text and the ACL principals copied onto every chunk. A `SyncManager` owns the content hashes, per-connector cursors and deletion reconciliation. Connectors never touch the database, so they can be tested against fixture JSON.

On the **query path**, the browser POSTs `{conversation_id, message, as_user}` to `/chat`. The server rebuilds the message history from Postgres rather than trusting the client, resolves the user to a principal set, and calls Claude with a small toolset: `search_kb`, `get_document`, `find_experts`, `whats_changed`, plus optional MCP tools. Claude writes its own search queries from the full history, which acts as implicit multi-turn rewriting. It can call `search_kb` several times in parallel, with filters it fills in itself (`source`, `since`). Each call runs dense top-50 and lexical top-50 inside one ACL-filtered SQL query, fuses them with RRF, reranks to the top 8–10, expands to neighbouring chunks, and returns `search_result` blocks with citations enabled. The loop streams `text_delta` and citation events to the browser as SSE, alongside `tool_start` and `tool_end` events that drive a live "Searching: 'PTO policy'…" panel. The final assistant content, including tool blocks, is saved as JSON so citation indices stay aligned across turns. Every stage emits an OpenTelemetry span to Phoenix. A `mode=single_shot` flag skips the agent (retrieve, then one call) to provide the latency path and the eval baseline.

This is the same pipeline the commercial products converge on: connector, normalized document with owners, timestamps, ACL and hierarchy, structure-aware chunks, hybrid index, rerank, cited LLM answer, optional actions ([Onyx](https://github.com/onyx-dot-app/onyx); [Microsoft semantic index](https://learn.microsoft.com/microsoftsearch/semantic-index-for-copilot); [Glean](https://docs.glean.com/security/knowledge-graph)). The industry is split between **index-first** designs (Glean, Onyx, Copilot) and **federated live search over MCP** (Claude connectors, parts of ChatGPT). Indexing the slow-changing sources and adding MCP for live state lets Jin tell judges a real architectural story. That hybrid is also what the RAG-vs-MCP analysis recommends: retrieval takes tens of milliseconds while MCP calls take 1–3 seconds, and MCP is always fresh and enforces permissions as the user ([StackOne](https://www.stackone.com/blog/rag-vs-mcp/)).

## Every component, opened up

### Connectors: content, metadata and permissions in one pass

A connector's only job is to turn a source's data model into normalized `RawDoc`s, permissions included. The design to copy is Onyx's capability split: `LoadConnector.load_from_state()` for full backfill, `PollConnector.poll_source(start, end)` for time-window increments, `SlimConnector.retrieve_all_slim_docs()` for ID-only listings used to detect deletions, plus permission-sync and checkpointed variants ([Onyx interfaces.py](https://github.com/onyx-dot-app/onyx/blob/main/backend/onyx/connectors/interfaces.py)). For tomorrow, implement `load_all()` everywhere and `poll(since)` where it is cheap. Use a `@register("slack")` decorator. Put connector settings in `brain.yaml` and secrets in `.env`.

Credential setup is where the time goes, so ranked by setup cost: local files (0 min), GitHub fine-grained PAT (~2 min), Notion internal integration (~5 min; pages must be shared with it explicitly), Slack internal app bot token (~10 min; the bot must be invited to each channel), Google "Desktop app" OAuth client in Testing mode covering both Drive and Gmail (~20–30 min). The researcher estimated these times from background knowledge rather than fetched docs.

The internals that matter per source are as follows. **Notion** pages are block trees fetched recursively. The `2025-09-03` API version is breaking: databases now contain "data sources", queries moved to `PATCH /v1/data_sources/:id/query`, and `@notionhq/client` v5 removed the old endpoints. Rate limits are ~3 rps per connection, with 429/529 plus `Retry-After` ([Notion upgrade guide](https://developers.notion.com/docs/upgrade-guide-2025-09-03); [limits](https://developers.notion.com/reference/request-limits)). **Drive** incremental sync uses `changes.getStartPageToken`, then `changes.list` with `includeRemoved`, then you store `newStartPageToken` ([Drive](https://developers.google.com/workspace/drive/api/guides/manage-changes)). **Gmail** does a full sync with `messages.list`, stores `historyId`, then syncs partially with `history.list`. An expired history returns 404 and forces a full re-sync ([Gmail](https://developers.google.com/workspace/gmail/api/guides/sync)). **Slack**'s `conversations.history` and `conversations.replies` are throttled to 1 request/min with 15 objects for commercially distributed non-Marketplace apps since May 29, 2025. Internal customer-built apps keep 50+ requests/min with 1,000 objects ([Slack changelog](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps)), so the kit's Slack connector must always be installed as an *internal* app. **GitHub** allows 5,000 requests/hr with a PAT, with secondary limits of 100 concurrent requests and 900 points/min ([GitHub](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)). Use `git clone --depth 1` for repo contents, which costs no API calls, and use REST with `since=` for issues and PRs.

The main knobs are scopes (read-only everywhere), channel, repo and folder allowlists, a `max_items` cap for demo backfills, and backoff settings in a shared `RateLimitedClient` that honours `Retry-After`. Failure modes:

- **Silent truncation from pagination bugs.** Notion paginates children at 100.
- **Notion loaders on the old API version.**
- **Google refresh tokens expiring after 7 days** in Testing mode with external users ([Google OAuth](https://developers.google.com/identity/protocols/oauth2)).
- **A Slack bot not invited to a channel.** It returns nothing, without an error.
- **Connectors that crash a whole run.** Yield `ConnectorFailure` records instead, following Onyx's `Resolver`.

### Parsing: two tiers, routed by MIME type

The parser turns bytes into Markdown with a section tree. **Docling** runs a layout-detection model, then a table-structure model, then optional OCR, then reading-order assembly into a typed `DoclingDocument` tree with provenance. Only the VLM, OCR and export facts were confirmed by the research. The rest of this description of the internals is the researcher's background knowledge. Docling is MIT-licensed, needs Python ≥3.10, accepts PDF/DOCX/PPTX/XLSX/HTML/email/audio and more, and exports Markdown or lossless JSON ([Docling](https://github.com/docling-project/docling)). It ranked first on PDFs in an independent April 2026 survey (0.882 overall, table fidelity 0.887), but needs ~1 GB of install with PyTorch and runs at ~3.1 s/page on CPU ([yage.ai](https://yage.ai/share/markitdown-survey-en-20260412.html)). **MarkItDown** is a per-format converter registry wrapping existing libraries, so it is light (~251 MB, no GPU). It does no layout analysis, which explains its weak PDF scores (tables 0.273, heading hierarchy 0.000) ([yage.ai](https://yage.ai/share/markitdown-survey-en-20260412.html); [MarkItDown](https://github.com/microsoft/markitdown)). The router sends Markdown and HTML through as-is, Office files and HTML to MarkItDown, and PDFs to Docling, with results cached per content hash.

The knobs are the per-MIME routing table, Docling's OCR on/off, and the VLM pipeline (`--pipeline vlm --vlm-model granite_docling`). Failure modes:

- **Docling's first-run model download** stalls a fresh environment. Warm it up tonight.
- **CPU slowness on large PDF sets.** Pre-parse them.
- **Hidden text in documents.** White-on-white text, zero-width characters and HTML comments carry prompt injections, so strip them at extraction ([OWASP LLM08](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/)).
- **License traps.** PyMuPDF/pymupdf4llm is AGPL-3.0 ([pymupdf4llm](https://github.com/pymupdf/pymupdf4llm)). Marker's weights are restricted for companies above $5M in funding or revenue, and sources disagree on its code license: the README says Apache 2.0, older third-party posts say GPL-3.0 ([Marker](https://github.com/datalab-to/marker)).

### Chunking: structure first, small chunks, the thread as the unit

The chunker decides the retrieval unit. The default is to split on Markdown headings, then recursively down to ~400 tokens (max 512) with 0–50 overlap, and prepend `"{title} > {section_path}"` to `embed_text`. Chroma's evaluation found a 200-token recursive splitter with zero overlap a strong baseline (precision 7.0 against 1.4 for the popular 800/400 setting), and that reducing overlap improved IoU ([Chroma Research](https://www.trychroma.com/research/evaluating-chunking)). That study used five corpora and top-5 retrieval, so treat 200–400 as a starting range to tune with evals rather than a law.

Per-source rules are reasoned practice, not benchmarked:

- **Slack:** a thread is one chunk, written as `author: text` lines with timestamps. Long threads get a sliding window with 1–2 messages of overlap.
- **Email:** one message per chunk, with quoted replies and signatures stripped and a Subject/From/To/Date header prepended. Attachments become child documents.
- **Code:** split on functions and classes, with the file path and symbol name prepended.
- **Tables:** keep each table whole, repeating headers on splits.

Semantic chunking rarely justifies its cost. In Chroma's run, Kamradt semantic chunking reached 83.6 recall against 89.5 for plain recursive at 400 tokens ([Chroma](https://www.trychroma.com/research/evaluating-chunking)). Late chunking gains ~1.5–1.9 nDCG points but needs a long-context mean-pooling embedder ([arXiv 2409.04701](https://arxiv.org/html/2409.04701v3)).

The knobs are `chunk.size`, `chunk.overlap`, the separator list, a minimum chunk size to avoid fragments, and a `chunker_version` that goes into chunk IDs. Failure modes:

- **Tiny heading-only fragments.**
- **Tables split mid-row.**
- **Email chains duplicated across every reply.**
- **Silent invalidation of the golden set** when re-chunking changes chunk IDs. Version the chunker.

### Contextual enrichment: Anthropic's cheapest big win

Anthropic's Contextual Retrieval gives Claude the whole document and one chunk and asks for 50–100 tokens that situate the chunk. That text is prepended before both embedding *and* BM25 indexing. Results (failure rate = 1 − recall@20):

| Setup | Failure rate | Reduction from 5.7% baseline |
|---|---|---|
| Contextual embeddings | 3.7% | 35% |
| + contextual BM25 | 2.9% | 49% |
| + reranking | 1.9% | 67% |

The one-time cost was about $1.02 per million document tokens with prompt caching ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)). These figures date from September 2024 and used Claude 3 Haiku and embedders that have since been superseded. No independent replication was found, and the cost should be recomputed at current Haiku 5.5 prices.

Mechanically, the cached block is the full document and each chunk prompt reads from that cache. Caching needs ≥512 tokens on 5.x models, and below that minimum nothing is cached and no error is raised ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). Store the context in its own column so it can be shown, regenerated or dropped. The knobs are an on/off ingest flag, the model, and the prompt. Anthropic suggests domain-specific prompts that include a glossary. Failure modes:

- **Documents over the context budget.** Split them into sections first.
- **Costs creeping up when caching silently fails.** Check `cache_read_input_tokens`.
- **Hallucinated context on terse chunks.**

### Embeddings: asymmetric, versioned, 1024 dimensions

A bi-encoder maps text to a vector so that cosine similarity approximates relevance. Documents and queries are embedded separately, which is why it is fast and precomputable but also lossy. The embedder interface needs separate `embed_documents()` and `embed_query()` methods because every serious model treats the two asymmetrically: Voyage's `input_type`, Nomic's `search_query:`/`search_document:` prefixes, Qwen3's instructions.

Pick **1024 dimensions** as the kit-wide default. voyage-4 supports it via Matryoshka truncation; qwen3-embedding:0.6b, bge-m3 and snowflake-arctic-embed2 produce it natively; and it sits under pgvector's 2,000-dim HNSW limit for `vector` ([pgvector](https://github.com/pgvector/pgvector)). Matryoshka training makes the first *k* dimensions usable on their own, so you truncate and then re-normalize. The Voyage 4 family shares one embedding space, so you can embed documents once with voyage-4-large and queries with voyage-4-lite or the open-weight voyage-4-nano ([Voyage blog](https://hn.jakechampion.name/reader/https://blog.voyageai.com/2026/01/15/voyage-4)).

Treat the vendor benchmark claims cautiously. Voyage reports voyage-4-large beating OpenAI v3-large by 14.05% on RTEB, which is its own number. Aggregators mix MTEB variants that cannot be compared directly ([Morph](https://www.morphllm.com/ollama-embedding-models)). Anthropic's 2024 experiments named Voyage and Gemini embeddings "particularly effective" ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)).

The knobs are model, dimension, batch size and output quantization. Failure modes:

- **Mixing vectors from two models.** They are not interchangeable even at the same dimension. Store `embed_model` and `dim` on the collection and fail loudly on a mismatch.
- **Forgetting Nomic's prefixes.**
- **mxbai-embed-large's 512-token context** truncating chunks silently.
- **Non-commercial licenses.** Jina v3 and NV-Embed-v2 are CC-BY-NC ([PremAI](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/)).

### Store and ANN index: HNSW with three knobs that matter

HNSW is a multi-layer proximity graph. Each vector's top layer is drawn from an exponentially decaying distribution, so upper layers are sparse "highways". A search descends greedily from the top layer, which gives roughly logarithmic scaling, a bit like a skip list ([Malkov & Yashunin](https://arxiv.org/abs/1603.09320)). In pgvector the knobs and defaults are:

- **`m` = 16:** graph degree, which trades memory for recall.
- **`ef_construction` = 64:** build-time candidate list size.
- **`hnsw.ef_search` = 40:** the runtime recall/latency dial.

Builds are much faster when the graph fits in `maintenance_work_mem` ([pgvector](https://github.com/pgvector/pgvector)). IVFFlat, the alternative, clusters vectors into `lists` (≈rows/1000) and probes `probes` clusters (default 1). It must be built after the data exists.

Two gotchas will bite a company brain. First, pgvector applies `WHERE` filters *after* the index scan, so a restrictive ACL can silently shrink top-k. Turn on `hnsw.iterative_scan = relaxed_order` (capped by `hnsw.max_scan_tuples`, default 20,000) whenever filters apply ([pgvector](https://github.com/pgvector/pgvector)). Second, ef_search caps how many rows the index returns. This comes from the researcher's background knowledge rather than a fetched source, but it means asking for 50–150 candidates at the default of 40 can silently return fewer. Set `hnsw.ef_search` ≥ your candidate count, around 100–200.

At hackathon scale (<100k chunks) an exact scan would also be fast. Keep HNSW anyway so behaviour matches production. The memory levers when the corpus grows are `halfvec` (half the size, little recall loss) and binary quantization with a full-precision rescore ([pgvector](https://github.com/pgvector/pgvector)).

### Hybrid search and RRF: ranks, not scores

Company data is full of exact tokens: ticket IDs, codenames, acronyms, people's names. Dense vectors blur these and BM25 nails them, which is the core argument for hybrid search. Reciprocal Rank Fusion scores each document as RRF(d) = Σ 1/(k + rankᵣ(d)), with k=60 from Cormack et al. (SIGIR 2009). It works because it ignores incomparable raw scores (BM25 values against cosine similarities). At k=60, rank 1 is worth only ~15% more than rank 10, so documents ranked well by *both* retrievers rise to the top ([BigData Boutique](https://bigdataboutique.com/blog/reciprocal-rank-fusion-how-it-works-and-when-to-use-it)). A lower k (30–40) favours top-1 precision ([Digital Applied](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)).

Vanilla Postgres `ts_rank_cd` is not true BM25. This is background knowledge: it lacks BM25's term-frequency saturation and length normalization. It is good enough because RRF only consumes ranks. ParadeDB `pg_search` (Tantivy BM25, fused in one SQL query) is the upgrade path, but its AGPL license was not verified. Learned sparse retrieval is possible with BGE-M3, whose single model emits dense, sparse and multi-vector outputs ([arXiv 2402.03216](https://arxiv.org/abs/2402.03216)). Skip Qdrant's BM42: its benchmark was contested and Qdrant acknowledged errors ([xagi.in](https://xagi.in/blog/qdrants-bm42-please-dont-trust-us/)).

The knobs are `k`, the per-retriever window (50), per-retriever weights, and the text-search config (`'english'` or `'simple'` for mixed languages). Failure modes:

- **Single-retriever hits get penalized.** A document only one retriever finds scores low.
- **Stemming mangles codenames.** Consider `'simple'` for an ID column.
- **Different ACL filters on the two sides.** Both retrievers must apply the identical filter.

### Reranking: a cross-encoder on a shortlist

A cross-encoder feeds `[query, passage]` through one transformer so every query token attends to every passage token. That is far more accurate than a bi-encoder but costs one model pass per candidate, so it runs only on the fused shortlist. ColBERT's MaxSim late interaction sits between the two at a large storage cost. (These mechanism descriptions are background knowledge.) Anthropic's pipeline reranked the top 150 down to 20 and cut failures from 2.9% to 1.9% ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)).

Voyage rerank-3 went GA on 2026-09-29 with 32K context. Its quality numbers are vendor-reported only ([gradually.ai](https://www.gradually.ai/en/ai-models/voyage-rerank-3/)). Billing is (query tokens × number of docs) + total doc tokens, so cost scales with shortlist size times chunk length ([Voyage pricing](https://docs.voyageai.com/docs/pricing)). Hosted rerankers take ~600 ms. Self-hosted Jina Reranker v3 took ~188 ms in one comparison ([Particula](https://particula.tech/blog/reranker-models-compared-cohere-voyage-jina-bge-latency-ndcg)). There is no independent head-to-head of Voyage rerank-3 against Cohere Rerank 4.

The knobs are the shortlist size (50), the final `top_n` (8–10 per tool call), and a minimum score below which results count as "not found". Failure modes:

- **Latency stacking.** Every agent iteration pays the rerank cost.
- **Truncation of long chunks.**
- **A threshold set without evals** that refuses good answers.

Ship a `NoopReranker` so the reranker can be cut instantly.

### The agent loop: sixty lines that Jin must own

The loop works like this:

1. Call `messages.create` with `tools`, `system`, the history and a top-level `cache_control={"type":"ephemeral"}`.
2. While `stop_reason == "tool_use"`, run every `tool_use` block. Use `asyncio.gather` for independent searches, since Claude 4+ models issue parallel calls by default.
3. Append the assistant `content` unchanged.
4. Append **one** user message whose content starts with all the `tool_result` blocks. Separate messages per result "teach" Claude to stop calling tools in parallel ([parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)).

`tool_result` blocks must come first in the content array, with any text after them ([handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)). Handle all seven stop reasons ([stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)):

- `end_turn`: done.
- `pause_turn`: append the content and resend.
- `max_tokens`: raise the limit or continue.
- `refusal`: returns HTTP 200 with `stop_details`.
- `model_context_window_exceeded`: compact or truncate the history.
- `stop_sequence`: done.
- `tool_use`: loop.

Cap the loop at 6–8 iterations as a cost guard. One critical current-model gotcha: Opus 5.5, Sonnet 5.5 and Fable 5.1 **return a 400 for `tool_choice` `any` or `tool`**, so use `auto`, with `strict: true` tools where needed ([implement tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)).

Keep the toolset small. Dust found that 16 high-level tools "produced more errors" ([ZenML/Dust](https://zenml.io/llmops-database/building-a-horizontal-enterprise-agent-platform-with-infrastructure-first-approach)). Write descriptions of at least 3–4 sentences. Return human-readable titles rather than IDs, which Anthropic says reduced hallucinations in retrieval tasks ([writing tools](https://www.anthropic.com/engineering/writing-tools-for-agents)).

Multi-turn rewriting comes free in agent mode because Claude writes its own queries from the full history. In single-shot mode, add an explicit Haiku rewrite step. In a SemEval-2026 multi-turn RAG task, a standalone rewrite of the last turn raised nDCG@5 from 0.3457 to 0.4144 ([arXiv 2606.28352](https://arxiv.org/pdf/2606.28352)). HyDE, multi-query and decomposition each help some query types and hurt others, and each adds ~300 ms ([Anhaia](https://dev.to/gabrielanhaia/hyde-multi-query-decomposition-which-query-rewrite-actually-moves-recall-1m08); that source has internal arithmetic inconsistencies, so treat it as directional). HyDE is especially risky on private company data the model has never seen, because it invents names and IDs. Keep all three off by default.

### Generation and citations: let the API guarantee provenance

`search_kb` returns a list of blocks shaped like `{"type":"search_result","source":"kb://notion:abc#3","title":"PTO Policy > Carryover","content":[{"type":"text","text":...}],"citations":{"enabled":true}}`. Claude's answer then carries `search_result_location` citations with `cited_text`, `search_result_index` and block indices. `cited_text` does not count toward output tokens. The feature is GA on the Claude API, Bedrock, Vertex and Foundry ([search results](https://platform.claude.com/docs/en/build-with-claude/search-results)). Anthropic says API citations are "guaranteed to point to valid locations" ([citations](https://platform.claude.com/docs/en/build-with-claude/citations)).

Three rules break naive implementations:

- **Citations are all-or-nothing** across every search result in a request, and a `tool_result` containing any `search_result` must contain only `search_result`s. Return a plain text "No results found." block on empty results rather than an error.
- **Granularity is per block.** Split each chunk into a few text blocks for sentence-level citations.
- **Citations cannot be combined with structured outputs.** That returns a 400 ([citations](https://platform.claude.com/docs/en/build-with-claude/citations)).

`search_result_index` counts across *all* search results in the request, including earlier turns. So keep a per-conversation registry that maps indices to chunk IDs, built from the stored history.

Pair this with Anthropic's anti-hallucination contract: permission to say "I don't know", answer only from the provided sources, and remove claims without a supporting quote ([reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)). Add a code-level guard that shows a "low grounding" badge when an answer has zero citations. For non-Anthropic providers, the adapter falls back to numbered `[n]` markers in the prompt, parsed with a regex, and approximates the same internal `Citation` type.

### Memory, caching, MCP, tracing and evals: the supporting cast

**Conversation memory** is two Postgres tables, `conversations` and `messages(content_json)`, holding the raw Anthropic blocks, including tool and search-result blocks. History is replayed on every request, which is cheap because cache reads cost 0.1× base input (0.05× on Opus 5.5 and Sonnet 5.5). Writes cost 1.25× for the 5-minute TTL and 2× for 1 hour ([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). The cache prefix order is tools → system → messages, so changing tool definitions, the system prompt or the citations toggle invalidates everything downstream. For long sessions, trim old tool-result payloads or use server-side compaction through `context_management` ([tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)).

**MCP** works in two directions:

- **Inbound:** an adapter lists remote MCP tools at startup and registers them under prefixed names such as `github__search_issues`. It wraps their text output as `search_result` blocks only when every block in the result can be wrapped.
- **Outbound:** `mcp_server.py` exposes `search_kb` and `get_document` over stdio, so Claude Desktop and Claude Code can query the brain.

The spec changed in 2026. The 2026-07-28 version removed protocol-level sessions and the `initialize` handshake. "Most clients still use 2025-11-25", and the Python SDK v2 renames `FastMCP` to `MCPServer`. Maintainers recommended `mcp>=1.27,<2` until v2 is stable ([MCP Playground](https://mcpplaygroundonline.com/blog/build-mcp-server-2026-spec); [python-sdk](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0b1)). Whether v2 is stable now, and which protocol Claude Desktop speaks, are both **unverified**, so test tonight.

**Tracing** emits one trace per request with spans for rewrite, embedding, retrieval, rerank and the LLM call, using the OpenTelemetry GenAI conventions. These are still in "Development" status, now live in their own repo, and capture prompt and response content only when you opt in ([Greptime](https://www.greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions); [semconv-genai](https://github.com/open-telemetry/semantic-conventions-genai)). Pin the semconv version and keep the attribute names in one constants module. Phoenix listens on :4317 gRPC and :6006 HTTP, and defaults to SQLite in a temp directory, so mount `PHOENIX_WORKING_DIR` before storing real data. Its server is Elastic License 2.0, which allows self-hosting but not resale as a hosted service ([Phoenix license](https://www.arize.com/docs/phoenix/self-hosting/license.md)). Langfuse's OTLP endpoint is HTTP-only ([testmuai](https://www.testmuai.com/blog/langfuse-vs-phoenix/)).

**Evals** split into two kinds. Retrieval metrics are deterministic arithmetic over `{query: set(relevant_chunk_ids)}`:

- **Recall@k:** relevant chunks retrieved divided by total relevant. Prioritize it, because "generators can ignore noise but can't recover missing content".
- **Precision@k.**
- **MRR:** the mean of 1/rank of the first relevant result.
- **Hit rate.**
- **nDCG@k:** pair it with recall ([Hamel evals-skills](https://tessl.io/registry/skills/github/hamelsmu/evals-skills/evaluate-rag)).

Generation metrics are LLM-as-judge and should be **binary** PASS/FAIL. Fine-grained scores correlated poorly with humans (Spearman ρ 0.27–0.46). Calibrate judges against ~30–50 hand labels using Cohen's κ, and report recall on failures specifically, since judges catch only 30–60% of inconsistent outputs ([Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/)). Bootstrap the golden set by generating one question per sampled chunk, with that chunk's ID as the label. Add adversarial "lookalike" questions, keep only those rated 4–5 for realism, and include a few not-in-corpus questions to test refusal ([Hamel evals-skills](https://tessl.io/registry/skills/github/hamelsmu/evals-skills/evaluate-rag)). Synthetic questions echo chunk wording, which inflates BM25 scores, so say so when presenting the number.

### Permissions: filter inside the query, fail closed

Every serious product restricts answers to what the user can already see. Copilot grounds only on content the user can access and "the index doesn't change access permissions" ([Microsoft Learn](https://learn.microsoft.com/microsoftsearch/semantic-index-for-copilot)). Onyx stores `external_access {user_emails, group_ids, is_public}` per document ([Onyx ingest API](https://docs.onyx.app/developers/api_reference/files_connectors/ingest_document)). Glean models allowed users and groups with nested memberships ([Glean](https://developers.glean.com/api-info/indexing/documents/permissions)).

The kit maps each source's native permissions to principal strings at index time:

| Source | Principals |
|---|---|
| Drive | `permissions.list` entries |
| Slack public channel | `group:slack-workspace` |
| Slack private channel | the channel's members |
| GitHub public repo | `public` |
| GitHub private repo | `group:gh:org/repo` |
| Notion | workspace-wide |
| Gmail | the mailbox owner and recipients |

Copy the principals onto every chunk. At query time, resolve the user to `{user:email} ∪ groups ∪ {public}` and apply `acl_principals && :principals` inside **both** the dense and the lexical SQL. Never post-filter in Python after the LLM has seen the text. That is OWASP LLM08's "permission-aware vector DB" mitigation ([OWASP](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/)). Documents without an ACL default to private to the ingesting user. Hierarchical inheritance (Drive folders, Notion parents) is the hard part, so use the per-file effective permissions the API returns.

### Knobs at a glance

| Knob | Default | Where | Turn it when |
|---|---|---|---|
| `chunk.size` / `overlap` | 400 / 0–50 tokens | `brain.yaml` → `chunking/markdown.py` | Recall@k drops on long sections; grid-search with evals |
| `enrich.contextual` | off (on for the final demo index) | `enrich/contextual.py` | Corpus is static and under a few million tokens |
| `embed.model` / `dim` | `voyage-4` / 1024 | `embed/` | Going local (`qwen3-embedding:0.6b`). Re-embed everything |
| `hnsw.m` / `ef_construction` | 16 / 64 | `store/schema.sql` | Recall is low at scale; costs build time |
| `hnsw.ef_search` | raise to 100–200 | `store/postgres.py` session `SET` | Always ≥ the candidate window |
| `hnsw.iterative_scan` | `relaxed_order` | `store/postgres.py` | Always on when ACL or source filters apply |
| `retrieval.window` | 50 per retriever | `retrieval/hybrid.py` | Raise for reranker headroom |
| `rrf.k` / weights | 60 / 1:1 | `retrieval/hybrid.py` | Lower k for top-1 precision; weight BM25 up for ID-heavy corpora |
| `rerank.model` / `top_n` | `rerank-3` / 8 | `retrieval/rerank.py` | Noop when cutting latency or cost |
| `agent.max_iterations` | 6 | `agent/loop.py` | Cost guard |
| `llm.answer_model` / `aux_model` | `claude-sonnet-5-5` / `claude-haiku-5-5` | `brain.yaml` | Opus 5.5 for hard synthesis |
| `mode` | `agent` | API param | `single_shot` for the baseline and FAQ latency |
| `otel.capture_content` | false | `.env` | True only locally (PII) |

## Repo shape and the six interfaces that make it pluggable

```
company-brain/
  brain.yaml  .env.example  docker-compose.yml  Makefile  pyproject.toml
  brain/
    config.py            # pydantic-settings Settings (LLM__PROVIDER=anthropic ...), config_hash()
    models.py            # RawDoc, ParsedDoc, Chunk, ACL, Principal, SearchHit, Citation
    connectors/          # base.py (Protocol + @register), _http.py (RateLimitedClient)
                         # local_files.py notion.py gdrive.py gmail.py slack.py github.py
    parsing/             # router.py, markitdown_parser.py, docling_parser.py, passthrough.py
    chunking/            # markdown.py, slack_thread.py, email.py, code.py
    enrich/contextual.py # Claude Haiku + prompt caching; writes chunks.context
    embed/               # base.py, voyage.py, ollama.py, openai_compat.py
    store/               # schema.sql, postgres.py  (later: qdrant.py, lancedb.py)
    retrieval/           # hybrid.py (SQL + rrf), rerank.py, expand.py (neighbors/parents), acl.py
    llm/                 # base.py, anthropic_provider.py, openai_compat_provider.py
    agent/               # loop.py, tools.py, citations.py, prompts/system.md
    sync/manager.py      # hashes, cursors, deletions
    api/                 # app.py, routes_chat.py (SSE), routes_ingest.py, sse.py
    mcp_server.py        # exposes search_kb/get_document over stdio
    tracing.py           # OTel spans: span_retrieval(), span_llm(); semconv constants
  evals/  make_golden.py  metrics.py  run_retrieval.py  judges.py  golden.jsonl  baseline.json
  web/    (Next.js: chat, sources panel, role switcher)
  tests/  docs/architecture.md  docs/frameworks.md
```

The interfaces are deliberately small. Each one is the seam where a provider or backend gets swapped. They are shown as abbreviated Python protocols:

```python
class Connector(Protocol):
    name: str

    def configure(self, cfg: dict, secrets: dict) -> None: ...
    def validate(self) -> None: ...  # fail fast on bad tokens/scopes
    def load_all(self) -> Iterator[RawDoc]: ...

    # optional: poll(since, until) -> Iterator[RawDoc]; list_ids() -> Iterator[str]


class Parser(Protocol):
    def parse(self, content: bytes | str, mime: str) -> ParsedDoc: ...  # markdown + sections


class Chunker(Protocol):
    def chunk(self, doc: ParsedDoc, raw: RawDoc) -> list[Chunk]: ...  # sets embed_text


class Embedder(Protocol):
    model: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Store(Protocol):
    def upsert(self, doc: RawDoc, chunks: list[Chunk]) -> None: ...
    def delete_doc(self, doc_id: str) -> None: ...
    def dense(self, qvec, principals, filters, k) -> list[SearchHit]: ...
    def lexical(self, query, principals, filters, k) -> list[SearchHit]: ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]: ...


class LLMProvider(Protocol):  # normalized events: text_delta, tool_call, citation, stop
    def stream(self, system, messages, tools, **opts) -> AsyncIterator[LLMEvent]: ...
```

The schema is where most of the "company brain" behaviour lives. `owners` and `updated_at` make "who knows X" and "what changed" nearly free, and `acl_principals` makes the role-switch demo a one-line `WHERE` clause:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
  doc_id text PRIMARY KEY, source text, source_type text, url text, title text,
  owners text[], created_at timestamptz, updated_at timestamptz, parent_id text,
  acl_principals text[] NOT NULL, content_hash text, extra jsonb DEFAULT '{}');
CREATE TABLE chunks (
  chunk_id text PRIMARY KEY, doc_id text REFERENCES documents ON DELETE CASCADE,
  ordinal int, section_path text[], text text, context text, embed_text text,
  embedding vector(1024), source text, updated_at timestamptz, acl_principals text[] NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (tsv);
CREATE INDEX ON chunks USING gin (acl_principals);
CREATE TABLE sync_state (connector text PRIMARY KEY, cursor jsonb, updated_at timestamptz);
```

Fusion is short enough to read in one glance, which is the point:

```python
def rrf(*ranked_lists: list[str], k: int = 60, weights=None) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for i, lst in enumerate(ranked_lists):
        w = weights[i] if weights else 1.0
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

## Tomorrow's build order, with cut lines drawn in advance

The plan depends on one rule to check tonight: whether pre-existing code is allowed if declared. Rubrics like TinyFish's require declaring prior work and score "only work done during the hackathon" ([TinyFish](https://sg-hackathon.tinyfish.ai/judging)).

**If the rules allow declared pre-built code**, pre-build tonight:

- the generic kit: repo skeleton, Compose with `pgvector/pgvector:pg17`, provider adapters, embedder, the local-files connector, the MarkItDown/Docling router, the Markdown chunker, the hybrid SQL plus RRF, the agent loop with `search_result` citations, SSE, and a bare chat page;
- a pre-built index of the demo corpus;
- `make demo` booting from that index.

Write a `PRIOR_WORK.md` listing all of it.

**If the rules forbid pre-written code**, pre-build only what is not code:

- every credential, with Google first because it has the most friction (Notion integration shared to pages, Slack internal app invited to channels, GitHub PAT);
- the downloaded datasets and the hand-written fake-company docs;
- warmed Docling models and Docker images;
- this design.

Then use the "from scratch" column below.

Either way, write the 90-second demo script tonight. It should name a user, show how they work today, trigger the demo, land the "aha", and close with why it matters at scale. Guides recommend scripting the demo before writing code, getting a skeleton working in the first quarter of the time, and submitting 30+ minutes early with a backup video ([Masai School](https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/), secondary).

| Hour | With declared pre-built kit | From scratch | Exit criterion |
|---|---|---|---|
| 0–0.5 | Lock the demo script, the 8–10 demo questions and the persona | Same | Questions written down |
| 0.5–2 | Theme connectors: Notion + Slack + GitHub on the SDKs; ingest with ACLs | Compose + schema, local files, Markdown chunker, Voyage embed, dense search, minimal loop | A cited answer from real data |
| 2–3.5 | Role switcher (principals in the query); contextual-retrieval flag; reranker | Add tsvector + RRF, `search_result` citations, SSE to a bare page | Same question, different answer per role |
| 3.5–5 | Differentiator tools: `find_experts`, `whats_changed(since)`, conflict prompt | One theme connector (Notion or GitHub) + ACL role switch | Two "beyond Q&A" moments working |
| 5–6 | Sources/trace panel (tool_start/tool_end, chunks, scores); UI polish | Sources panel; `find_experts` | Panel visible during streaming |
| 6–7 | Golden set (~50 synthetic + 10 adversarial + 3 unanswerable); recall@5, MRR for single-shot vs agent, and with/without rerank | Small golden set (~20); one recall number | One honest number for the slide |
| 7–8 | Hardening: cached answers for scripted questions, backup video, README, architecture slide, submit early | Same | Submitted 30+ min before deadline |

Cut lines are ordered, so the decisions are already made under pressure:

1. **First cut: Gmail and MCP live tools.** Gmail has the highest OAuth cost and only weekly-valid tokens. MCP tools are a talking point, not the core.
2. **Second cut: contextual enrichment and the reranker.** Swap in the `NoopReranker`, which costs quality but not function.
3. **Third cut: the Next.js UI.** Fall back to a ~50-line Streamlit client on the same SSE endpoint.
4. **Fourth cut: Drive and Slack.** Keep local files plus one live connector.

**Never cut:** citations, hybrid search, the role-switch ACL demo, one eval number, and the backup video. These are what make it judgeable. The standard failure modes to avoid are a live API call with no cached fallback, code the team cannot explain in Q&A, no named user, and running out of time for the demo ([Masai School](https://www.masaischool.com/blog/how-to-win-an-ai-hackathon-in-2026-full-guide/)).

## Beat "chat with docs" with visible judgment

Judges reward a working, polished live demo of a named user's real problem, where the AI is clearly necessary. A widely shared rubric scores technical complexity, tool integration, utility, innovation and demo ([TinyFish](https://sg-hackathon.tinyfish.ai/judging)). One practitioner write-up, which describes itself as anecdotal, argues that the weakest of these variables caps the score and names "opaque technical work" and "ornamental model choice" as killers ([DEV Community](https://dev.to/jsb-securedme/what-ai-hackathon-winners-actually-have-in-common-1kla)). Anthropic's recent hackathon winners solved concrete workflows, such as permit corrections using parallel sub-agents ([Claude blog](https://claude.com/blog/meet-the-winners-of-our-built-with-opus-4-6-claude-code-hackathon)).

The differentiators below map to features the big products ship, ranked by wow per hour of build:

- **"Who should I ask about X?"** Aggregate owners of the top-k chunks, weighted by recency, and return people with linked evidence. Glean says "one in ten enterprise searches are people-related" ([Glean](https://docs.glean.com/security/knowledge-graph)).
- **The 15-second role switch.** Ask the same question as Engineer and as HR, and get different, correctly scoped answers.
- **"What changed since Monday?"** Filter on `updated_at`, then cluster and summarize with citations. Quivr and Dust both build around change alerts and digests ([Quivr](https://github.com/QuivrHQ/quivr); [ZenML/Dust](https://zenml.io/llmops-database/building-a-horizontal-enterprise-agent-platform-with-infrastructure-first-approach)).
- **Conflict and staleness flags.** "The handbook says X (March), but #eng decided Y (last week)" surfaces both dated citations and recommends the newer one.
- **A live sources panel.** ChatGPT's company knowledge shows the sources being examined, with snippets ([VentureBeat](https://venturebeat.com/ai/openai-launches-company-knowledge-in-chatgpt-letting-you-access-your-firms)).
- **Draft-and-act behind an approve button.** For example, a GitHub issue or a Slack reply, mirroring Claude connectors' Always allow / Needs approval / Blocked controls ([Claude docs](https://www.claude.com/docs/connectors/getting-started)).
- **An eval badge** showing agent against single-shot recall. This is rare at hackathons.
- **The same brain answering inside Claude Desktop via MCP.** A bonus moment if the stdio server works.

None of this ranking is empirically tied to judge scores. It is synthesis from the rubrics and products above.

The demo corpus should feel like one company:

- **Handbook.** The PostHog handbook (~401 Markdown/MDX files in `PostHog/posthog.com/contents/handbook`, under what reads as MIT-style permission for `/contents/`) plays the Notion/Drive docs ([posthog.com repo](https://github.com/PostHog/posthog.com)). The GitLab handbook is the larger alternative, but its license could not be confirmed ([GitLab Handbook](https://handbook.gitlab.com/handbook/about/)).
- **Email.** An EnronQA inbox or two, with 103,638 emails and 528,304 QA pairs built for private-data RAG, gives an instant eval set. BM25 alone reaches Recall@5 of 87.5 on it ([EnronQA](https://arxiv.org/html/2505.00263v1)). **Its license is not stated**, so check the Hugging Face page before redistributing.
- **Engineering chatter.** A slice of one open-source project's GitHub issues and PRs.
- **Planted moments.** Five to ten hand-written "Acme" documents with deliberate contradictions, dated policy updates, and ACL-restricted HR and finance pages.

Ingest real Slack and Notion workspaces seeded with the Acme content so the connectors are demonstrably live. No public Slack-export-style dataset was researched.

## From hackathon repo to portfolio flagship

After tomorrow, the kit should grow along the axes hiring managers probe: correctness, evaluation, operability and extensibility.

- **v0.2 hardening.** Unit tests for chunkers and metrics; a Postgres testcontainer integration test; and a CI job that fails when recall@5 or MRR drops below the committed `baseline.json`. That job is deterministic and costs nothing in LLM calls. Generation judges run nightly, not per PR.
- **v0.3 sync engine.** Onyx-style `poll` and `list_ids` capabilities, checkpointed backfills, and two-level hashing: a doc hash to skip unchanged documents and a chunk hash to skip re-embedding unchanged chunks. Deletion reconciliation via slim ID listings ([Onyx](https://github.com/onyx-dot-app/onyx/blob/main/backend/onyx/connectors/interfaces.py)).
- **v0.4 pluggability proof.** A Qdrant store adapter, the OpenAI-compatible LLM adapter, and a fully local mode (Ollama Qwen3 embeddings, `bge-reranker-v2-m3`, a local LLM). Python entry points let third-party packages register connectors. Local mode also makes the kit a natural retrieval plugin for Jin's planned self-hosted, everything-is-a-plugin harness (from his memory notes).
- **v0.5 evaluation maturity.** An error-analysis page (trace beside retrieved chunks, a note field and failure tags). Judges calibrated against 30–50 hand labels with κ reported, since a calibration number is strong portfolio signal ([Eugene Yan](https://eugeneyan.com/writing/llm-evaluators/)). HyDE, multi-query and decomposition behind toggles, kept only if evals justify them.
- **v0.6 permissions and freshness.** Drive `permissions.list` sync, private Slack channel membership, and a periodic ACL refresh that does not re-embed.
- **v0.7 deployment.** Neon free Postgres (pgvector included, no card, suspends after 5 min idle) plus an API host, or a ~$5 VPS running Compose behind Caddy ([Neon](https://neon.com/pricing)). Avoid Render's free Postgres, which expires after 30 days, for a lasting link ([Render](https://render.com/articles/platforms-with-a-real-free-tier-for-developers-in-2026.md), vendor-authored).

Graph RAG belongs at the very end, if anywhere. GraphRAG-Bench found GraphRAG "frequently underperforms vanilla RAG" ([arXiv 2506.05690](https://arxiv.org/abs/2506.05690)). One cost model puts LightRAG indexing at ~$330 against ~$1.40 for vector RAG on a 10M-token corpus ([Beri](https://www.beri.net/article/graphrag-vs-vector-rag-indexing-query-cost-lightrag-lazygraphrag-2026)). Add an entities-and-mentions table only if evals show a multi-hop gap. Finally, publish `docs/frameworks.md` with the Pydantic AI equivalent of the agent and an honest comparison of lines of code, transparency and feature coverage.

## Risks and gotchas, with what is and is not verified

| Risk | Evidence | Mitigation | Status |
|---|---|---|---|
| `tool_choice` `any`/`tool` returns 400 on Opus 5.5, Sonnet 5.5 and Fable 5.1 | [Anthropic docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use) | Use `auto` + `strict` tools | Primary doc |
| Citations + structured outputs returns 400; citations all-or-nothing per request | [Citations](https://platform.claude.com/docs/en/build-with-claude/citations); [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results) | Separate calls; text block on empty results | Primary doc |
| Prompt cache silently skipped under 512 tokens | [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) | Log `cache_*_input_tokens` | Primary doc |
| Model IDs and prices change | [Models overview](https://platform.claude.com/docs/en/about-claude/models/overview) | Model names in config, not code | Check on the day |
| Notion API 2025-09-03 breaking change | [Notion](https://developers.notion.com/docs/upgrade-guide-2025-09-03) | Own connector on the current SDK | Primary; loader breakage inferred |
| Slack throttling for distributed apps (1 req/min, 15 objects) | [Slack](https://docs.slack.dev/changelog/2025/05/29/rate-limit-changes-for-non-marketplace-apps) | Internal app only | Primary |
| Google Testing-mode refresh tokens expire in 7 days; `gmail.readonly` is restricted (CASA for public apps) | [Google OAuth](https://developers.google.com/identity/protocols/oauth2); [Nylas](https://developer.nylas.com/docs/cookbook/use-cases/build/google-oauth-scopes/) | Re-auth before the demo; never ship as a public app | Primary/secondary; 100-user cap unverified |
| Gmail `history.list` 404 after history expiry | [Gmail](https://developers.google.com/workspace/gmail/api/guides/sync) | Fall back to full sync | Primary |
| LiteLLM supply-chain compromise (Mar 24, 2026) | [Arthur.ai](https://www.arthur.ai/column/litellm-supply-chain-attack-pypi-compromise-2026) | Own adapters; pin with hashes | Multiple secondary sources |
| MCP spec churn (2026-07-28 stateless); Python SDK v2 in beta | [python-sdk](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0b1) | Pin `mcp>=1.27,<2`, stdio | v2 stable status and Claude Desktop protocol unverified |
| Vercel AI SDK v7 renames, Node 22+, ESM-only | [issoh.co.jp](https://www.issoh.co.jp/tech/details/10402/) | Own SSE schema instead of `useChat` | Single secondary source, unverified |
| Pydantic AI may not pass `search_result` citations through | Not tested in research | Test before adopting | Unverified |
| pgvector filters shrink top-k; `ef_search` caps results | [pgvector](https://github.com/pgvector/pgvector) | `iterative_scan`, raise `ef_search` | Filtering primary; ef cap is background knowledge |
| Licenses: PyMuPDF AGPL; Marker weights restricted; Jina v3 / NV-Embed CC-BY-NC; Phoenix ELv2; Open WebUI branding clause; ParadeDB pg_search AGPL | [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm); [Marker](https://github.com/datalab-to/marker); [PremAI](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/); [Phoenix](https://www.arize.com/docs/phoenix/self-hosting/license.md); [Open WebUI](https://github.com/open-webui/open-webui) | Note in the README; prefer MIT/Apache parts | ParadeDB and Marker code license unverified |
| Dataset licenses: EnronQA not stated; GitLab handbook unconfirmed | [EnronQA](https://arxiv.org/html/2505.00263v1) | Prefer PostHog + hand-written Acme docs | Unverified |
| RAGAS stalled with CVE-2026-6587 | [ecorpit](https://ecorpit.com/rag-evaluation-tools-ragas-deepeval-braintrust-giskard-2026/) | Own metrics; DeepEval optional | Vendor-affiliated source |
| Vendor-only quality claims (Voyage 4, rerank-3, Cohere Rerank 4) | [Voyage blog](https://hn.jakechampion.name/reader/https://blog.voyageai.com/2026/01/15/voyage-4); [gradually.ai](https://www.gradually.ai/en/ai-models/voyage-rerank-3/) | Validate on your own golden set | No independent benchmarks |
| Contextual Retrieval numbers dated (2024, Claude 3 Haiku) | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) | Measure with and without on your own evals | Not replicated |
| Indirect prompt injection from ingested docs | [OWASP LLM08](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/); [Invicti on LLM01](https://invicti.com/blog/web-security/owasp-top-10-risks-llm-security-2025) | Content only in `tool_result`; strip hidden text; approval on write tools; red-team items in the golden set | Primary/secondary |
| Docling ~1 GB install and first-run model download | [yage.ai](https://yage.ai/share/markitdown-survey-en-20260412.html) | Pre-pull tonight; pymupdf4llm fallback | Survey data |

## Conclusion

The research changes what "starter kit" should mean here. The reusable asset is not a framework wrapper or a fork of Onyx, whose default stack runs about ten services ([Onyx](https://github.com/onyx-dot-app/onyx)). It is a set of small, owned contracts: a connector protocol that captures ACLs, a principal-filtered hybrid SQL query, an RRF function, a `search_result` tool, and an SSE event vocabulary. Each vendor-specific part sits behind them as a swappable adapter. The most volatile parts of the 2026 ecosystem are exactly what this design keeps at arm's length: AI SDK versions, MCP protocol revisions, LlamaIndex's TS port, LiteLLM's supply chain and Notion's data-source migration.

For tomorrow, the competitive edge is less retrieval quality than *visible judgment*. Judges see many "chat with docs" bots. Few will see answers that change per user, a cited conflict between last week's Slack thread and the handbook, and a slide showing that the agent beat a single-shot baseline on the team's own golden set. The kit has no unique feature that wins; showing why each part is there, and how it was measured, is what makes it stand out.
