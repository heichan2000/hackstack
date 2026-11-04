"""Every span and attribute name hackstack's tracing emits, in one place.

GenAI names follow the OpenTelemetry GenAI semantic conventions **as of semconv v1.41.0**, the
last release that carried them before they moved to their own repository
(https://github.com/open-telemetry/semantic-conventions-genai). They are still "Development"
status and get renamed between versions, so they are pinned here rather than imported from
``opentelemetry-semantic-conventions``: an upgrade is a reviewed edit of this file.

Phoenix reads OpenInference attributes first-class and converts ``gen_ai.*`` server-side
(Phoenix 15.10+), so each span also carries ``openinference.span.kind`` for its icon and kind
filter. Everything kit-specific is ``hackstack.*``.

Content (prompts, queries, answers, tool arguments) goes on spans only when
``tracing.capture_content`` is true: the ``*_TEXT`` / ``*_ARGUMENTS`` / ``OI_*_VALUE`` names below
are the only ones that carry it.
"""

from __future__ import annotations

GENAI_SEMCONV_VERSION = "1.41.0"

# --- span names (``{name} {model or tool}`` where the conventions say so) ------------------------
SPAN_INVOKE_AGENT = "invoke_agent"
SPAN_CHAT = "chat"
SPAN_EXECUTE_TOOL = "execute_tool"
SPAN_RETRIEVAL = "retrieval"
SPAN_EMBEDDINGS = "embeddings"
SPAN_DENSE = "dense"
SPAN_LEXICAL = "lexical"
SPAN_RRF = "rrf"
SPAN_RERANK = "rerank"
SPAN_EXPAND = "expand"

# --- gen_ai.* (semconv v1.41.0) ------------------------------------------------------------------
OPERATION_NAME = "gen_ai.operation.name"
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"
OP_RETRIEVAL = "retrieval"
OP_EMBEDDINGS = "embeddings"

PROVIDER_NAME = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
REQUEST_MODEL = "gen_ai.request.model"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
USAGE_CACHE_READ_TOKENS = "gen_ai.usage.cache_read.input_tokens"
USAGE_CACHE_CREATION_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
FINISH_REASONS = "gen_ai.response.finish_reasons"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"  # opt-in content
TOOL_CALL_RESULT = "gen_ai.tool.call.result"  # opt-in content
DATA_SOURCE_ID = "gen_ai.data_source.id"
RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"  # opt-in content
RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"  # JSON [{"id", "score"}]: ids, no text
EMBEDDING_DIMENSIONS = "gen_ai.embeddings.dimension.count"

# --- general OTel --------------------------------------------------------------------------------
ERROR_TYPE = "error.type"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
SERVICE_NAME = "service.name"
SERVICE_VERSION = "service.version"

# --- OpenInference (what Phoenix reads natively) -------------------------------------------------
OI_SPAN_KIND = "openinference.span.kind"
OI_PROJECT_NAME = "openinference.project.name"  # resource attribute: the Phoenix project
OI_INPUT_VALUE = "input.value"  # opt-in content
OI_OUTPUT_VALUE = "output.value"  # opt-in content
OI_INPUT_MIME = "input.mime_type"

KIND_AGENT = "AGENT"
KIND_LLM = "LLM"
KIND_TOOL = "TOOL"
KIND_RETRIEVER = "RETRIEVER"
KIND_EMBEDDING = "EMBEDDING"
KIND_RERANKER = "RERANKER"
KIND_CHAIN = "CHAIN"

# --- hackstack.* ---------------------------------------------------------------------------------
HS_CONFIG_HASH = "hackstack.config_hash"  # resource: which config produced this trace
HS_PROFILE = "hackstack.profile"
HS_APP = "hackstack.app"
HS_MODE = "hackstack.agent.mode"  # agent | single_shot
HS_ITERATION = "hackstack.agent.iteration"  # 1-based LLM call number within the turn
HS_ITERATIONS = "hackstack.agent.iterations"
HS_STOP_REASON = "hackstack.agent.stop_reason"
HS_LOW_GROUNDING = "hackstack.agent.low_grounding"
HS_CITATIONS = "hackstack.agent.citations"
HS_TOOL_RESULTS = "hackstack.tool.results"  # passages a tool call returned
HS_PRINCIPALS = "hackstack.retrieval.principals"  # how many, never which
HS_FILTERS = "hackstack.retrieval.filters"  # filter keys in use, e.g. ("source", "since")
HS_WINDOW = "hackstack.retrieval.window"
HS_TOP_N = "hackstack.retrieval.top_n"
HS_INPUT_COUNT = "hackstack.stage.input_count"
HS_OUTPUT_COUNT = "hackstack.stage.output_count"
HS_RRF_K = "hackstack.rrf.k"
HS_RERANK_MODEL = "hackstack.rerank.model"
HS_EXPAND_RADIUS = "hackstack.expand.radius"
