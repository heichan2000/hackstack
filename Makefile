# Hackstack make targets. Recipes are single commands so they run under Git Bash, sh or cmd.
# Pick a profile per command: HACKSTACK_PROFILE=local make ingest-sample
# Phoenix tracing: COMPOSE_PROFILES=tracing make up

.PHONY: help setup ollama-models up down ingest-sample dev dev-api dev-web test test-web lint fmt eval golden mcp reset

help:
	@echo "setup ollama-models up down ingest-sample dev test lint fmt eval golden mcp reset  (see README: Make targets)"

## Install Python (uv) and web (npm) dependencies; the rerank extra is the local cross-encoder
setup:
	uv sync --extra rerank
ifneq ($(wildcard web/package.json),)
	npm --prefix web install
endif

## Local profile: pull the Ollama models and build hackstack-chat (qwen3.5:4b at a 16k context)
ollama-models:
	ollama pull qwen3.5:4b
	ollama pull qwen3-embedding:0.6b
	ollama create hackstack-chat -f ollama/hackstack-chat.Modelfile

## Start / stop Postgres (and Phoenix with COMPOSE_PROFILES=tracing)
up:
	docker compose up -d --wait

down:
	docker compose --profile tracing down

## Unit and integration tests; offline, no keys needed
test:
	uv run pytest

## Web UI: vitest unit tests, eslint, typecheck (needs `make setup` for web/node_modules)
test-web:
	npm --prefix web test
	npm --prefix web run lint
	npm --prefix web run typecheck

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

## Drop and recreate the index (needed after changing the embedder); asks first
reset:
	uv run hackstack reset

## Ingest the active app's sample corpus (`app:` in hackstack.yaml); safe to re-run
ingest-sample:
	uv run hackstack ingest

## Compare dense / lexical / hybrid / hybrid+rerank on evals/golden.jsonl; fails below baseline
eval:
	uv run python -m evals.run_retrieval --check

## Regenerate evals/golden.jsonl from the corpus with the aux LLM (then review it, re-baseline)
golden:
	uv run python -m evals.make_golden --lookalikes 30

## API on :8000 (and the web UI on :3000 once web/ exists), with reload. SSE contract: docs/sse.md
ifneq ($(wildcard web/package.json),)
dev:
	$(MAKE) -j2 dev-api dev-web
else
dev: dev-api
endif

dev-api:
	uv run uvicorn hackstack.api.app:create_app --factory --reload --reload-dir hackstack --host 127.0.0.1 --port 8000

dev-web:
	npm --prefix web run dev

## Serve search_kb/get_document over stdio for Claude Desktop/Code (as $HACKSTACK_AS_USER)
mcp:
	uv run python -m hackstack.mcp_server
