# Hackstack make targets. Recipes are single commands so they run under Git Bash, sh or cmd.
# Pick a profile per command: HACKSTACK_PROFILE=local make ingest-sample
# Phoenix tracing: COMPOSE_PROFILES=tracing make up

.PHONY: help setup up down ingest-sample dev test lint fmt eval mcp reset

help:
	@echo "setup up down ingest-sample dev test lint fmt eval mcp reset  (see README: Make targets)"

## Install Python (uv) and web (npm) dependencies
setup:
	uv sync
ifneq ($(wildcard web/package.json),)
	npm --prefix web install
endif

## Start / stop Postgres (and Phoenix with COMPOSE_PROFILES=tracing)
up:
	docker compose up -d --wait

down:
	docker compose --profile tracing down

## Unit and integration tests; offline, no keys needed
test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

## Drop and recreate the index (needed after changing the embedder); asks first
reset:
	uv run hackstack reset

# --- not built yet: each lands with its own map ticket -----------------------------------
ingest-sample dev eval mcp:
	@echo make $@: not built yet - see the Hackstack v1 map, heichan2000/hackstack issue 1
	@exit 1
