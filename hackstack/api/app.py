"""The FastAPI app: ``/chat`` over SSE, conversation memory, ``/ingest``, ``/users``, ``/health``.

Run it with ``make dev`` (``uvicorn --factory hackstack.api.app:create_app`` on :8000). Settings
come from ``hackstack.yaml`` and ``HACKSTACK_PROFILE`` as everywhere else.

Built once at startup and shared by every request: the :class:`~hackstack.agent.runtime.Agent`
(LLM client, retriever with its store connection and reranker) and the conversation store.
Tests pass their own (fakes) to :func:`create_app` instead.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hackstack.apps import DEFAULT_APPS_DIR
from hackstack.config import Settings, load_settings

log = logging.getLogger(__name__)


def _conversations_for(settings: Settings) -> Any:
    if settings.store.provider == "postgres":
        from hackstack.store.conversations import PostgresConversations

        return PostgresConversations.from_settings(settings)
    from hackstack.store.conversations import InMemoryConversations

    log.warning("store.provider is %r: conversations kept in memory only", settings.store.provider)
    return InMemoryConversations()


def create_app(
    settings: Settings | None = None,
    *,
    agent: Any = None,
    conversations: Any = None,
    apps_dir: str | Path = DEFAULT_APPS_DIR,
) -> FastAPI:
    settings = settings if settings is not None else load_settings()
    from hackstack.tracing import setup_tracing

    setup_tracing(settings)  # no-op unless tracing.enabled

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = app.state
        owned = []
        if state.agent is None:
            from hackstack.agent.runtime import Agent

            state.agent = Agent.from_settings(settings, apps_dir=apps_dir)
            reranker = getattr(state.agent.retriever, "reranker", None)
            if hasattr(reranker, "load"):
                await asyncio.to_thread(reranker.load)  # skip the cold load on the first question
        if state.conversations is None:
            state.conversations = await asyncio.to_thread(_conversations_for, settings)
            owned.append(state.conversations)
        log.info("hackstack API ready: app=%s profile=%s", settings.app, settings.profile)
        try:
            yield
        finally:
            for thing in owned:
                thing.close()

    app = FastAPI(title="Hackstack", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.apps_dir = Path(apps_dir)
    app.state.agent = agent
    app.state.conversations = conversations
    app.state.locks = defaultdict(asyncio.Lock)
    app.state.ingest_lock = asyncio.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.api.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Conversation-Id"],
    )

    from hackstack.api.routes_chat import router as chat_router
    from hackstack.api.routes_ingest import router as ingest_router

    app.include_router(chat_router)
    app.include_router(ingest_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "app": settings.app, "profile": settings.profile}

    return app
