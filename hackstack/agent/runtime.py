"""Wiring: settings -> an :class:`Agent` that answers as a given user.

``hackstack ask``, the ``/chat`` API and the MCP server all build one of these::

    agent = Agent.from_settings(load_settings())
    result = await agent.ask("What is the PTO carryover policy?", user="eng@northwind.test")

Build it once and reuse it: the retriever holds the store connection and the reranker model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from hackstack.agent.citations import CitationRegistry
from hackstack.agent.loop import AgentResult, collect, stream_agent, stream_single_shot
from hackstack.agent.tools import CORE_TOOLS, Tool, ToolContext, load_tools
from hackstack.apps import DEFAULT_APPS_DIR, App, load_app
from hackstack.config import Settings
from hackstack.llm.base import LLMProvider
from hackstack.retrieval.acl import UnknownUserError, principals_for

DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "system.md"
Mode = Literal["agent", "single_shot"]


def default_system_prompt(*, app_title: str = "Hackstack", user: str = "", role: str = "") -> str:
    """The kit's own prompt (``agent/prompts/system.md``), for setups without an app folder."""
    text = DEFAULT_PROMPT.read_text(encoding="utf-8")
    values = {
        "{app_title}": app_title,
        "{user_name}": user or "a user",
        "{user_role}": role or "employee",
        "{today}": date.today().isoformat(),
    }
    for key, value in values.items():
        text = text.replace(key, value)
    return text


@dataclass
class Agent:
    settings: Settings
    llm: LLMProvider
    retriever: Any  # HybridRetriever (anything with .search() and .store.neighbors())
    tools: list[Tool]
    app: App | None = None
    llm_opts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        apps_dir: str | Path = DEFAULT_APPS_DIR,
        llm: LLMProvider | None = None,
        retriever: Any = None,
    ) -> Agent:
        from hackstack.llm import make_llm

        app = load_app(settings, apps_dir=apps_dir) if _has_app(settings, apps_dir) else None
        if retriever is None:
            from hackstack.retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever.from_settings(settings)
        names = app.tools if app is not None and app.tools else CORE_TOOLS
        return cls(
            settings=settings,
            llm=llm if llm is not None else make_llm(settings),
            retriever=retriever,
            tools=load_tools(names, app_root=app.root if app is not None else None),
            app=app,
        )

    @property
    def default_user(self) -> str | None:
        return self.app.default_user if self.app is not None else None

    def principals(self, user: str) -> list[str]:
        """``user``'s principal strings. Raises :class:`UnknownUserError` for unknown users."""
        if self.app is None:
            raise UnknownUserError(f"no app folder for {self.settings.app!r}, so no users")
        try:
            return principals_for(self.app.principal(user))
        except KeyError as exc:
            raise UnknownUserError(exc.args[0] if exc.args else user) from exc

    def system_prompt(self, user: str) -> str:
        if self.app is not None:
            return self.app.render_system_prompt(user)
        return default_system_prompt(user=user)

    def context(self, user: str, messages: list[dict[str, Any]] | None = None) -> ToolContext:
        return ToolContext(
            principals=self.principals(user),
            retriever=self.retriever,
            citations=CitationRegistry.from_messages(messages or []),
            user=user,
            document_chars=4 * self.settings.expand.max_chars,
            settings=self.settings,
            app=self.app,
        )

    def stream(
        self,
        question: str | None = None,
        *,
        user: str,
        history: list[dict[str, Any]] | None = None,
        mode: Mode | None = None,
    ) -> AsyncIterator[Any]:
        """Events for one turn: ``question`` appended to ``history`` (or ``history`` as is)."""
        messages = list(history or [])
        if question is not None:
            messages.append({"role": "user", "content": question})
        ctx = self.context(user, messages)
        system = self.system_prompt(user)
        if (mode or self.settings.agent.mode) == "single_shot":
            return stream_single_shot(
                self.llm, system=system, messages=messages, ctx=ctx, llm_opts=self.llm_opts
            )
        return stream_agent(
            self.llm,
            system=system,
            messages=messages,
            tools=self.tools,
            ctx=ctx,
            max_iterations=self.settings.agent.max_iterations,
            llm_opts=self.llm_opts,
        )

    async def ask(
        self,
        question: str,
        *,
        user: str,
        history: list[dict[str, Any]] | None = None,
        mode: Mode | None = None,
    ) -> AgentResult:
        return await collect(self.stream(question, user=user, history=history, mode=mode))


def _has_app(settings: Settings, apps_dir: str | Path) -> bool:
    return (Path(apps_dir) / settings.app / "app.yaml").is_file()
