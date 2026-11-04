"""``POST /chat`` (SSE), ``GET /conversations/{id}``, ``GET /users``.

One turn::

    validate user + conversation (4xx before any streaming)
    history = stored messages (Postgres), never the client's
    sent    = fit_history(history)               # old tool results trimmed to fit the context
    stream agent events as SSE; on done, store result.messages[len(sent):] and send grounding/done

Nothing is stored for a turn that fails: the history stays valid for the next one.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from hackstack.agent.history import fit_history
from hackstack.agent.loop import AgentDone
from hackstack.api.sse import SSEEncoder
from hackstack.retrieval.acl import UnknownUserError
from hackstack.store.conversations import Conversation, ConversationStore

log = logging.getLogger(__name__)
router = APIRouter()

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    """The whole request: the history is rebuilt server-side, so there is no field for it."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str | None = Field(default=None, pattern=ID_PATTERN)
    message: str = Field(min_length=1, max_length=8000)
    as_user: str | None = None  # the demo user switcher; default: the app's default_user
    mode: Literal["agent", "single_shot"] | None = None


def _state(request: Request) -> Any:
    return request.app.state


def _resolve_user(agent: Any, as_user: str | None) -> str:
    user = as_user or agent.default_user
    if not user:
        raise HTTPException(400, "as_user is required: this app has no default_user")
    try:
        agent.principals(user)
    except UnknownUserError as exc:
        raise HTTPException(400, str(exc.args[0] if exc.args else exc)) from exc
    return user


async def _conversation(
    store: ConversationStore, conversation_id: str, user: str, app_name: str
) -> Conversation:
    conv = await asyncio.to_thread(store.get, conversation_id)
    if conv is None:
        conv = await asyncio.to_thread(store.create, conversation_id, as_user=user, app=app_name)
    if conv.as_user != user:
        # Its history holds passages only conv.as_user may read: start a new conversation.
        raise HTTPException(
            409, f"conversation {conversation_id!r} belongs to another user; start a new one"
        )
    return conv


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    state = _state(request)
    agent, store = state.agent, state.conversations
    user = _resolve_user(agent, body.as_user)
    conversation_id = body.conversation_id or uuid.uuid4().hex
    await _conversation(store, conversation_id, user, agent.settings.app)

    async def events():
        async with state.locks[conversation_id]:  # one turn at a time per conversation
            try:
                history = await asyncio.to_thread(store.messages, conversation_id)
                # Fit with the new question as the protected last turn, so earlier turns'
                # tool results are trimmed (oldest first) whenever the budget needs it.
                question = {"role": "user", "content": body.message}
                sent = fit_history(
                    [*history, question],
                    budget_tokens=agent.settings.agent.history_tokens,
                    keep_turns=1,
                )[:-1]
                encoder = SSEEncoder(conversation_id, sent)
                async for event in agent.stream(
                    body.message, user=user, history=sent, mode=body.mode
                ):
                    if isinstance(event, AgentDone):
                        new = event.result.messages[len(sent) :]
                        await asyncio.to_thread(
                            store.append, conversation_id, new, start=len(history)
                        )
                    for chunk in encoder.encode(event):
                        yield chunk
            except asyncio.CancelledError:  # client went away: nothing stored, nothing to send
                raise
            except Exception as exc:
                log.exception("chat turn failed (conversation %s)", conversation_id)
                yield SSEEncoder.error(f"{type(exc).__name__}: {exc}")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "X-Conversation-Id": conversation_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    """The conversation as UI turns (questions and answers), plus the raw stored messages."""
    store = _state(request).conversations
    conv = await asyncio.to_thread(store.get, conversation_id)
    if conv is None:
        raise HTTPException(404, f"no conversation {conversation_id!r}")
    messages = await asyncio.to_thread(store.messages, conversation_id)
    turns: list[dict[str, str]] = []
    for m in messages:
        content = m["content"]
        is_tool_results = isinstance(content, list) and all(
            b.get("type") == "tool_result" for b in content
        )
        if m["role"] == "user" and not is_tool_results:
            turns.append({"role": "user", "text": _text(content)})
        elif m["role"] == "assistant":
            text = _text(content)
            if turns and turns[-1]["role"] == "assistant":
                turns[-1]["text"] = text or turns[-1]["text"]  # the turn's last answer wins
            elif text:
                turns.append({"role": "assistant", "text": text})
    return {"id": conv.id, "as_user": conv.as_user, "app": conv.app, "turns": turns,
            "messages": messages}  # fmt: skip


@router.get("/users")
async def users(request: Request) -> dict[str, Any]:
    """The app's demo users, for the UI's user switcher."""
    agent = _state(request).agent
    app = agent.app
    return {
        "app": agent.settings.app,
        "title": app.title if app is not None else "Hackstack",
        "default_user": agent.default_user,
        "users": [u.model_dump() for u in app.users] if app is not None else [],
    }
