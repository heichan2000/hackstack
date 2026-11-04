"""``POST /ingest``: sync the active app's sources from the API (same as ``hackstack ingest``).

Runs in a worker thread so chats keep streaming; one ingest at a time (a second gets 409). The
kit has no auth: bind the API to localhost (``make dev`` does) and do not expose it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

router = APIRouter()


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    connectors: list[str] | None = None  # default: every connector the app configures
    full: bool = False  # ignore saved cursors and re-read every source item


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest | None = None) -> dict[str, Any]:
    from hackstack.ingest import ingest_app

    state = request.app.state
    body = body or IngestRequest()
    if state.ingest_lock.locked():
        raise HTTPException(409, "an ingest is already running")
    async with state.ingest_lock:
        try:
            reports = await asyncio.to_thread(
                ingest_app,
                state.settings,
                connectors=body.connectors,
                full=body.full,
                apps_dir=state.apps_dir,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc.args[0] if exc.args else exc)) from exc
    return {"reports": [r.model_dump(mode="json") for r in reports]}
