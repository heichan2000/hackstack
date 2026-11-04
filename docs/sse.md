# The `/chat` SSE contract

The API and the UI talk through one owned event vocabulary instead of a UI SDK's, so either side can be swapped. Code: `hackstack/api/sse.py` (encoder) and `hackstack/api/routes_chat.py` (route). Run the API with `make dev`, which serves it on `http://127.0.0.1:8000`.

## Request

`POST /chat` with a JSON body:

| Field | Type | Notes |
|---|---|---|
| `message` | string, required | The user's question (1–8000 chars). |
| `conversation_id` | string, optional | `[A-Za-z0-9_-]{1,64}`. Omit it to start a new conversation; an id the server hasn't seen also starts one under that id, so a client may generate its own. |
| `as_user` | string, optional | A user id from `GET /users`. Defaults to the app's `default_user`. |
| `mode` | `"agent"` \| `"single_shot"`, optional | Defaults to `agent.mode` in `hackstack.yaml`. `single_shot` retrieves once, then makes one call without tools. |

Unknown fields are rejected with 422. In particular **there is no `history` field**: the server rebuilds the conversation from Postgres on every turn and never trusts the client's.

Errors before streaming starts are plain JSON (`{"detail": "..."}`):

- **400**: unknown `as_user`, or no `as_user` and the app has no default user.
- **409**: the conversation belongs to another user. Its history holds passages only that user may read, so **switching users must start a new conversation**.
- **422**: bad body or bad `conversation_id`.

## Response

`200`, `Content-Type: text/event-stream`. The response header `X-Conversation-Id` carries the conversation id; CORS exposes it to the dev UI origin. The id is also in the final `done` event.

Each frame is one `event:` line plus one `data:` line of JSON. The JSON always repeats the event name as `type`, so a client can ignore the `event:` line:

```
event: text_delta
data: {"type": "text_delta", "text": "Five days "}

```

## Events

### Order

```
( tool_start | tool_end | text_delta | citation )*   grounding   done
                                                      or, at any point:   error
```

- Every `tool_start` is followed later by a `tool_end` with the same `id`. Parallel calls send all their `tool_start`s, then all their `tool_end`s, in call order.
- Text can arrive before tools: the model may say "Let me check" and then search. The answer is the text of the **last** model step: when a `tool_start` follows some text, that text was a preamble. `done.answer` holds the final text either way.
- A stream ends after exactly one `done` (always preceded by one `grounding`) or one `error`.

### `text_delta`

```json
{"type": "text_delta", "text": "Five days [1]."}
```

Streamed answer text. Append in order. `[n]` markers are citation numbers; see `citation`.

### `citation`

```json
{"type": "citation", "number": 1,
 "citation": {"doc_id": "handbook-pto", "chunk_id": "handbook-pto#3",
              "title": "PTO Policy > Carryover (updated 2026-03-02)",
              "url": null, "cited_text": "Up to five days carry over."}}
```

The answer cited a passage. `number` is the passage's number in the conversation: the same `n` as a `[n]` marker, and the same as `hits[].number` in the `tool_end` that showed it. It can be `null` if the model cited something the server can't match. Numbers keep counting across turns, so turn 2's first new passage is not `[1]`.

### `tool_start`

```json
{"type": "tool_start", "id": "call_0", "name": "search_kb",
 "input": {"query": "PTO carryover", "since": null}, "query": "PTO carryover"}
```

A tool call began. Drives the live "Searching: 'PTO carryover'…" panel. `query` is lifted from the input (`query`, `topic`, `doc`, …) and is `null` for tools without one. In `single_shot` mode there is one synthetic call named `retrieve`.

### `tool_end`

```json
{"type": "tool_end", "id": "call_0", "name": "search_kb", "query": "PTO carryover",
 "is_error": false, "results": 2, "seconds": 1.42,
 "summary": "PTO Policy > Carryover (updated 2026-03-02); ...",
 "hits": [{"number": 1, "chunk_id": "handbook-pto#3", "doc_id": "handbook-pto",
           "title": "PTO Policy > Carryover (updated 2026-03-02)", "score": 0.9132, "url": null}]}
```

The call finished.

- `hits` are the passages the model was shown, already ACL-filtered, with their citation numbers and retrieval scores. With the reranker on, `score` is the reranker's.
- `results` is `len(hits)`. It is 0 for text-only tools such as `get_document`, and when every match was already shown earlier in the conversation.
- On failure, `is_error` is true and `summary` holds the error the model was given; the turn goes on.

### `grounding`

```json
{"type": "grounding", "low_grounding": false, "cited": [1],
 "sources": [{"number": 1, "chunk_id": "handbook-pto#3", "doc_id": "handbook-pto",
              "title": "PTO Policy > Carryover (updated 2026-03-02)", "url": null, "cited": true}]}
```

Sent once, just before `done`.

- `sources` lists every passage shown to the model this conversation, so far as the model was sent it this turn, with `cited` marking those this answer cited. It feeds the sources panel.
- `low_grounding` is true when the answer cited nothing: show a "low grounding" badge.

### `done`

```json
{"type": "done", "conversation_id": "3f2a…", "answer": "Five days [1].",
 "citations": [ … ], "low_grounding": false, "stop_reason": "end_turn",
 "truncated": false, "refusal": null, "iterations": 2,
 "usage": {"input_tokens": 2310, "output_tokens": 48}, "seconds": 6.8, "mode": "agent"}
```

The turn finished and was stored. `stop_reason` is one of these:

- `end_turn` or `stop_sequence`: normal.
- `max_tokens`: the answer was cut off; `truncated` is true.
- `refusal`: see `refusal.category` and `refusal.explanation`.
- `max_iterations`: the tool budget ran out.
- `model_context_window_exceeded`: the history grew too long. Start a new conversation.

The raw message history is never sent to the client.

### `error`

```json
{"type": "error", "code": "internal_error", "message": "LLMError: connection refused"}
```

The turn failed after streaming began, for example because the model or the database went away. **Nothing from this turn is stored**, so the user can simply ask again. A client that disconnects mid-turn also stores nothing.

## Conversation memory

- **Tables:** `conversations(id, as_user, app, …)` and `messages(conversation_id, seq, role, content_json)`, created on API startup (`hackstack/store/conversations.sql`). `content_json` holds the provider's content blocks verbatim, including tool_use, tool_result, search_result and thinking signatures. That way the history replays exactly and citation numbers stay aligned.
- **Long conversations:** before each turn the stored history is fitted to `agent.history_tokens` (`hackstack/agent/history.py`).
  - First, passage text in older turns is trimmed to a placeholder. Citation numbers are unchanged, and a trimmed passage comes back if the model searches again.
  - If that's not enough, the oldest turns are dropped.
  - The stored copy stays complete.
  - The local profile uses `history_tokens: 5000` for its 16k-context model.
- **One turn at a time:** turns run one at a time per conversation. Concurrent chats in different conversations run in parallel, but the local Ollama serves one model request at a time, so they queue there.

## Other routes

| Route | Returns |
|---|---|
| `GET /users` | `{app, title, default_user, users: [{id, name, role, groups}]}`, for the user switcher. |
| `GET /conversations/{id}` | `{id, as_user, app, turns: [{role, text}], messages}`, so a page reload can redraw the chat. `404` if unknown. |
| `POST /ingest` | Body `{connectors?: [...], full?: bool}` runs the active app's ingest, the same as `hackstack ingest`, and returns `{reports: [...]}`. `409` while another ingest runs. |
| `GET /health` | `{status: "ok", app, profile}`. |

The kit has no authentication. `as_user` is a demo switcher, trusted as sent. Keep the API on localhost, which is where `make dev` binds it.

## A minimal client

```js
const res = await fetch("http://127.0.0.1:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ conversation_id, message, as_user }),
});
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
let buf = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += value;
  let i;
  while ((i = buf.indexOf("\n\n")) >= 0) {
    const frame = buf.slice(0, i); buf = buf.slice(i + 2);
    const data = frame.split("\n").filter(l => l.startsWith("data: ")).map(l => l.slice(6)).join("");
    if (data) handle(JSON.parse(data));   // switch on .type
  }
}
```

`curl` shows the raw stream: `curl -N -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"What is the PTO carryover policy?"}'`.
