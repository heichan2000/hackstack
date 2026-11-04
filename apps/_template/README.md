# App template

Copy this folder to start a hackathon app:

```bash
cp -r apps/_template apps/<idea>
```

Then set `app: <idea>` in `hackstack.yaml`. Check it loads with `uv run hackstack app`.

| File | What it is |
|---|---|
| `app.yaml` | UI title, system prompt, enabled tools, demo users, connector configs |
| `prompts/system.md` | The system prompt. `{app_title}`, `{user_name}`, `{user_role}` and `{today}` are filled in per request |
| `users.yaml` | Demo personas and their groups, for permission-aware answers |
| `sample_corpus/` | Demo documents, ingested by `make ingest-sample` |
| `demo-script.md` | The 90-second pitch and the demo questions |
| `tools.py` | Optional: your app's own agent tools, listed by name in `app.yaml` |

Corpus metadata (ACLs, owners, dates) is described in `apps/company_brain/README.md`.
