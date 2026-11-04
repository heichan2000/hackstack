# Hackstack

`README.md` is the canonical spec. Design rationale lives in the research doc, [`docs/research/2026-11-03-rag-starter-kit.md`](docs/research/2026-11-03-rag-starter-kit.md) — read the section a ticket cites, not the whole thing.

## Dev loop

- `make test` (offline; fakes in `tests/fakes.py`), `make lint` (ruff). `uv run hackstack config` prints the resolved settings + `config_hash()`.
- Components register with `@register(kind, name)` (`hackstack/registry.py`) and are picked by name in `hackstack.yaml`; interfaces live in each package's `base.py`.

## Agent skills

### Issue tracker

GitHub Issues on heichan2000/hackstack via `gh`; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Active work

The build is driven by the wayfinder map [Map: Hackstack v1 — the README, built](https://github.com/heichan2000/hackstack/issues/1). Commit straight to `main`; the human reviews results.
