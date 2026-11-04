# Prior work

Hackstack is pre-existing work: everything in this repository outside `apps/<event-app>/` was
built **before** the hackathon, so it can be disclosed to organizers as infrastructure. The
event build is the app folder created at the event.

This file is a stub. It grows with each piece of the kit and is finalized by the
*Demo hardening, PRIOR_WORK.md and backup video* ticket on the
[Hackstack v1 map](https://github.com/heichan2000/hackstack/issues/1).

## Built before the event

| Date | Piece | Where |
|---|---|---|
| 2026-11-03 | Design research | `docs/research/2026-11-03-rag-starter-kit.md` |
| 2026-11-03 | Repo skeleton: config + profiles, data models, registry, component interfaces, CLI entry point, offline test fakes, Makefile, Docker Compose | `pyproject.toml`, `Makefile`, `docker-compose.yml`, `hackstack.yaml`, `hackstack/`, `tests/` |
| 2026-11-04 | Web UI: Next.js chat with streamed cited answers, live sources panel, user switcher, mode toggle | `web/`, `docs/screenshots/` |
