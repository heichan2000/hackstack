# Set up Hackstack on a new machine (for Claude Code)

This file is written for **Claude Code** running on a fresh dev machine, such as the user's laptop. The human opens Claude Code in the cloned repo and says: *"Follow `docs/setup-laptop.md`."*

Work through the steps in order. Run every check and show its real output before you say a step is done. Stop and hand the human a precise checklist only when a step truly needs them, such as clicking a UAC prompt, rebooting, signing in, or copying secrets. Don't commit anything this guide creates locally: `.env`, `.secrets/` and model files stay out of git.

## Reference: the desktop this was first set up on (2026-11-03/04)

- Windows 11, RTX 3070 Laptop GPU (8 GB VRAM), 32 GB RAM, Python 3.12, uv 0.13, Node 20, Git Bash and PowerShell.
- Docker Desktop 4.98 (WSL2 backend) and Ollama 0.40.14, both installed with `winget`. GNU make came from `winget install ezwinports.make`.
- **Gotcha 1, stale PATH:** after a `winget` install, shells that were already open can't see `docker`, `ollama` or `make`. Use full paths, or rebuild `PATH` in PowerShell:
  `$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`
  - Docker: `C:\Program Files\Docker\Docker\resources\bin\docker.exe`
  - Ollama: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`
  - make: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\ezwinports.make_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\make.exe`
- **Gotcha 2, port 5432:** the desktop has a native Postgres on 5432. The kit's Compose Postgres therefore publishes on **5433**, and `hackstack.yaml` points there too. If 5433 is also taken on this machine, set both `HACKSTACK_PG_PORT` and `HACKSTACK_STORE__DSN` in `.env` (see `.env.example`).
- **Gotcha 3, piping to prompts:** piping `"y"` into a prompt from PowerShell sends text the prompt rejects. For `make reset`, use Git Bash (`echo y | make reset`) or run `uv run hackstack reset --yes`.
- No Anthropic or Voyage keys. The demo runs on the **`local` profile** (Ollama): `HACKSTACK_PROFILE=local`.

The laptop may differ. Check, don't assume: if it isn't Windows, use the platform's own package manager (Homebrew or apt) for the same tools.

## 0. Look at the machine first

Report these to the human in a short table before installing anything:

- OS and version, CPU, RAM.
- GPU and VRAM: run `nvidia-smi`. No NVIDIA GPU means Ollama runs on CPU or Apple Metal, which is slow but works.
- Free disk space. Plan for about 25 GB: Docker images, Ollama models and the WSL2 disk.
- What's already installed: `git`, `gh`, `uv`, `python --version`, `node --version`, `docker`, `ollama`, `make`.
- Whether port 5433 is free: run `Get-NetTCPConnection -LocalPort 5433 -State Listen` on Windows, or `lsof -i :5433` elsewhere.
- WSL status (Windows only): `wsl --status`. Docker Desktop needs WSL2.

## 1. Repo and GitHub access

1. `gh auth status`. If it isn't logged in, ask the human to run `! gh auth login`. They sign in through the browser, so this step is theirs.
2. If the repo isn't cloned yet, run `gh repo clone heichan2000/hackstack` and `cd` into it. Then run `git pull --ff-only` on `main`.
3. Set git identity if it's missing. Match the desktop with `git config user.name` set to `Chronicz`, and confirm the email with the human.

## 2. Toolchain

Install only what step 0 found missing. On Windows, use `winget install --id <id> -e --accept-package-agreements --accept-source-agreements --silent`:

| Tool | winget id | Check |
|---|---|---|
| Python 3.12 | `Python.Python.3.12` | `python --version` (3.11+ is fine) |
| uv | `astral-sh.uv` | `uv --version` |
| Node 20+ | `OpenJS.NodeJS.LTS` | `node --version` |
| GitHub CLI | `GitHub.cli` | `gh --version` |
| GNU make | `ezwinports.make` | `make --version` |
| Docker Desktop | `Docker.DockerDesktop` | see step 3 |
| Ollama | `Ollama.Ollama` | see step 4 |

- Docker Desktop's installer raises a **UAC prompt**. Tell the human to click Yes.
- If WSL2 isn't installed, `wsl --install` needs admin rights and a **reboot**. Say so plainly and stop. After the reboot the human restarts Claude Code and says "continue `docs/setup-laptop.md` from step 3".

## 3. Docker and the Postgres image

1. Start Docker Desktop (`"C:\Program Files\Docker\Docker\Docker Desktop.exe"`). Wait for the engine with a bounded loop: poll `docker info` for up to about 3 minutes.
2. `docker run --rm hello-world` and `docker compose version` must both succeed.
3. Pre-pull the images so the event day needs no downloads: `docker pull pgvector/pgvector:pg17` and `docker pull arizephoenix/phoenix:latest`.
4. If a pull fails with a credential-helper error, the shell has a stale PATH (gotcha 1). Fix the PATH and retry.

## 4. Ollama and the local models

1. `ollama --version`. `curl http://localhost:11434` should answer "Ollama is running". If it doesn't, start the Ollama app.
2. **Which models:** read the `profiles.local` block of `hackstack.yaml` (the `llm.answer_model`, `llm.aux_model` and `embed.model` keys). If `docs/research/local-models.md` exists, read it too: it holds the chosen chat model, embedder and reranker, the VRAM each one measured, and a fallback for each. Pull exactly those with `ollama pull <name>`.
3. If the laptop has **less VRAM than the desktop** (8 GB), use the fallback that `docs/research/local-models.md` names. Don't edit the committed `hackstack.yaml` for a machine-specific choice. Override it in `.env` instead, for example `HACKSTACK_LLM__ANSWER_MODEL=<fallback>`, and tell the human.
4. Smoke check: `ollama run <chat-model> "say hi"` returns text, and `ollama list` shows every model the profile names.

## 5. Project dependencies, config and secrets

1. Run `make setup`. That's `uv sync`, plus `npm --prefix web install` once a `web/` folder exists.
2. `cp .env.example .env`, then set `HACKSTACK_PROFILE=local`.
3. **Secrets** aren't in git. Ask the human to copy what they've set up on the desktop: the token lines in the desktop's `.env` (`GITHUB_TOKEN`, `NOTION_TOKEN`, `SLACK_BOT_TOKEN`) and, if Google is set up, the `GOOGLE_CLIENT_SECRETS_FILE` JSON and the `.secrets/` folder. Don't ask them to paste secrets into the chat. Have them copy the files over (USB, password manager or a private cloud folder) and confirm where they put them. Leave a key blank if it isn't set up yet: the demo works without connectors.
4. Run `uv run hackstack config` and check that `profile` is `local` and that the DSN port matches the one Compose publishes.

## 6. Bring it up and verify

Run these and show the output:

1. `make test`. It must pass. Tests marked `pg` skip while Postgres is down.
2. `make up`. Postgres reports healthy.
3. `make test` again. The `pg` integration tests now run as well, and all must pass.
4. `uv run hackstack reset --yes`, with `HACKSTACK_PROFILE=local` in `.env`. It prints `index reset for <local embed model> (<dim> dim)`.
5. Whatever has landed by now from the README's make targets. Check `git log` and the [v1 map](https://github.com/heichan2000/hackstack/issues/1): for example `make ingest-sample`, then `uv run hackstack ask "..."`, and `make dev`. A target that prints "not built yet" isn't a failure; say which ones did that.

## 7. Claude Code on this machine

Claude Code memory is per machine, so the laptop starts without the desktop's memory. Save one project memory so later sessions here know how this repo is worked:

- Hackstack v1 is built ticket by ticket from the wayfinder map, [GitHub issue #1](https://github.com/heichan2000/hackstack/issues/1). Claude does all the work and the human only reviews. Commit straight to `main` (no PRs) and push after each ticket. It runs fully local on Ollama, with no Anthropic or Voyage keys. Each ticket is closed with a resolution comment, and gets a line in the map's Decisions so far.

`CLAUDE.md` and `docs/agents/issue-tracker.md` are in the repo already, so the laptop gets them on clone.

## 8. Report

Finish with a short summary for the human:

- What was installed, with versions.
- What every check printed, as pass or fail.
- Which Ollama models were pulled, and whether a fallback was used.
- Which secrets are still missing.
- Every manual step they took or still need to take.
