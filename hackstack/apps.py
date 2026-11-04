"""Apps: everything specific to one hackathon idea lives in ``apps/<name>/``.

``app: <name>`` in ``hackstack.yaml`` picks the folder. Its ``app.yaml`` names the system prompt,
the agent tools, the demo users and the connector configs, plus the UI title::

    title: Northwind Brain
    system_prompt: prompts/system.md   # relative paths resolve against the app folder
    users: users.yaml
    default_user: eng@northwind.test
    tools: [search_kb, get_document]
    connectors:
      local_files: {path: sample_corpus}

``hackstack.yaml``'s own ``connectors:`` overrides the app's, connector by connector and key by key.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hackstack.config import Settings
from hackstack.models import Principal

DEFAULT_APPS_DIR = Path("apps")
_PLACEHOLDER = re.compile(r"\{(app_title|user_name|user_role|today|weekday)\}")


class Persona(BaseModel):
    """A demo user from ``users.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = ""
    role: str = ""
    groups: list[str] = Field(default_factory=list)

    def principal(self) -> Principal:
        return Principal(user=self.id, groups=list(self.groups))


class App(BaseModel):
    """A loaded app: ``app.yaml`` with its files read and its paths resolved."""

    name: str
    root: Path
    title: str
    description: str = ""
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    users: list[Persona] = Field(default_factory=list)
    default_user: str | None = None
    connectors: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def user(self, user_id: str) -> Persona:
        for persona in self.users:
            if persona.id == user_id:
                return persona
        known = ", ".join(p.id for p in self.users) or "none"
        raise KeyError(f"app {self.name!r} has no user {user_id!r}; users: {known}")

    def principal(self, user_id: str) -> Principal:
        return self.user(user_id).principal()

    def render_system_prompt(self, user_id: str, *, today: date | None = None) -> str:
        """The system prompt with ``{app_title}``, ``{user_name}``, ``{user_role}``, ``{today}``
        and ``{weekday}`` (today's day name, so "since Monday" can be turned into a date).

        Only those placeholders are replaced; any other braces are left as written.
        """
        persona = self.user(user_id)
        day = today or date.today()
        values = {
            "app_title": self.title,
            "user_name": persona.name or persona.id,
            "user_role": persona.role or "employee",
            "today": day.isoformat(),
            "weekday": day.strftime("%A"),
        }
        return _PLACEHOLDER.sub(lambda m: values[m.group(1)], self.system_prompt)


class _AppFile(BaseModel):
    """The schema of ``app.yaml``. Unknown keys are refused so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str = ""
    system_prompt: str = "prompts/system.md"
    users: str = "users.yaml"
    default_user: str | None = None
    tools: list[str] = Field(default_factory=list)
    connectors: dict[str, dict[str, Any]] = Field(default_factory=dict)


def load_app(settings: Settings, *, apps_dir: str | Path = DEFAULT_APPS_DIR) -> App:
    """Load the app that ``settings.app`` names from ``apps_dir/<name>/``."""
    root = (Path(apps_dir) / settings.app).resolve()
    if not (root / "app.yaml").is_file():
        known = ", ".join(available_apps(apps_dir)) or "none"
        raise ValueError(f"no app {settings.app!r} in {apps_dir}/ (known apps: {known})")
    spec = _AppFile(**_read_yaml(root / "app.yaml"))

    users_file = root / spec.users
    users_data = _read_yaml(users_file) if users_file.exists() else {}
    users = [Persona(**u) for u in users_data.get("users", [])]

    connectors = {name: _resolve_paths(cfg, root) for name, cfg in spec.connectors.items()}
    for name, override in settings.connectors.items():  # paths here stay relative to the cwd
        connectors[name] = {**connectors.get(name, {}), **(override or {})}
    return App(
        name=settings.app,
        root=root,
        title=spec.title,
        description=spec.description,
        system_prompt=(root / spec.system_prompt).read_text(encoding="utf-8"),
        tools=spec.tools,
        users=users,
        default_user=spec.default_user,
        connectors=connectors,
    )


def available_apps(apps_dir: str | Path = DEFAULT_APPS_DIR) -> list[str]:
    """Sorted names of the folders in ``apps_dir`` that hold an ``app.yaml``."""
    base = Path(apps_dir)
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob("*/app.yaml"))


def _resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    out = dict(cfg)
    if isinstance(out.get("path"), str) and not Path(out["path"]).is_absolute():
        out["path"] = str((root / out["path"]).resolve())
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data
