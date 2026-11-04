"""Principal resolution: a demo user -> ``{user:<id>} ∪ groups ∪ {public}``."""

from __future__ import annotations

from pathlib import Path

import pytest

from hackstack.config import Settings
from hackstack.models import Principal
from hackstack.retrieval.acl import PUBLIC, UnknownUserError, principals_for, resolve_principals

REPO_APPS = Path(__file__).resolve().parents[1] / "apps"


def test_principals_for_adds_public_once():
    p = Principal(user="eng@northwind.test", groups=["everyone", "engineering"])
    assert principals_for(p) == [
        "user:eng@northwind.test",
        "group:everyone",
        "group:engineering",
        PUBLIC,
    ]
    assert PUBLIC == "public"


def test_principals_for_dedupes_repeated_groups():
    p = Principal(user="a@x.test", groups=["eng", "eng"])
    assert principals_for(p) == ["user:a@x.test", "group:eng", "public"]


def test_resolve_principals_reads_the_active_apps_users_yaml():
    settings = Settings(app="company_brain")
    got = resolve_principals(settings, "hr@northwind.test", apps_dir=REPO_APPS)
    assert got == ["user:hr@northwind.test", "group:everyone", "group:hr", "public"]


def test_each_northwind_persona_resolves_to_different_groups():
    settings = Settings(app="company_brain")
    eng = resolve_principals(settings, "eng@northwind.test", apps_dir=REPO_APPS)
    fin = resolve_principals(settings, "finance@northwind.test", apps_dir=REPO_APPS)
    assert "group:engineering" in eng and "group:engineering" not in fin
    assert "group:finance" in fin and "group:finance" not in eng


def test_unknown_user_fails_loudly_rather_than_seeing_public_only():
    settings = Settings(app="company_brain")
    with pytest.raises(UnknownUserError, match="nobody@northwind.test"):
        resolve_principals(settings, "nobody@northwind.test", apps_dir=REPO_APPS)


def test_unknown_user_error_is_a_lookup_error():
    assert issubclass(UnknownUserError, LookupError)


def test_resolve_principals_uses_the_app_settings_name(tmp_path):
    root = tmp_path / "apps" / "tiny"
    root.mkdir(parents=True)
    (root / "app.yaml").write_text("title: Tiny\n", encoding="utf-8")
    (root / "prompts").mkdir()
    (root / "prompts" / "system.md").write_text("hi", encoding="utf-8")
    (root / "users.yaml").write_text(
        "users:\n  - id: solo@tiny.test\n    groups: [crew]\n", encoding="utf-8"
    )
    got = resolve_principals(Settings(app="tiny"), "solo@tiny.test", apps_dir=tmp_path / "apps")
    assert got == ["user:solo@tiny.test", "group:crew", "public"]
