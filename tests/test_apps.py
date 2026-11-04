from datetime import date
from pathlib import Path

import pytest

from hackstack.apps import load_app
from hackstack.config import load_settings

REPO_APPS = Path(__file__).resolve().parents[1] / "apps"


def settings_for(tmp_path, body: str):
    path = tmp_path / "hackstack.yaml"
    path.write_text(body, encoding="utf-8")
    return load_settings(path)


def test_company_brain_loads_prompt_tools_users_and_corpus(tmp_path):
    app = load_app(settings_for(tmp_path, "app: company_brain\n"), apps_dir=REPO_APPS)

    assert app.name == "company_brain"
    assert app.title == "Northwind Brain"
    assert "Northwind Labs" in app.system_prompt
    assert "search_kb" in app.tools
    finance = app.principal("finance@northwind.test")
    assert finance.user == "finance@northwind.test"
    assert "finance" in finance.groups
    corpus = Path(app.connectors["local_files"]["path"])
    assert corpus.is_absolute()
    assert corpus == (REPO_APPS / "company_brain" / "sample_corpus").resolve()


def test_switching_app_in_hackstack_yaml_changes_prompt_tools_and_users(tmp_path):
    brain = load_app(settings_for(tmp_path, "app: company_brain\n"), apps_dir=REPO_APPS)
    template = load_app(settings_for(tmp_path, "app: _template\n"), apps_dir=REPO_APPS)

    assert template.name == "_template"
    assert template.title != brain.title
    assert template.system_prompt != brain.system_prompt
    assert "find_experts" in brain.tools and "find_experts" not in template.tools
    assert {u.id for u in template.users}.isdisjoint({u.id for u in brain.users})
    assert template.connectors["local_files"]["path"] != brain.connectors["local_files"]["path"]


def make_app(apps_dir: Path, name: str, app_yaml: str, prompt: str = "Hi {user_name}.") -> Path:
    root = apps_dir / name
    (root / "prompts").mkdir(parents=True)
    (root / "app.yaml").write_text(app_yaml, encoding="utf-8")
    (root / "prompts" / "system.md").write_text(prompt, encoding="utf-8")
    return root


def test_hackstack_yaml_connectors_override_the_apps_key_by_key(tmp_path):
    make_app(
        tmp_path / "apps",
        "demo",
        "title: Demo\nconnectors:\n  local_files: {path: corpus, glob: '*.md'}\n"
        "  notion: {pages: [abc]}\n",
    )
    settings = settings_for(
        tmp_path,
        "app: demo\nconnectors:\n"
        "  local_files: {path: /data/elsewhere}\n  slack: {channels: [eng]}\n",
    )

    app = load_app(settings, apps_dir=tmp_path / "apps")

    assert app.connectors["local_files"] == {"path": "/data/elsewhere", "glob": "*.md"}
    assert app.connectors["notion"] == {"pages": ["abc"]}
    assert app.connectors["slack"] == {"channels": ["eng"]}


def test_system_prompt_is_rendered_for_the_asking_user(tmp_path):
    make_app(
        tmp_path / "apps",
        "demo",
        "title: Demo App\n",
        prompt='{app_title} for {user_name} ({user_role}) on {today}. JSON: {"k": 1} {unknown}',
    )
    (tmp_path / "apps" / "demo" / "users.yaml").write_text(
        "users:\n  - {id: a@x.test, name: Ann, role: Engineer, groups: [eng]}\n", encoding="utf-8"
    )
    app = load_app(settings_for(tmp_path, "app: demo\n"), apps_dir=tmp_path / "apps")

    prompt = app.render_system_prompt("a@x.test", today=date(2026, 11, 4))

    assert prompt == 'Demo App for Ann (Engineer) on 2026-11-04. JSON: {"k": 1} {unknown}'


def test_unknown_app_or_user_fails_with_the_known_names(tmp_path):
    make_app(tmp_path / "apps", "demo", "title: Demo\n")

    with pytest.raises(ValueError, match="no app 'nope'.*known apps: demo"):
        load_app(settings_for(tmp_path, "app: nope\n"), apps_dir=tmp_path / "apps")

    brain = load_app(settings_for(tmp_path, "app: company_brain\n"), apps_dir=REPO_APPS)
    with pytest.raises(KeyError, match="finance@northwind.test"):
        brain.principal("nobody@northwind.test")
