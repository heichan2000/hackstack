import json

from typer.testing import CliRunner

from hackstack.cli import app

runner = CliRunner()


def write_cfg(tmp_path):
    path = tmp_path / "hackstack.yaml"
    path.write_text(
        "app: demo\nprofiles:\n  local:\n    embed: {provider: ollama, model: m}\n",
        encoding="utf-8",
    )
    return path


def test_config_command_shows_profile_hash_and_settings(tmp_path):
    result = runner.invoke(app, ["config", "--file", str(write_cfg(tmp_path))])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["profile"] == "default"
    assert len(out["config_hash"]) == 16
    assert out["settings"]["app"] == "demo"


def test_config_command_honours_hackstack_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HACKSTACK_PROFILE", "local")
    result = runner.invoke(app, ["config", "--file", str(write_cfg(tmp_path))])
    out = json.loads(result.output)
    assert out["profile"] == "local"
    assert out["settings"]["embed"]["provider"] == "ollama"


def test_app_command_shows_the_active_apps_title_users_tools_and_connectors(tmp_path):
    from pathlib import Path

    apps = Path(__file__).resolve().parents[1] / "apps"
    cfg = tmp_path / "hackstack.yaml"
    cfg.write_text("app: company_brain\n", encoding="utf-8")

    result = runner.invoke(app, ["app", "--file", str(cfg), "--apps-dir", str(apps)])

    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["name"] == "company_brain"
    assert out["title"] == "Northwind Brain"
    assert "finance@northwind.test" in [u["id"] for u in out["users"]]
    assert "find_experts" in out["tools"]
    assert out["connectors"]["local_files"]["path"].endswith("sample_corpus")

    cfg.write_text("app: missing\n", encoding="utf-8")
    result = runner.invoke(app, ["app", "--file", str(cfg), "--apps-dir", str(apps)])
    assert result.exit_code != 0
    assert "known apps: _template, company_brain" in result.output


def test_bad_profile_exits_nonzero_with_a_readable_error(tmp_path):
    result = runner.invoke(app, ["config", "--file", str(write_cfg(tmp_path)), "--profile", "x"])
    assert result.exit_code != 0
    assert "unknown profile 'x'" in result.output


def test_reset_asks_before_dropping_and_does_nothing_when_declined(tmp_path):
    result = runner.invoke(app, ["reset", "--file", str(write_cfg(tmp_path))], input="n\n")
    assert result.exit_code != 0
    assert "Drop" in result.output
    assert "Aborted" in result.output


def test_reset_reports_an_unreachable_database_readably(tmp_path, monkeypatch):
    monkeypatch.setenv("HACKSTACK_STORE__DSN", "postgresql://u:p@127.0.0.1:1/x?connect_timeout=2")
    result = runner.invoke(app, ["reset", "--yes", "--file", str(write_cfg(tmp_path))])
    assert result.exit_code == 2
    assert "error:" in result.output
    assert "make up" in result.output
