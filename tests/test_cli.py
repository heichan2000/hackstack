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


def test_bad_profile_exits_nonzero_with_a_readable_error(tmp_path):
    result = runner.invoke(app, ["config", "--file", str(write_cfg(tmp_path)), "--profile", "x"])
    assert result.exit_code != 0
    assert "unknown profile 'x'" in result.output
