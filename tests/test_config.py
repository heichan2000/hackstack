from pathlib import Path

import pytest

from hackstack.config import config_hash, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]

YAML = """
app: demo_app
llm:
  provider: anthropic
  answer_model: claude-sonnet-5-5
embed:
  provider: voyage
  model: voyage-4
  dim: 1024
rerank:
  top_n: 8
profiles:
  local:
    llm:
      provider: openai_compat
      answer_model: some-local-model
      base_url: http://localhost:11434/v1
    embed:
      provider: ollama
      model: qwen3-embedding:0.6b
"""


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "hackstack.yaml"
    path.write_text(YAML, encoding="utf-8")
    return path


def load(path, **kw):
    return load_settings(path, env_file=None, **kw)


def test_values_come_from_the_yaml_file(cfg_file):
    s = load(cfg_file)
    assert s.app == "demo_app"
    assert s.llm.provider == "anthropic"
    assert s.embed.model == "voyage-4"


def test_unset_knobs_fall_back_to_readme_defaults(cfg_file):
    s = load(cfg_file)
    assert s.chunk.size == 400
    assert s.rrf.k == 60
    assert s.retrieval.window == 50
    assert s.agent.max_iterations == 6
    assert s.enrich.contextual is False


def test_default_profile_is_the_base_config(cfg_file):
    s = load(cfg_file)
    assert s.profile == "default"
    assert s.embed.provider == "voyage"


def test_hackstack_profile_env_var_switches_profile(cfg_file, monkeypatch):
    monkeypatch.setenv("HACKSTACK_PROFILE", "local")
    s = load(cfg_file)
    assert s.profile == "local"
    assert s.llm.provider == "openai_compat"
    assert s.llm.base_url == "http://localhost:11434/v1"
    assert s.embed.provider == "ollama"


def test_profile_overlay_keeps_base_values_it_does_not_mention(cfg_file, monkeypatch):
    monkeypatch.setenv("HACKSTACK_PROFILE", "local")
    s = load(cfg_file)
    assert s.embed.dim == 1024
    assert s.app == "demo_app"


def test_explicit_profile_argument_wins_over_env(cfg_file, monkeypatch):
    monkeypatch.setenv("HACKSTACK_PROFILE", "local")
    assert load(cfg_file, profile="default").embed.provider == "voyage"


def test_unknown_profile_is_refused(cfg_file):
    with pytest.raises(ValueError, match=r"'nope'.*local"):
        load(cfg_file, profile="nope")


def test_env_var_overrides_a_nested_yaml_value(cfg_file, monkeypatch):
    monkeypatch.setenv("HACKSTACK_RERANK__TOP_N", "3")
    assert load(cfg_file).rerank.top_n == 3


def test_typo_in_a_section_is_refused(tmp_path):
    path = tmp_path / "hackstack.yaml"
    path.write_text("chunk:\n  sise: 300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sise"):
        load(path)


def test_unknown_top_level_key_is_refused(tmp_path):
    path = tmp_path / "hackstack.yaml"
    path.write_text("chunks:\n  size: 300\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunks"):
        load(path)


def test_config_hash_is_equal_for_equal_configs(cfg_file, tmp_path):
    copy = tmp_path / "copy.yaml"
    copy.write_text(YAML, encoding="utf-8")
    assert config_hash(load(cfg_file)) == config_hash(load(copy))


def test_config_hash_ignores_key_order(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("chunk: {size: 300, overlap: 20}\nrrf: {k: 40}\n", encoding="utf-8")
    b.write_text("rrf: {k: 40}\nchunk: {overlap: 20, size: 300}\n", encoding="utf-8")
    assert config_hash(load(a)) == config_hash(load(b))


def test_config_hash_changes_when_a_knob_changes(cfg_file, monkeypatch):
    before = config_hash(load(cfg_file))
    monkeypatch.setenv("HACKSTACK_RRF__K", "40")
    assert config_hash(load(cfg_file)) != before


def test_config_hash_is_a_short_hex_string(cfg_file):
    h = config_hash(load(cfg_file))
    assert len(h) == 16
    int(h, 16)


def test_repo_hackstack_yaml_loads_in_both_profiles():
    default = load(REPO_ROOT / "hackstack.yaml", profile="default")
    local = load(REPO_ROOT / "hackstack.yaml", profile="local")
    assert default.llm.provider == "anthropic"
    assert default.embed.model == "voyage-4"
    assert local.llm.provider == "openai_compat"
    assert local.embed.model == "qwen3-embedding:0.6b"
    assert config_hash(default) != config_hash(local)


def test_secrets_are_read_from_env_and_never_appear_in_settings(cfg_file, monkeypatch):
    from hackstack.config import load_secrets

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    secrets = load_secrets(env_file=None)
    assert secrets.anthropic_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in str(secrets)
    assert "sk-test-123" not in load(cfg_file).model_dump_json()


def test_missing_secrets_are_none(monkeypatch):
    from hackstack.config import load_secrets

    assert load_secrets(env_file=None).voyage_api_key is None


def test_blank_secret_lines_copied_from_env_example_count_as_missing(tmp_path):
    from hackstack.config import load_secrets

    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=\nVOYAGE_API_KEY=vk-1\n", encoding="utf-8")
    secrets = load_secrets(env_file=env)
    assert secrets.anthropic_api_key is None
    assert secrets.voyage_api_key.get_secret_value() == "vk-1"
