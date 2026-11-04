from pathlib import Path

import pytest

from hackstack.config import Secrets, load_settings
from hackstack.llm import make_llm
from hackstack.llm.anthropic_provider import AnthropicProvider
from hackstack.llm.openai_compat_provider import OpenAICompatProvider

REPO_YAML = Path(__file__).parents[1] / "hackstack.yaml"


def test_local_profile_builds_the_ollama_chat_model():
    settings = load_settings(REPO_YAML, profile="local", env_file=None)

    llm = make_llm(settings, Secrets(_env_file=None))

    assert isinstance(llm, OpenAICompatProvider)
    assert llm.model == "hackstack-chat"
    assert llm.base_url == "http://localhost:11434/v1"
    assert llm.reasoning_effort == "none"
    assert llm.api_key is None


def test_default_profile_builds_claude_for_answers_and_helpers():
    settings = load_settings(REPO_YAML, env_file=None)
    secrets = Secrets(_env_file=None, anthropic_api_key="sk-ant-test")

    answer = make_llm(settings, secrets)
    helper = make_llm(settings, secrets, aux=True)

    assert isinstance(answer, AnthropicProvider)
    assert (answer.model, helper.model) == ("claude-sonnet-5-5", "claude-haiku-5-5")


def test_openai_compat_passes_the_openai_key_when_set():
    settings = load_settings(REPO_YAML, profile="local", env_file=None)

    llm = make_llm(settings, Secrets(_env_file=None, openai_api_key="sk-oa"))

    assert llm.api_key == "sk-oa"


def test_anthropic_without_a_key_says_how_to_fix_it():
    settings = load_settings(REPO_YAML, env_file=None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY.*HACKSTACK_PROFILE=local"):
        make_llm(settings, Secrets(_env_file=None))
