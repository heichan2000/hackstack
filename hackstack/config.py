"""Settings: ``hackstack.yaml`` (non-secret) + environment / ``.env`` (overrides and secrets).

Precedence, highest first:

1. ``HACKSTACK_*`` environment variables, nested with ``__`` (``HACKSTACK_RERANK__TOP_N=3``)
2. the same variables in ``.env``
3. the active profile's overlay in ``hackstack.yaml`` (``profiles.<name>``)
4. the base of ``hackstack.yaml``
5. the defaults below (they match the README's settings table)

The profile is picked by the ``profile=`` argument, else ``HACKSTACK_PROFILE`` (env, then
``.env``), else ``default``. ``default`` is the base config (Claude + Voyage); ``local`` overlays
Ollama. Secrets never live in :class:`Settings`; they are in :class:`Secrets`, so they never
reach :func:`config_hash`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_PROFILE = "default"
CONFIG_ENV_VAR = "HACKSTACK_CONFIG"
PROFILE_ENV_VAR = "HACKSTACK_PROFILE"


class _Section(BaseModel):
    """A config section. Unknown keys are refused so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class LLMSettings(_Section):
    provider: str = "anthropic"  # registry name: anthropic | openai_compat
    answer_model: str = "claude-sonnet-5-5"
    aux_model: str = "claude-haiku-5-5"  # helper tasks: enrichment, judges, query rewrite
    base_url: str | None = None  # OpenAI-compatible endpoints (Ollama, vLLM, Groq, ...)


class EmbedSettings(_Section):
    provider: str = "voyage"  # registry name: voyage | ollama | openai_compat
    model: str = "voyage-4"
    dim: int = 1024
    base_url: str | None = None


class StoreSettings(_Section):
    provider: str = "postgres"
    dsn: str = "postgresql://hackstack:hackstack@localhost:5433/hackstack"


class ChunkSettings(_Section):
    size: int = 400  # tokens
    overlap: int = 0  # tokens; the research doc's range is 0-50


class EnrichSettings(_Section):
    contextual: bool = False


class RetrievalSettings(_Section):
    window: int = 50  # candidates per retriever before fusion


class RRFSettings(_Section):
    k: int = 60
    weights: list[float] = Field(default_factory=lambda: [1.0, 1.0])  # dense, lexical


class RerankSettings(_Section):
    provider: str = "voyage"  # registry name: voyage | cross_encoder | noop
    model: str = "rerank-3"
    top_n: int = 8


class AgentSettings(_Section):
    max_iterations: int = 6
    mode: Literal["agent", "single_shot"] = "agent"


class TracingSettings(_Section):
    enabled: bool = False
    endpoint: str = "http://localhost:6006/v1/traces"
    capture_content: bool = False  # true only locally: spans would carry PII


class Settings(BaseSettings):
    """Every non-secret setting. Build it with :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="HACKSTACK_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",  # .env also holds secrets and unrelated variables
    )

    app: str = "company_brain"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    enrich: EnrichSettings = Field(default_factory=EnrichSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    rrf: RRFSettings = Field(default_factory=RRFSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    connectors: dict[str, dict[str, Any]] = Field(default_factory=dict)

    _profile: str = PrivateAttr(DEFAULT_PROFILE)

    @property
    def profile(self) -> str:
        """Name of the profile this config was loaded with (not part of the hash)."""
        return self._profile

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # The YAML (passed as init kwargs by load_settings) is the lowest-priority source.
        return env_settings, dotenv_settings, init_settings


class Secrets(BaseSettings):
    """API keys and connector tokens, from the environment or ``.env``. See ``.env.example``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    anthropic_api_key: SecretStr | None = None
    voyage_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    notion_token: SecretStr | None = None
    slack_bot_token: SecretStr | None = None
    google_client_secrets_file: str | None = None
    google_token_file: str | None = None


def load_settings(
    path: str | Path | None = None,
    *,
    profile: str | None = None,
    env_file: str | Path | None = ".env",
) -> Settings:
    """Load ``hackstack.yaml`` with the active profile applied, then env overrides."""
    data = _read_yaml(path)
    profiles = data.pop("profiles", None) or {}
    _refuse_unknown_keys(data, where="hackstack.yaml")

    name = profile or _profile_from_env(env_file) or DEFAULT_PROFILE
    if name in profiles:
        overlay = profiles[name] or {}
        _refuse_unknown_keys(overlay, where=f"profiles.{name}")
        data = _deep_merge(data, overlay)
    elif name != DEFAULT_PROFILE:
        known = ", ".join(sorted({DEFAULT_PROFILE, *profiles}))
        raise ValueError(f"unknown profile {name!r}; known profiles: {known}")

    settings = Settings(_env_file=env_file, **data)
    settings._profile = name
    return settings


def load_secrets(env_file: str | Path | None = ".env") -> Secrets:
    return Secrets(_env_file=env_file)


def config_hash(settings: Settings) -> str:
    """Stable 16-hex-char fingerprint of every setting; equal configs hash equal."""
    canonical = json.dumps(settings.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _read_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        candidate = Path(os.environ.get(CONFIG_ENV_VAR, "hackstack.yaml"))
        if not candidate.exists():
            return {}
        path = candidate
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _refuse_unknown_keys(data: dict[str, Any], *, where: str) -> None:
    unknown = sorted(set(data) - set(Settings.model_fields))
    if unknown:
        known = ", ".join(sorted(Settings.model_fields))
        raise ValueError(f"{where}: unknown setting(s) {', '.join(unknown)}; known: {known}")


def _profile_from_env(env_file: str | Path | None) -> str | None:
    if value := os.environ.get(PROFILE_ENV_VAR):
        return value
    if env_file is not None and Path(env_file).exists():
        return dotenv_values(env_file).get(PROFILE_ENV_VAR) or None
    return None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
