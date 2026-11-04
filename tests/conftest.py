import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Tests never see the developer's HACKSTACK_* variables, API keys or ./.env file."""
    for key in list(os.environ):
        if key.startswith("HACKSTACK_") or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
