import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Tests never see the developer's HACKSTACK_* variables, API keys or ./.env file."""
    for key in list(os.environ):
        if key.startswith("HACKSTACK_") or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


# --- Postgres (integration tests marked ``pg``) ------------------------------------------------

PG_TEST_DB = "hackstack_test"


def _base_pg_dsn() -> str:
    # Not HACKSTACK_-prefixed on purpose: _isolated_env clears those for every test.
    port = os.environ.get("HACKSTACK_PG_PORT", "5433")
    return os.environ.get(
        "TEST_PG_DSN", f"postgresql://hackstack:hackstack@localhost:{port}/hackstack"
    )


@pytest.fixture(scope="session")
def pg_dsn():
    """DSN of a throwaway ``hackstack_test`` database on the Compose Postgres; skips if absent."""
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    base = _base_pg_dsn()
    try:
        admin = psycopg.connect(base, connect_timeout=2, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {base} (run `make up`): {exc}".splitlines()[0])
    with admin:
        admin.execute(f"DROP DATABASE IF EXISTS {PG_TEST_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {PG_TEST_DB}")
    params = conninfo_to_dict(base)
    params["dbname"] = PG_TEST_DB
    yield make_conninfo(**params)
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {PG_TEST_DB} WITH (FORCE)")
