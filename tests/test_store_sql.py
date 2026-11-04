"""Offline unit tests for the Postgres store's SQL building blocks (no database needed)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from hackstack.models import ACL, RawDoc
from hackstack.store.postgres import (
    effective_principals,
    filter_clause,
    render_schema,
    vector_literal,
)


def test_schema_is_rendered_with_the_configured_dimension():
    sql = render_schema(768)
    assert "vector(768)" in sql
    assert "__DIM__" not in sql
    assert "hnsw" in sql and "acl_principals" in sql and "collection_meta" in sql


@pytest.mark.parametrize("bad", [0, -1, "1024; drop table x"])
def test_schema_refuses_a_bad_dimension(bad):
    with pytest.raises(ValueError):
        render_schema(bad)


def test_vector_literal():
    assert vector_literal([1, 0.5, -2]) == "[1.0,0.5,-2.0]"


def test_no_filters_gives_only_the_acl_clause():
    sql, params = filter_clause(["public"], None)
    assert sql == "c.acl_principals && %(principals)s::text[]"
    assert params == {"principals": ["public"]}


def test_source_and_since_filters():
    sql, params = filter_clause(["public"], {"source": "slack", "since": "2026-11-01"})
    assert "c.source = ANY(%(sources)s)" in sql
    assert "c.updated_at >= %(since)s" in sql
    assert params["sources"] == ["slack"]
    assert params["since"] == datetime(2026, 11, 1, tzinfo=UTC)


def test_since_accepts_dates_and_aware_datetimes():
    _, p = filter_clause([], {"since": date(2026, 1, 2)})
    assert p["since"] == datetime(2026, 1, 2, tzinfo=UTC)
    aware = datetime(2026, 1, 2, 3, tzinfo=UTC)
    _, p = filter_clause([], {"since": aware})
    assert p["since"] == aware


def test_none_valued_filters_are_ignored():
    sql, _ = filter_clause(["public"], {"source": None, "since": None})
    assert sql == "c.acl_principals && %(principals)s::text[]"


def test_unknown_filter_keys_fail_loudly():
    with pytest.raises(ValueError, match="sourec"):
        filter_clause(["public"], {"sourec": "slack"})


def test_principals_are_always_bound_as_a_list():
    _, params = filter_clause(("user:a", "public"), None)
    assert params["principals"] == ["user:a", "public"]


def raw(principals):
    return RawDoc(doc_id="d", source="s", title="t", content="c", acl=ACL(principals=principals))


def test_effective_principals_keeps_a_real_acl():
    assert effective_principals(raw(["group:eng"]), ingest_user="me@x") == ["group:eng"]


def test_effective_principals_makes_an_empty_acl_private_to_the_ingesting_user():
    assert effective_principals(raw([]), ingest_user="me@x") == ["user:me@x"]


def test_effective_principals_without_an_ingesting_user_is_nobody():
    assert effective_principals(raw([]), ingest_user=None) == []
