"""Vector + full-text stores. Every search is ACL-filtered inside the store."""

from hackstack.store import postgres as _postgres  # noqa: E402,F401  registers "postgres"
