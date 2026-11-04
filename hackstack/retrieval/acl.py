"""Who is asking: resolve a demo user to the principal strings the store filters on.

A user resolves to ``{user:<id>} ∪ {group:<g> for each group} ∪ {public}`` from the active app's
``users.yaml``. Those strings are matched against each chunk's ``acl_principals`` *inside* both
the dense and the lexical SQL (see ``store/postgres.py``), never by post-filtering in Python.

An unknown user is an error, not "public only": a typo in ``as_user`` should fail loudly rather
than quietly answer from a different permission set.
"""

from __future__ import annotations

from pathlib import Path

from hackstack.apps import DEFAULT_APPS_DIR, load_app
from hackstack.config import Settings
from hackstack.models import Principal

#: Readable by everyone, e.g. a public GitHub repo. Every resolved user carries it.
PUBLIC = "public"


class UnknownUserError(LookupError):
    """The active app's ``users.yaml`` has no such user."""


def principals_for(principal: Principal) -> list[str]:
    """``principal``'s user and group strings plus ``public``, in order, without repeats."""
    return list(dict.fromkeys([*principal.principals(), PUBLIC]))


def resolve_principals(
    settings: Settings, user_id: str, *, apps_dir: str | Path = DEFAULT_APPS_DIR
) -> list[str]:
    """The principal strings for ``user_id`` in the app that ``settings.app`` names."""
    app = load_app(settings, apps_dir=apps_dir)
    try:
        principal = app.principal(user_id)
    except KeyError as exc:
        raise UnknownUserError(exc.args[0] if exc.args else user_id) from exc
    return principals_for(principal)
