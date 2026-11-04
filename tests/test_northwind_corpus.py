"""The Northwind demo corpus keeps its promises: every doc has an ACL, dates and owners, and the
planted demo moments listed in apps/company_brain/README.md are really there.

The metadata contract (front matter, then the nearest ``_defaults.yaml``) is read here directly so
the corpus is checked independently of the connector that ingests it.
"""

from datetime import date
from pathlib import Path

import pytest
import yaml

from hackstack.models import ACL, Principal

CORPUS = Path(__file__).resolve().parents[1] / "apps" / "company_brain" / "sample_corpus"
USERS = yaml.safe_load((CORPUS.parent / "users.yaml").read_text(encoding="utf-8"))["users"]
PERSONAS = {u["id"]: Principal(user=u["id"], groups=u["groups"]) for u in USERS}


def metadata(path: Path) -> dict:
    meta: dict = {}
    for folder in reversed([p for p in path.parents if p.is_relative_to(CORPUS)]):
        defaults = folder / "_defaults.yaml"
        if defaults.exists():
            meta.update(yaml.safe_load(defaults.read_text(encoding="utf-8")) or {})
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        meta.update(yaml.safe_load(text.split("---\n", 2)[1]) or {})
    return meta


DOCS = {p.relative_to(CORPUS).as_posix(): metadata(p) for p in sorted(CORPUS.rglob("*.md"))}
NORTHWIND = {k: v for k, v in DOCS.items() if k.startswith("northwind/")}


def visible_to(user: str) -> set[str]:
    who = PERSONAS[user]
    return {k for k, m in DOCS.items() if ACL(principals=m.get("acl", [])).allows(who)}


@pytest.mark.parametrize("doc", sorted(DOCS))
def test_every_document_has_an_acl_and_a_date(doc):
    meta = DOCS[doc]
    assert meta.get("acl"), f"{doc} has no ACL, so nobody could read it"
    assert isinstance(meta.get("updated_at"), date), f"{doc} has no updated_at date"


def test_northwind_docs_have_titles_and_owners():
    assert 8 <= len(NORTHWIND) <= 13
    for doc, meta in NORTHWIND.items():
        assert meta.get("title"), doc
        assert meta.get("owners") and all(o.endswith("@northwind.test") for o in meta["owners"]), (
            doc
        )


def test_hr_and_finance_pages_are_restricted_by_role():
    eng, hr, finance, ceo = (
        visible_to(f"{r}@northwind.test") for r in ("eng", "hr", "finance", "ceo")
    )
    comp = "northwind/hr/compensation-bands-2026.md"
    budget = "northwind/finance/q4-2026-budget.md"

    assert comp in hr and comp in ceo and comp not in eng and comp not in finance
    assert budget in finance and budget in ceo and budget not in eng and budget not in hr
    assert "northwind/handbook/pto-policy.md" in eng & hr & finance & ceo


def test_pto_contradiction_is_dated_and_visible_to_everyone():
    old = DOCS["northwind/handbook/pto-policy.md"]
    new = DOCS["northwind/announcements/2026-10-29-pto-carryover-change.md"]
    assert old["updated_at"] < new["updated_at"]
    assert old["updated_at"].month == 3
    for user in PERSONAS:
        assert {
            "northwind/handbook/pto-policy.md",
            "northwind/announcements/2026-10-29-pto-carryover-change.md",
        } <= visible_to(user)


def test_what_changed_since_monday_differs_by_role():
    monday = date(2026, 11, 2)
    changed = {k for k, m in DOCS.items() if m["updated_at"] >= monday}
    assert len(changed) >= 4
    # Only Northwind's planted updates are recent; the handbook slice is dated earlier.
    assert all(k.startswith("northwind/") for k in changed)
    assert changed & visible_to("hr@northwind.test") != changed & visible_to("eng@northwind.test")


def test_one_person_owns_the_database_migration_story():
    owned = [k for k, m in NORTHWIND.items() if "sam.okafor@northwind.test" in m["owners"]]
    assert len(owned) >= 2
