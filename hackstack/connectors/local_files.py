"""Local files: Markdown, plain text, HTML, PDF and DOCX from a folder tree.

Config (``connectors.local_files``, from the app's ``app.yaml`` or ``hackstack.yaml``)::

    path: apps/company_brain/sample_corpus   # required
    default_acl: [group:everyone]            # optional; otherwise private to `owner`
    owner: sam@northwind.test                # optional; the ingesting user (default: OS user)
    max_items: 200                           # optional cap for demo backfills

Each file becomes one :class:`~hackstack.models.RawDoc` with ``doc_id`` ``local_files:<relative
posix path>``. Metadata comes from, highest precedence first, key by key:

1. **YAML front matter** at the top of a Markdown/text file (stripped from the content)::

       ---
       title: Travel policy
       owners: [priya@northwind.test]
       updated_at: 2026-10-30
       acl: [group:finance, user:sam@northwind.test]
       source_type: page
       ---

2. **A sidecar** next to the file, ``<file name>.meta.yaml`` (e.g. ``salaries.pdf.meta.yaml``),
   with the same keys. This is how binary files get metadata.
3. **Folder defaults**, ``_defaults.yaml``: the same keys, applying to the folder and everything
   below it. Nested ones cascade, and the nearest folder wins key by key.

This is the corpus metadata contract in ``apps/company_brain/README.md``. Files and folders whose
names start with ``_`` or ``.`` are configuration, never documents.

Recognised keys: ``title``, ``owners``, ``created_at``, ``updated_at``, ``url``,
``source_type``, ``acl``. Anything else lands in ``RawDoc.extra``. Without a ``title``, the first
heading names the document, else the file name. ``updated_at`` comes only from metadata, never
from the file's modification time, which a fresh checkout resets. The mtime is kept in
``extra["mtime"]``.

``acl`` entries are principal strings: ``user:<id>``, ``group:<name>`` or ``public``. A document
with no ``acl`` anywhere gets ``default_acl`` from config, else ``[user:<owner>]``: private to
whoever ingested it, so it fails closed.

A file that cannot be read, or whose metadata is malformed, is yielded as a
:class:`~hackstack.connectors.base.ConnectorFailure`, and the walk continues.
"""

from __future__ import annotations

import getpass
import re
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from hackstack.connectors.base import ConnectorFailure
from hackstack.models import ACL, RawDoc
from hackstack.registry import CONNECTOR, register

NAME = "local_files"
SIDECAR_SUFFIX = ".meta.yaml"
FOLDER_DEFAULTS = "_defaults.yaml"

MIME_BY_EXT = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
TEXT_MIMES = {"text/markdown", "text/plain", "text/html"}
FRONT_MATTER_MIMES = {"text/markdown", "text/plain"}
KNOWN_KEYS = {"title", "owners", "created_at", "updated_at", "url", "source_type", "acl"}

_FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


class MetadataError(ValueError):
    pass


@register(CONNECTOR, NAME)
class LocalFilesConnector:
    name = NAME
    #: ``extra`` keys the sync manager leaves out of the doc hash: a checkout resets mtimes.
    volatile_extra = frozenset({"mtime"})

    def __init__(self) -> None:
        self.root = Path(".")
        self.default_acl: list[str] | None = None
        self.owner: str | None = None
        self.max_items: int | None = None
        self._defaults: dict[Path, dict[str, Any]] = {}

    def configure(self, cfg: dict[str, Any], secrets: dict[str, Any]) -> None:
        if "path" not in cfg:
            raise ValueError("connectors.local_files.path is required")
        self.root = Path(cfg["path"]).expanduser()
        self.default_acl = list(cfg["default_acl"]) if cfg.get("default_acl") else None
        self.owner = cfg.get("owner")
        self.max_items = cfg.get("max_items")
        self._defaults.clear()

    def validate(self) -> None:
        if not self.root.is_dir():
            raise ValueError(f"local_files: folder not found: {self.root}")

    def load_all(self) -> Iterator[RawDoc | ConnectorFailure]:
        for path in self._capped_files():
            yield self._load_or_fail(path)

    def poll(
        self, since: datetime, until: datetime | None = None
    ) -> Iterator[RawDoc | ConnectorFailure]:
        """Files changed at or after ``since``: the file, its sidecar or any ``_defaults.yaml``
        above it. ``until`` is not enforced: mtimes come from a coarser clock than the cursor
        and can land just past it, and a file edited during the walk is better read now than
        next run (the sync manager's hash check makes re-reading free). A rename or a copy that
        keeps the old mtime is not seen here; the sync manager finds those via :meth:`list_ids`.
        """
        del until
        lo = since.timestamp()
        for path in self._capped_files():
            if self._changed_at(path) >= lo:
                yield self._load_or_fail(path)

    def list_ids(self) -> Iterator[str]:
        """Every document ``load_all`` would yield, without reading any of them."""
        for path in self._capped_files():
            yield self.doc_id(path)

    def doc_id(self, path: Path) -> str:
        return f"{NAME}:{path.relative_to(self.root).as_posix()}"

    # --- walking -----------------------------------------------------------------------------

    def _capped_files(self) -> Iterator[Path]:
        for emitted, path in enumerate(self._files()):
            if self.max_items is not None and emitted >= self.max_items:
                return
            yield path

    def _load_or_fail(self, path: Path) -> RawDoc | ConnectorFailure:
        doc_id = self.doc_id(path)
        try:
            return self._load(path, doc_id)
        except Exception as exc:  # one bad file must not crash the run
            return ConnectorFailure.from_exc(NAME, doc_id, exc)

    def _changed_at(self, path: Path) -> float:
        """Latest mtime among the file and everything its metadata comes from."""
        sources = [path, path.with_name(path.name + SIDECAR_SUFFIX)]
        folder = path.parent
        while True:
            sources.append(folder / FOLDER_DEFAULTS)
            if folder == self.root or self.root not in folder.parents:
                break
            folder = folder.parent
        return max(p.stat().st_mtime for p in sources if p.exists())

    def _files(self) -> Iterator[Path]:
        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root)
            if any(part.startswith((".", "_")) for part in rel.parts):
                continue
            if not path.is_file():
                continue
            if path.name.endswith(SIDECAR_SUFFIX):
                continue
            if path.suffix.lower() in MIME_BY_EXT:
                yield path

    def _load(self, path: Path, doc_id: str) -> RawDoc:
        mime = MIME_BY_EXT[path.suffix.lower()]
        data = path.read_bytes()
        content: str | bytes = data
        meta: dict[str, Any] = {}

        if mime in TEXT_MIMES:
            content = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
            if mime in FRONT_MATTER_MIMES:
                content, meta = _split_front_matter(content)

        sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
        side = _read_yaml(sidecar) if sidecar.exists() else {}
        # front matter beats the sidecar beats folder defaults, key by key
        merged = {**self._folder_defaults(path.parent), **side, **meta}
        acl = merged.get("acl")
        if acl is None:
            acl = self.default_acl or [f"user:{self.owner or getpass.getuser()}"]

        stat = path.stat()
        title = merged.get("title") or _title_from(content, mime) or _humanize(path.stem)
        rel = path.relative_to(self.root).as_posix()
        return RawDoc(
            doc_id=doc_id,
            source=NAME,
            source_type=merged.get("source_type") or "document",
            title=str(title),
            content=content,
            mime=mime,
            url=merged.get("url"),
            owners=_str_list(merged.get("owners"), "owners"),
            created_at=_when(merged.get("created_at")),
            updated_at=_when(merged.get("updated_at")),
            acl=ACL(principals=_str_list(acl, "acl")),
            extra={
                "path": rel,
                "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                **{k: v for k, v in merged.items() if k not in KNOWN_KEYS},
            },
        )

    def _folder_defaults(self, folder: Path) -> dict[str, Any]:
        """``_defaults.yaml`` merged from the root down to ``folder``; the nearest wins.

        A malformed one raises :class:`MetadataError`, so every document below it becomes a
        ``ConnectorFailure`` instead of silently losing its ACL.
        """
        if folder in self._defaults:
            return self._defaults[folder]
        merged: dict[str, Any] = {}
        if folder != self.root and self.root in folder.parents:
            merged.update(self._folder_defaults(folder.parent))
        path = folder / FOLDER_DEFAULTS
        if path.exists():
            merged.update(_read_yaml(path))
        self._defaults[folder] = merged
        return merged


# --- helpers ---------------------------------------------------------------------------------


def _split_front_matter(text: str) -> tuple[str, dict[str, Any]]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return text, {}
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise MetadataError(f"front matter: {exc}") from exc
    if not isinstance(meta, dict):
        raise MetadataError("front matter must be a YAML mapping")
    return text[m.end() :], meta


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise MetadataError(f"{path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise MetadataError(f"{path.name}: must be a YAML mapping")
    return data


def _title_from(content: str | bytes, mime: str) -> str | None:
    if mime != "text/markdown" or not isinstance(content, str):
        return None
    m = _HEADING.search(content)
    return m.group(1).strip() if m else None


def _humanize(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip()


def _str_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise MetadataError(f"{key} must be a string or a list of strings, got {value!r}")


def _when(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    raise MetadataError(f"not a date: {value!r}")
