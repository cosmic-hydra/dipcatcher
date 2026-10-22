"""Flash-context entry schema and the append-only local store.

Storage layout: one JSONL file at ``<flash dir>/entries.jsonl`` where
``<flash dir>`` defaults to ``$FX1_CONFIG_DIR/flash`` (so tests isolate it
the same way the endpoint store is isolated — see ``fx1.interactive.profiles``).
Every write appends one line and never rewrites history: corrections are new
revisions (``updated_at`` advances), and removal writes a tombstone line, so the
store is trivially auditable and a torn last line can never corrupt earlier
entries.

Honesty is structural, not cosmetic:
- every finding carries ``sources`` (URL + title + accessed-at) or is
  explicitly unsourced;
- ``uncertainty`` and ``conflicts`` are first-class fields, never prose;
- ``private`` marks findings that must not leave the machine (or be folded
  into web-search queries) without explicit user approval;
- retrieval freshness is expressible via ``stale_after_s`` and checked at
  read time, and ``refresh`` re-verifies sources rather than guessing.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fx1.interactive.profiles import config_dir

ENTRY_SCHEMA = "fx1.flash-entry/v1"
FILE_NAME = "entries.jsonl"
_NO_DIRECTORY_FD = -1
_SUPPORTS_DIRECTORY_FD = os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd

Uncertainty = Literal["low", "medium", "high"]


class FlashSource(BaseModel):
    """One provenance record: where a finding came from."""

    model_config = ConfigDict(extra="ignore")

    url: str
    title: str = ""
    accessed_at: str = ""


class FlashEntry(BaseModel):
    """One saved research finding with full provenance and honesty fields."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = ENTRY_SCHEMA
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=lambda: _now_iso())
    updated_at: str = Field(default_factory=lambda: _now_iso())
    last_used_at: str = ""
    uses: int = 0
    task: str = ""
    tags: list[str] = Field(default_factory=list)
    text: str
    sources: list[FlashSource] = Field(default_factory=list)
    uncertainty: Uncertainty | None = None
    conflicts: list[str] = Field(default_factory=list)
    verified: bool = False
    private: bool = False
    stale_after_s: int | None = None
    evidence_at: str = ""
    refreshed_at: str = ""
    refresh_attempted_at: str = ""
    refresh_note: str = ""

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("flash entry text must not be empty")
        return value

    @field_validator("stale_after_s")
    @classmethod
    def _stale_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("stale_after_s must be positive when set")
        return value


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FlashStore:
    """Append-only JSONL store for flash entries.

    Thread-safe per instance (a module-level lock serializes appends across
    instances in one process). Reads tolerate a torn trailing line and report
    it; corrupt interior lines are skipped and surfaced via :meth:`read_errors`.
    """

    # A revision spans a read and two append operations (tombstone + value),
    # so append-only locking is not sufficient: another instance can read the
    # tombstone between those writes or calculate a revision from stale state.
    # Keep the complete replay/mutation transaction process-wide and reentrant
    # because public helpers call one another while holding the lock.
    _transaction_lock = threading.RLock()

    def __init__(self, path: Path | None = None) -> None:
        # ``resolve()`` follows an existing symlink and would permanently hide
        # that the configured store path was redirected. Keep the lexical
        # absolute path so every open can enforce no-follow semantics.
        self._path = (path or default_store_path()).expanduser().absolute()

    @property
    def path(self) -> Path:
        return self._path

    # -- low-level persistence ---------------------------------------------

    def _open_parent(self, *, create: bool) -> int | None:
        parent = self._path.parent
        if create:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            parent_stat = parent.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("flash store parent must be a real directory")
        if not _SUPPORTS_DIRECTORY_FD:
            if create and os.name == "posix":
                parent.chmod(0o700)
            return _NO_DIRECTORY_FD
        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(parent, flags)
        if create:
            os.fchmod(directory_fd, 0o700)
        return directory_fd

    def _path_stat(self, directory_fd: int) -> os.stat_result | None:
        try:
            if directory_fd == _NO_DIRECTORY_FD:
                return self._path.lstat()
            return os.stat(self._path.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def _open_path(self, directory_fd: int, flags: int, mode: int) -> int | None:
        try:
            if directory_fd == _NO_DIRECTORY_FD:
                return os.open(self._path, flags, mode)
            return os.open(self._path.name, flags, mode, dir_fd=directory_fd)
        except FileNotFoundError:
            return None

    def _open_file(self, *, write: bool) -> int | None:
        directory_fd = self._open_parent(create=write)
        if directory_fd is None:
            return None
        try:
            target_stat = self._path_stat(directory_fd)
            if target_stat is not None and not stat.S_ISREG(target_stat.st_mode):
                raise ValueError("flash store path must be a regular file")
            flags = os.O_RDONLY
            mode = 0
            if write:
                flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
                flags |= getattr(os, "O_NONBLOCK", 0)
                mode = 0o600
            flags |= getattr(os, "O_NOFOLLOW", 0)
            file_fd = self._open_path(directory_fd, flags, mode)
            if file_fd is None:
                return None
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                os.close(file_fd)
                raise ValueError("flash store path must be a regular file")
            if write:
                os.fchmod(file_fd, 0o600)
            return file_fd
        finally:
            if directory_fd != _NO_DIRECTORY_FD:
                os.close(directory_fd)

    def _load_raw(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Replay the append log in order: tombstones suppress the matching
        id, revisions replace earlier lines for the same id. Returns
        (current entries, parse-error descriptions)."""
        with FlashStore._transaction_lock:
            file_fd = self._open_file(write=False)
            if file_fd is None:
                return [], []
            by_id: dict[str, dict[str, Any]] = {}
            errors: list[str] = []
            with os.fdopen(file_fd, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f"line {lineno}: not JSON")
                        continue
                    if not isinstance(raw, dict):
                        errors.append(f"line {lineno}: not an object")
                        continue
                    if raw.get("__tombstone__") is True:
                        dead = raw.get("id")
                        if isinstance(dead, str):
                            by_id.pop(dead, None)
                        continue
                    entry_id = raw.get("id")
                    if isinstance(entry_id, str) and entry_id:
                        by_id[entry_id] = raw
            return list(by_id.values()), errors

    def _append_line(self, raw: dict[str, Any]) -> None:
        with FlashStore._transaction_lock:
            file_fd = self._open_file(write=True)
            if file_fd is None:  # pragma: no cover - O_CREAT makes this unreachable
                raise FileNotFoundError(self._path)
            with os.fdopen(file_fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(raw, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    # -- entries ------------------------------------------------------------

    def read_errors(self) -> list[str]:
        """Parse errors seen in the store file (honest reporting, never hidden)."""
        _, errors = self._load_raw()
        return errors

    def all(self) -> list[FlashEntry]:
        entries: list[FlashEntry] = []
        for raw in self._load_raw()[0]:
            try:
                entries.append(FlashEntry.model_validate(raw))
            except Exception:  # noqa: BLE001 — a bad line must not hide the rest
                continue
        return entries

    def get(self, entry_id: str) -> FlashEntry | None:
        for entry in self.all():
            if entry.id == entry_id:
                return entry
        return None

    def add(
        self,
        *,
        text: str,
        task: str = "",
        tags: list[str] | None = None,
        sources: list[FlashSource] | list[dict[str, str]] | None = None,
        uncertainty: Uncertainty | None = None,
        conflicts: list[str] | None = None,
        verified: bool = False,
        private: bool = False,
        stale_after_s: int | None = None,
    ) -> FlashEntry:
        entry = FlashEntry(
            text=text,
            task=task,
            tags=list(tags or []),
            sources=[FlashSource.model_validate(s) for s in (sources or [])],
            uncertainty=uncertainty,
            conflicts=list(conflicts or []),
            verified=verified,
            private=private,
            stale_after_s=stale_after_s,
        )
        self._append_line(entry.model_dump())
        return entry

    def update(self, entry_id: str, patch: dict[str, Any]) -> FlashEntry | None:
        """Correct an entry: append a new revision line and tombstone the old id.

        Only editable fields are accepted; identity and timestamps are managed
        by the store. Returns the revised entry or ``None`` when *entry_id*
        does not exist.
        """
        with FlashStore._transaction_lock:
            current = self.get(entry_id)
            if current is None:
                return None
            allowed = {
                "text",
                "task",
                "tags",
                "sources",
                "uncertainty",
                "conflicts",
                "verified",
                "private",
                "stale_after_s",
                "refreshed_at",
                "refresh_attempted_at",
                "refresh_note",
                # usage bookkeeping (written by mark_used); identity fields
                # (id / schema_version / created_at / updated_at) stay managed.
                "last_used_at",
                "uses",
            }
            unknown = set(patch) - allowed
            if unknown:
                raise ValueError(f"cannot correct fields: {sorted(unknown)}")
            merged = current.model_dump()
            for key, value in patch.items():
                merged[key] = value
            revision_at = _now_iso()
            merged["updated_at"] = revision_at
            # Only changes to the finding itself establish a new evidence
            # freshness anchor. Usage and refresh-attempt bookkeeping must not
            # make stale evidence appear current.
            evidence_fields = {"text", "sources", "uncertainty", "conflicts", "verified"}
            if evidence_fields & patch.keys():
                merged["evidence_at"] = revision_at
            revised = FlashEntry.model_validate(merged)
            self._append_line({"__tombstone__": True, "id": entry_id, "at": _now_iso()})
            self._append_line(revised.model_dump())
        return revised

    def remove(self, entry_id: str) -> bool:
        with FlashStore._transaction_lock:
            if self.get(entry_id) is None:
                return False
            self._append_line({"__tombstone__": True, "id": entry_id, "at": _now_iso()})
        return True

    def mark_used(self, entry_id: str) -> FlashEntry | None:
        with FlashStore._transaction_lock:
            current = self.get(entry_id)
            if current is None:
                return None
            return self.update(
                entry_id,
                {"last_used_at": _now_iso(), "uses": current.uses + 1},
            )

    def list(
        self,
        *,
        task: str | None = None,
        tags: list[str] | None = None,
        query: str | None = None,
    ) -> list[FlashEntry]:
        entries = self.all()
        if task is not None:
            needle = task.lower()
            entries = [e for e in entries if needle in e.task.lower()]
        if tags:
            wanted = {t.lower() for t in tags}
            entries = [e for e in entries if wanted & {t.lower() for t in e.tags}]
        if query is not None:
            from fx1.flash.retrieve import retrieve

            hits = retrieve(self, query, k=max(1, len(entries)), min_score=0.0)
            ids = {h.entry.id for h in hits}
            entries = [e for e in entries if e.id in ids]
        return entries


def default_store_path() -> Path:
    """``$FX1_CONFIG_DIR/flash/entries.jsonl`` — same config root as endpoints."""
    return config_dir() / "flash" / FILE_NAME


def default_store() -> FlashStore:
    return FlashStore()


def add(store: FlashStore, **fields: Any) -> FlashEntry:
    return store.add(**fields)


def get(store: FlashStore, entry_id: str) -> FlashEntry | None:
    return store.get(entry_id)


def list_entries(
    store: FlashStore, *, task: str | None = None, tags: list[str] | None = None
) -> list[FlashEntry]:
    return store.list(task=task, tags=tags)


def update(store: FlashStore, entry_id: str, patch: dict[str, Any]) -> FlashEntry | None:
    return store.update(entry_id, patch)


def remove(store: FlashStore, entry_id: str) -> bool:
    return store.remove(entry_id)
