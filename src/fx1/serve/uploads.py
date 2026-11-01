"""Chunked uploads — the ``/v1/uploads`` contract.

Large inputs (BYOK eval datasets, fine-tune corpora) can exceed the
request body cap; the wire accepts them as uploads instead: create an
intent record, add up to ``UPLOAD_MAX_PARTS`` parts, then ``complete``
assembles the parts into a ``/v1/files`` record.

Durability rides the same hash-chained ``JobJournal`` + blob layout as
``_FileStore``:

- part blobs land in ``uploads/<upload_id>/<part_id>.bin``
  (tmp+rename+fsync) BEFORE the journaled part line — replay only ever
  names parts whose bytes are already durable;
- terminal transitions (``completed``/``cancelled``/``expired``) journal
  a tombstone; ``expired`` is a derived status (``created_at + ttl <
  now``) that journals itself on first touch;
- pending uploads evict LRU; their whole directory unlinks;
- blobs/dirs with no journaled record are GC'd on boot — a part id can
  never resurrect pointing at content that isn't there.

Without ``--state-dir`` parts live in a parallel in-memory map under the
same declared-bytes bound — the store is restart-lossy exactly like the
file store.

The store is transport-free: it raises ``UploadStoreError`` carrying the
HTTP status + wire code; the route layer maps it to ``ApiError`` and the
in-process SDK lets it propagate.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from fx1.serve.journal import JobJournal

UPLOAD_TTL_S = 3600
UPLOAD_MAX_PARTS = 64


class UploadStoreError(Exception):
    """Fail-closed upload fault — carries the HTTP status and wire code
    so the route layer can translate verbatim."""

    def __init__(self, status: int, message: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


def validate_upload_intent(purpose: str, filename: str, accept: frozenset[str]) -> None:
    """Fail-closed intent check — same purposes/extensions as a direct
    ``/v1/files`` upload; shared by the wire route and the in-process SDK."""
    if purpose not in accept:
        raise UploadStoreError(
            400,
            f"unsupported purpose {purpose!r} — only {sorted(accept)} are served",
            "invalid_request",
        )
    if not filename.endswith(".jsonl"):
        raise UploadStoreError(
            400, f"input must be a .jsonl file, got {filename!r}", "invalid_request"
        )


class UploadMeta(BaseModel):
    """One upload intent record — parts tracked as an ordered
    ``part_id -> byte length`` map (insertion order is arrival order;
    ``complete`` concatenates in the caller's ``part_ids`` order)."""

    model_config = ConfigDict(extra="forbid")

    upload_id: str
    filename: str
    purpose: str
    nbytes: int
    mime_type: str
    created_at: int
    expires_at: int
    status: str  # pending | completed | cancelled | expired
    file_id: str | None = None
    parts: dict[str, int] = {}


def upload_status(meta: UploadMeta, now: float | None = None) -> str:
    """``expired`` is derived — never stored as a guess."""
    if meta.status == "pending" and (now or time.time()) > meta.expires_at:
        return "expired"
    return meta.status


def upload_object(meta: UploadMeta, *, file_obj: dict[str, Any] | None = None) -> dict[str, Any]:
    """The OpenAI ``upload`` envelope."""
    return {
        "id": meta.upload_id,
        "object": "upload",
        "bytes": meta.nbytes,
        "created_at": meta.created_at,
        "expires_at": meta.expires_at,
        "filename": meta.filename,
        "mime_type": meta.mime_type,
        "purpose": meta.purpose,
        "status": upload_status(meta),
        "file": file_obj,
    }


def upload_part_object(part_id: str, upload_id: str, created_at: int) -> dict[str, Any]:
    return {
        "id": part_id,
        "object": "upload.part",
        "created_at": created_at,
        "upload_id": upload_id,
    }


class UploadStore:
    """Bounded LRU store of pending/terminal uploads (the ``/v1/uploads``
    surface).

    Bounded by entry count AND declared bytes — ``create`` refuses
    ``nbytes`` above the per-file cap, and parts can never exceed the
    declared total, so pending parts can pin at most
    ``max_entries * max_bytes``.
    """

    def __init__(
        self,
        max_entries: int,
        max_bytes: int,
        state_dir: Path | None = None,
        ttl_s: int = UPLOAD_TTL_S,
    ) -> None:
        self._lock = threading.Lock()
        self._max = max_entries
        self._max_bytes = max_bytes
        self._ttl_s = ttl_s
        self._uploads: OrderedDict[str, UploadMeta] = OrderedDict()
        # upload_id -> {part_id: bytes} when state_dir is None
        self._inmem: dict[str, dict[str, bytes]] = {}
        self._dir = Path(state_dir) / "uploads" if state_dir is not None else None
        self._journal = (
            JobJournal(Path(state_dir) / "uploads.jsonl") if state_dir is not None else None
        )
        self.recover_warnings: list[str] = []
        if self._journal is not None:
            res = self._journal.replay()
            self.recover_warnings = list(res.warnings)
            for payload in res.payloads:
                expired_ids = set(payload.get("expired_uploads") or ())
                for expired in expired_ids:
                    expired_meta = self._uploads.get(str(expired))
                    if expired_meta is not None:
                        expired_meta.status = "expired"
                        expired_meta.parts = {}
                for evict in payload.get("evicted") or ():
                    uid = str(evict)
                    if uid in expired_ids:
                        continue
                    self._uploads.pop(uid, None)
                    if self._dir is not None:
                        self._rmtree(self._dir / uid)
                if "upload" in payload:
                    meta = UploadMeta.model_validate(payload["upload"])
                    meta.parts = {}
                    self._uploads[meta.upload_id] = meta
                elif "upload_part" in payload:
                    p = payload["upload_part"]
                    part_meta = self._uploads.get(str(p["upload_id"]))
                    pid = str(p["part_id"])
                    if part_meta is None:
                        continue
                    blob = self._blob(str(part_meta.upload_id), pid)
                    if not blob.is_file():
                        self.recover_warnings.append(
                            f"upload {part_meta.upload_id}: part {pid} journaled "
                            "without its blob — dropped"
                        )
                        continue
                    part_meta.parts[pid] = int(p["bytes"])
                elif "upload_terminal" in payload:
                    t = payload["upload_terminal"]
                    term_meta = self._uploads.get(str(t["upload_id"]))
                    if term_meta is not None:
                        term_meta.status = str(t["status"])
                        term_meta.file_id = t.get("file_id")
                        if term_meta.status != "completed":
                            term_meta.parts = {}
            self._gc_blobs()
            self._compact_locked()

    # ---- blob layout ----------------------------------------------------

    def _blob(self, upload_id: str, part_id: str) -> Path:
        if self._dir is None:
            raise RuntimeError("upload store has no state_dir — nothing durable to address")
        return self._dir / upload_id / f"{part_id}.bin"

    @staticmethod
    def _rmtree(path: Path | None) -> None:
        if path is None or not path.is_dir():
            return
        for child in path.iterdir():
            child.unlink(missing_ok=True)
        path.rmdir()

    def _gc_blobs(self) -> None:
        """Drop on-disk part dirs with no journaled record, and part
        blobs whose ids aren't in the record's parts map."""
        if self._dir is None or not self._dir.is_dir():
            return
        for udir in self._dir.iterdir():
            if not udir.is_dir():
                continue
            meta = self._uploads.get(udir.name)
            if meta is None or meta.status != "pending":
                self._rmtree(udir)
                continue
            for blob in udir.iterdir():
                if blob.stem not in meta.parts:
                    blob.unlink(missing_ok=True)

    @staticmethod
    def _meta(meta: UploadMeta) -> dict[str, Any]:
        return meta.model_dump(mode="json", exclude={"parts"})

    def _drop_parts(self, meta: UploadMeta) -> None:
        meta.parts = {}
        self._inmem.pop(meta.upload_id, None)
        if self._dir is not None:
            self._rmtree(self._dir / meta.upload_id)

    def _compact_locked(self) -> None:
        if self._journal is not None:
            live: list[dict[str, Any]] = []
            for meta in self._uploads.values():
                live.append({"upload": self._meta(meta)})
                if meta.status == "pending":
                    for pid, nbytes in meta.parts.items():
                        live.append(
                            {
                                "upload_part": {
                                    "upload_id": meta.upload_id,
                                    "part_id": pid,
                                    "bytes": nbytes,
                                }
                            }
                        )
                elif meta.status == "completed":
                    live.append(
                        {
                            "upload_terminal": {
                                "upload_id": meta.upload_id,
                                "status": "completed",
                                "file_id": meta.file_id,
                            }
                        }
                    )
            self._journal.compact(live)

    # ---- lifecycle -------------------------------------------------------

    def create(
        self,
        *,
        purpose: str,
        filename: str,
        nbytes: int,
        mime_type: str,
    ) -> UploadMeta:
        if nbytes <= 0:
            raise UploadStoreError(400, "bytes must be positive", "invalid_request")
        if nbytes > self._max_bytes:
            raise UploadStoreError(
                413,
                f"upload exceeds the {self._max_bytes}-byte cap",
                "file_too_large",
            )
        now = int(time.time())
        meta = UploadMeta(
            upload_id=f"upload_{uuid.uuid4().hex}",
            filename=filename,
            purpose=purpose,
            nbytes=nbytes,
            mime_type=mime_type,
            created_at=now,
            expires_at=now + self._ttl_s,
            status="pending",
        )
        with self._lock:
            count = max(0, len(self._uploads) + 1 - self._max)
            evicted = list(self._uploads)[:count]
            if self._journal is not None:
                pending_evicted = [uid for uid in evicted if self._uploads[uid].status == "pending"]
                self._journal.append(
                    {
                        "upload": self._meta(meta),
                        "expired_uploads": pending_evicted,
                        # Older readers honor evicted, so a downgrade cannot
                        # resurrect a pending intent (they return 404, not 410).
                        "evicted": pending_evicted,
                    }
                )
            self._uploads[meta.upload_id] = meta
            self._inmem[meta.upload_id] = {}
            for uid in evicted:
                old = self._uploads.pop(uid)
                self._drop_parts(old)
        return meta

    def _touch(self, meta: UploadMeta) -> UploadMeta:
        """Lazy expiry — journals the tombstone once, unlinks parts."""
        if meta.status == "pending" and time.time() > meta.expires_at:
            if self._journal is not None:
                self._journal.append(
                    {"upload_terminal": {"upload_id": meta.upload_id, "status": "expired"}}
                )
            meta.status = "expired"
            self._drop_parts(meta)
        return meta

    def get(self, upload_id: str) -> UploadMeta | None:
        with self._lock:
            meta = self._uploads.get(upload_id)
            if meta is None:
                return None
            return self._touch(meta)

    def _pending(self, upload_id: str) -> UploadMeta:
        meta = self._uploads.get(upload_id)
        if meta is None:
            raise UploadStoreError(404, f"upload {upload_id!r} not found", "upload_not_found")
        self._touch(meta)
        if meta.status == "expired":
            raise UploadStoreError(410, f"upload {upload_id!r} expired", "upload_expired")
        if meta.status != "pending":
            raise UploadStoreError(409, f"upload {upload_id!r} is {meta.status}", "upload_terminal")
        return meta

    def add_part(self, upload_id: str, data: bytes) -> dict[str, Any]:
        if not data:
            raise UploadStoreError(400, "part data is empty", "invalid_request")
        if len(data) > self._max_bytes:
            raise UploadStoreError(
                413, f"part exceeds the {self._max_bytes}-byte cap", "file_too_large"
            )
        part_id = f"part_{uuid.uuid4().hex}"
        with self._lock:
            meta = self._pending(upload_id)
            if len(meta.parts) >= UPLOAD_MAX_PARTS:
                raise UploadStoreError(
                    400,
                    f"upload {upload_id!r} exceeds the {UPLOAD_MAX_PARTS}-part cap",
                    "too_many_parts",
                )
            if sum(meta.parts.values()) + len(data) > meta.nbytes:
                raise UploadStoreError(
                    400,
                    "cumulative part bytes exceed the declared upload bytes",
                    "part_exceeds_declared_bytes",
                )
            blob: Path | None = None
            tmp: Path | None = None
            journal_size: int | None = None
            if self._journal is not None:
                try:
                    journal_size = self._journal.path.stat().st_size
                except FileNotFoundError:
                    journal_size = 0
                except OSError:
                    # If the pre-write size cannot be observed, cleanup cannot
                    # prove that a failed append left the chain unchanged.
                    journal_size = None
            try:
                if self._dir is not None:
                    # Blob first, fsync'd — the journal may only name durable parts.
                    udir = self._dir / meta.upload_id
                    udir.mkdir(parents=True, exist_ok=True)
                    blob = self._blob(meta.upload_id, part_id)
                    tmp = udir / f".{part_id}.tmp"
                    with tmp.open("wb") as fh:
                        fh.write(data)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, blob)
                created_at = int(time.time())
                if self._journal is not None:
                    self._journal.append(
                        {
                            "upload_part": {
                                "upload_id": meta.upload_id,
                                "part_id": part_id,
                                "bytes": len(data),
                            }
                        }
                    )
                if self._dir is None:
                    self._inmem[meta.upload_id][part_id] = bytes(data)
                meta.parts[part_id] = len(data)
            except Exception:
                if tmp is not None:
                    with suppress(OSError):
                        tmp.unlink(missing_ok=True)
                # A definitely pre-write failure leaves an unreferenced blob;
                # remove it immediately rather than leaking disk until restart.
                # If the journal grew (including a torn append), retain the
                # blob as recovery evidence because a durable record may name it.
                unchanged = self._journal is None
                if self._journal is not None and journal_size is not None:
                    try:
                        unchanged = self._journal.path.stat().st_size == journal_size
                    except OSError:
                        unchanged = False
                if blob is not None and unchanged:
                    with suppress(OSError):
                        blob.unlink(missing_ok=True)
                raise
        return upload_part_object(part_id, upload_id, created_at)

    def _assemble_locked(self, meta: UploadMeta, part_ids: list[str]) -> bytes:
        if not part_ids:
            raise UploadStoreError(400, "part_ids is empty", "invalid_request")
        missing = [p for p in part_ids if p not in meta.parts]
        if missing:
            raise UploadStoreError(400, f"parts not found: {missing!r}", "part_not_found")
        out = bytearray()
        for pid in part_ids:
            if self._dir is not None:
                blob = self._blob(meta.upload_id, pid)
                if not blob.is_file():
                    raise UploadStoreError(
                        400, f"part {pid!r} content is missing", "part_not_found"
                    )
                out += blob.read_bytes()
            else:
                out += self._inmem[meta.upload_id][pid]
        return bytes(out)

    def assemble(self, upload_id: str, part_ids: list[str]) -> bytes:
        """Read the caller's part order into one payload."""
        with self._lock:
            return self._assemble_locked(self._pending(upload_id), part_ids)

    @staticmethod
    def _validate_complete(
        meta: UploadMeta, part_ids: list[str], content: bytes, md5: str | None
    ) -> None:
        declared = sum(meta.parts[p] for p in part_ids)
        if declared != meta.nbytes or len(content) != meta.nbytes:
            raise UploadStoreError(
                400,
                f"assembled bytes {len(content)} != declared {meta.nbytes}",
                "upload_incomplete",
            )
        if md5 is not None and (
            hashlib.md5(content, usedforsecurity=False).hexdigest() != md5.lower()
        ):
            raise UploadStoreError(400, "md5 mismatch", "checksum_mismatch")

    def _complete_locked(self, meta: UploadMeta, file_id: str) -> None:
        if self._journal is not None:
            self._journal.append(
                {
                    "upload_terminal": {
                        "upload_id": meta.upload_id,
                        "status": "completed",
                        "file_id": file_id,
                    }
                }
            )
        meta.status = "completed"
        meta.file_id = file_id
        self._drop_parts(meta)

    def complete(
        self,
        upload_id: str,
        part_ids: list[str],
        *,
        content: bytes,
        file_id: str,
        md5: str | None = None,
    ) -> UploadMeta:
        """Validate and record an already-minted file as terminal.

        Callers that mint a file should use ``complete_with`` to serialize
        publication with cancellation, expiry, eviction and other completes.
        """
        with self._lock:
            meta = self._pending(upload_id)
            self._validate_complete(meta, part_ids, content, md5)
            self._complete_locked(meta, file_id)
        return meta

    def complete_with[ResultT](
        self,
        upload_id: str,
        part_ids: list[str],
        *,
        publish: Callable[[UploadMeta, bytes], tuple[str, ResultT]],
        rollback: Callable[[str], None],
        md5: str | None = None,
    ) -> tuple[UploadMeta, ResultT]:
        """Validate, publish and complete while owning the lifecycle lock.

        Callbacks must not re-enter this store. A pre-write upload-journal
        failure keeps the upload pending and rolls back the newly minted
        file. If a failed append changed the journal, retain the file for
        recovery: its terminal reference may already be durable. Separate
        file/upload journals are not crash-atomic, and rollback does not
        restore file-store capacity evictions.
        """
        with self._lock:
            meta = self._pending(upload_id)
            content = self._assemble_locked(meta, part_ids)
            if md5 is not None and (
                hashlib.md5(content, usedforsecurity=False).hexdigest() != md5.lower()
            ):
                raise UploadStoreError(400, "md5 mismatch", "checksum_mismatch")
            self._validate_complete(meta, part_ids, content, None)
            journal_size = 0
            if self._journal is not None:
                with suppress(FileNotFoundError):
                    journal_size = self._journal.path.stat().st_size
            file_id, result = publish(meta, content)
            try:
                self._complete_locked(meta, file_id)
            except Exception:
                # A cleanup failure after a successful terminal append must
                # not delete the file named by the committed upload.
                unchanged = self._journal is None
                if self._journal is not None:
                    with suppress(OSError):
                        unchanged = self._journal.path.stat().st_size == journal_size
                if meta.status == "pending" and unchanged:
                    rollback(file_id)
                raise
            return meta, result

    def cancel(self, upload_id: str) -> UploadMeta:
        """Cancel a pending upload. Completed uploads stay terminal
        (409); cancel on an already-cancelled upload replays 200; expired
        uploads cancel to ``expired`` (the honest terminal state)."""
        with self._lock:
            meta = self._uploads.get(upload_id)
            if meta is None:
                raise UploadStoreError(404, f"upload {upload_id!r} not found", "upload_not_found")
            self._touch(meta)
            if meta.status == "completed":
                raise UploadStoreError(409, f"upload {upload_id!r} is completed", "upload_terminal")
            if meta.status in ("cancelled", "expired"):
                return meta
            if self._journal is not None:
                self._journal.append(
                    {"upload_terminal": {"upload_id": upload_id, "status": "cancelled"}}
                )
            meta.status = "cancelled"
            self._drop_parts(meta)
        return meta
