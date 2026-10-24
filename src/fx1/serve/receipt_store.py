"""Content-addressed index over the sealed-receipts store.

Shared by the HTTP API (``GET /receipts*``) and the in-process SDK — one
staleness/availability contract on both surfaces. This is a metadata index,
not cryptographic receipt verification.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import stat
import threading
from pathlib import Path

SHA256_HEX = re.compile(r"[0-9a-f]{64}")
# Content digest is part of the stamp. Windows FindFirstFile stat leaves
# st_ino at 0, and os.utime can make creation time match mtime, so a
# same-size replace that keeps the nanosecond mtime is invisible to
# dev/ino/size/mtime/ctime alone.
_FileStamp = tuple[int, int, int, int, int, bytes]
_Snapshot = dict[Path, _FileStamp]


def _content_digest(path: str, size: int) -> bytes:
    """Hash file bytes without Path.read_bytes (callers track that for parses)."""
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        remaining = size
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 65536))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    return digest.digest()


def _receipt_sha(path: Path) -> str | None:
    """Read one regular file, ignoring invalid input but not programming errors."""
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None
    sha = doc.get("receipt_sha256") if isinstance(doc, dict) else None
    return sha if isinstance(sha, str) and SHA256_HEX.fullmatch(sha) else None


class ReceiptIndex:
    """Lazy sha256 → file index over a receipts directory.

    Scan file identities and nanosecond metadata to detect additions,
    removals, renames and ordinary replacements, even when count and maximum
    mtime stay unchanged. Decode only new or changed files and sort only
    changed inventories. A missing or
    unreadable directory clears the cache instead of serving stale lookups.
    Symlinks and non-regular files are not receipt blobs. Duplicate digest
    declarations resolve to the lexicographically first filename.

    Metadata caching relies on the immutable-receipt contract; it is not
    an integrity check against a writer capable of preserving all metadata.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._key: _Snapshot | None = None
        self._cache: dict[Path, tuple[_FileStamp, str | None]] = {}
        self._by_sha: dict[str, Path] = {}
        self._items: tuple[tuple[str, Path], ...] = ()
        self._available = False

    @property
    def root(self) -> Path:
        return self._root

    def _snapshot(self) -> _Snapshot:
        files: _Snapshot = {}
        # scandir propagates directory access errors; glob can suppress them.
        with os.scandir(self._root) as entries:
            for entry in entries:
                if not fnmatch.fnmatch(entry.name, "*.json"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                try:
                    content = _content_digest(entry.path, metadata.st_size)
                except OSError:
                    content = b"\x00unreadable"
                stamp = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    content,
                )
                files[self._root / entry.name] = stamp
        return files

    def _scan(self) -> None:
        try:
            key = self._snapshot()
        except OSError:
            self._available = False
            self._by_sha = {}
            self._items = ()
            self._cache = {}
            self._key = None
            return
        if key == self._key:
            return
        cache: dict[Path, tuple[_FileStamp, str | None]] = {}
        by_sha: dict[str, Path] = {}
        for path, stamp in sorted(key.items(), key=lambda item: item[0].name):
            previous = self._cache.get(path)
            if previous is not None and previous[0] == stamp:
                sha = previous[1]
            else:
                sha = _receipt_sha(path)
            cache[path] = (stamp, sha)
            if sha is not None:
                by_sha.setdefault(sha, path)
        # Publish together under the caller's lock after the entire scan.
        self._cache = cache
        self._by_sha = by_sha
        self._items = tuple(by_sha.items())
        self._key = key
        self._available = True

    def available(self) -> bool:
        with self._lock:
            self._scan()
            return self._available

    def lookup(self, sha256: str) -> Path | None:
        with self._lock:
            self._scan()
            return self._by_sha.get(sha256)

    def items(self) -> list[tuple[str, Path]]:
        with self._lock:
            self._scan()
            return list(self._items)
