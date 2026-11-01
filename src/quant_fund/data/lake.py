"""Parquet lake I/O."""

from __future__ import annotations

import os
import threading
from contextlib import suppress
from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig


def _directory_folds_case(directory: Path) -> bool:
    """True when ``directory`` treats different spellings of case as one file.

    ``os.path.normcase`` folds case on Windows and leaves POSIX unchanged.
    Default APFS is case-insensitive, so two POSIX lock keys would cover
    one inode and concurrent writers could race.
    """
    token = f".dipcatcher_case_{os.getpid()}_{threading.get_ident()}"
    probe = directory / token
    folded = directory / token.swapcase()
    try:
        probe.write_bytes(b"a")
    except OSError:
        return False
    try:
        return folded.exists()
    finally:
        with suppress(OSError):
            probe.unlink(missing_ok=True)


def _fsync_dir(path: Path) -> None:
    # Python's directory open/fsync sequence below is not supported on
    # Windows. The temporary file is flushed, but the subsequent rename is
    # not made power-loss durable by this helper on Windows.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Lake:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for part in ("raw", "bronze", "silver", "gold", "metadata"):
            (self.root / part).mkdir(parents=True, exist_ok=True)
        # Serialize replacement of the same destination within this Lake
        # instance, including normalized path aliases. Temporary files are
        # still private to each writer; locks do not coordinate processes
        # or separate Lake instances.
        self._rename_locks: dict[str, threading.Lock] = {}
        self._rename_locks_guard = threading.Lock()
        self._case_fold: bool | None = None

    def _folds_case(self) -> bool:
        cached = self._case_fold
        if cached is None:
            cached = _directory_folds_case(self.root)
            self._case_fold = cached
        return cached

    def _rename_lock(self, path: Path) -> threading.Lock:
        # Use the validated destination, not the caller's spelling. normcase
        # folds case on Windows. On a case-insensitive POSIX volume (APFS)
        # fold here too. A case-sensitive volume keeps distinct keys.
        # Do not resolve the final symlink: os.replace replaces that entry.
        key = os.path.normcase(os.path.abspath(path))
        if self._folds_case():
            key = key.casefold()
        with self._rename_locks_guard:
            lock = self._rename_locks.get(key)
            if lock is None:
                lock = self._rename_locks[key] = threading.Lock()
            return lock

    def _resolve(self, rel: str) -> Path:
        # Lake-relative paths only: no absolute paths, no traversal — a bad
        # caller must not be able to write outside the lake root. On Windows,
        # is_absolute() is False for drive-relative rooted paths such as
        # "/abs/x" or "C:x", so the drive/root checks are required as well.
        candidate = Path(rel)
        parts = candidate.parts
        if (
            not rel
            or candidate.is_absolute()
            or bool(candidate.drive)
            or bool(candidate.root)
            or "\\" in rel
            or any(part in (".", "..") for part in parts)
        ):
            raise ValueError(f"lake path must be a clean relative path, got {rel!r}")
        return self.root / candidate

    def write_parquet(self, frame: pl.DataFrame, rel: str) -> Path:
        """Publish a complete, flushed parquet via same-directory replacement.

        POSIX additionally fsyncs the destination directory. Windows skips
        that step, so successful return does not guarantee persistence of
        the rename across power loss. This is not a multi-process lock.
        """
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            frame.write_parquet(tmp)
            # Windows rejects fsync on a read-only handle (EBADF); open
            # read-write there so the flush still reaches the filesystem.
            fd = os.open(tmp, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            with self._rename_lock(path):
                os.replace(tmp, path)
            _fsync_dir(path.parent)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Memory-map a lake parquet file.

        ``columns`` projects before the frame is materialized. Omit it to read
        every stored column, which is what existing callers do.
        """
        return pl.read_parquet(self._resolve(rel), columns=columns, memory_map=True)

    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()


def lake_from_config(config: AppConfig) -> Lake:
    return Lake(Path(config.data.root))
