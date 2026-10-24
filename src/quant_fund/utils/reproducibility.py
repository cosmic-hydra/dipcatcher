"""Shared reproducibility fingerprints for research receipts and gates."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from quant_fund.utils.hashing import hash_bytes

# Files under ``filter=lfs`` are stored in this repo as ordinary blobs, not
# Git LFS pointers. ``git-lfs filter-process`` still rewrites them to pointers
# during ``git diff``. ``git diff HEAD --binary`` then emits the whole object
# (tens or hundreds of MB). Capturing that with a 5s timeout raises
# ``TimeoutExpired``, which this helper used to collapse to ``UNKNOWN``.
# A fresh ``actions/checkout`` often writes the worktree and the index in the
# same second, so the stat cache is racy and Git rechecks those files.
# Blank the LFS filter so byte-identical non-pointer blobs stay out of the
# diff. ``git diff --raw`` is not a substitute for the patch: an unstaged edit
# is reported with worktree blob id ``0000000``, so the raw line depends on
# the path and not the bytes. Hash those bytes instead.
_LFS_FILTER_OFF = (
    "-c",
    "filter.lfs.clean=",
    "-c",
    "filter.lfs.smudge=",
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.required=false",
)
_HASH_CHUNK = 1 << 20
_LFS_POINTER_MAX_BYTES = 1024
_CHECK_ATTR_CHUNK = 64


def git_worktree_sha256() -> str:
    """Hash tracked and relevant untracked changes in the current checkout."""
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    try:
        name_status = subprocess.run(
            [git, *_LFS_FILTER_OFF, "diff", "--name-status", "-z", "--no-renames", "HEAD"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        untracked = subprocess.run(
            [git, "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        ignored_prefixes = (
            ".venv/",
            ".pytest_cache/",
            "data/metadata/research/",
            "data/metadata/paper/",
            "htmlcov/",
        )
        entries: list[tuple[bytes, bytes]] = []
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8")
            if path.startswith(ignored_prefixes):
                continue
            file_path = Path(path)
            if file_path.is_file():
                entries.append((raw_path, file_path.read_bytes()))
        tracked = _tracked_changes(git, _parse_name_status(name_status or b""))
        return _fingerprint(tracked, _additions_blob(entries))
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


# One resolved HEAD per checkout. pytest-xdist workers share the index lock,
# so a later ``git rev-parse`` can time out after an earlier one succeeded.
# Caching the success keeps two seals of the same tree identical. A miss is
# not cached: the next call may still read the real revision.
_revision_lock = threading.Lock()
_revision_cache: dict[str, str] = {}


def clear_git_revision_cache() -> None:
    """Drop the process-local HEAD cache."""
    with _revision_lock:
        _revision_cache.clear()


def git_revision() -> str:
    """Return the checked-out revision, or an explicit unknown marker.

    A loaded runner can miss a short ``git rev-parse`` budget while the
    index lock is held. Retry, then reuse a resolved HEAD for this process
    so two seals of the same tree do not disagree by one call landing on
    ``UNKNOWN``.
    """
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    key = os.path.realpath(os.getcwd())
    with _revision_lock:
        cached = _revision_cache.get(key)
    if cached is not None:
        return cached
    delay = 0.05
    for attempt in range(6):
        try:
            rev = subprocess.run(
                [git, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            if attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
            continue
        if rev:
            with _revision_lock:
                _revision_cache.setdefault(key, rev)
                return _revision_cache[key]
    return "UNKNOWN"


def _parse_name_status(raw: bytes) -> list[tuple[bytes, bytes]]:
    """Split ``git diff --name-status -z`` into ``(status, path)`` pairs."""
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    entries: list[tuple[bytes, bytes]] = []
    index = 0
    while index + 1 < len(parts):
        status = parts[index]
        path = parts[index + 1]
        index += 2
        # ``--no-renames`` emits delete+add. Keep the destination if a
        # rename/copy record appears anyway (``R100\\0old\\0new\\0``).
        if status[:1] in (b"R", b"C") and index < len(parts):
            path = parts[index]
            index += 1
        entries.append((status, path))
    return entries


def _tracked_changes(git: str, entries: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Drop smudged LFS files whose bytes match the pointer recorded at HEAD.

    The fingerprint is ``git diff HEAD`` with LFS filters blanked. Git omits a
    path while the index stat cache matches, so a quiescent smudge never
    reaches this filter. A racy timestamp or a touch lists the path, because
    the worktree bytes are not the pointer blob. Matching the worktree sha256
    and size to the HEAD pointer's oid and size is the same comparison a clean
    filter would make, without running ``git-lfs``. The index pointer is the
    wrong object: a staged edit whose new pointer matches the worktree would
    look clean while ``git diff HEAD`` still shows the content change.
    """
    if not entries:
        return []
    candidates = [path for status, path in entries if status != b"D"]
    lfs_paths = _lfs_paths(git, candidates)
    pointers = _head_lfs_pointers(git, [path for path in candidates if path in lfs_paths])
    kept: list[tuple[bytes, bytes]] = []
    for status, path in entries:
        pointer = pointers.get(path)
        if pointer is not None and _worktree_matches_pointer(path, pointer):
            continue
        kept.append((status, path))
    return kept


def _lfs_paths(git: str, paths: list[bytes]) -> set[bytes]:
    marked: set[bytes] = set()
    for start in range(0, len(paths), _CHECK_ATTR_CHUNK):
        chunk = paths[start : start + _CHECK_ATTR_CHUNK]
        proc = subprocess.run(
            [git, "check-attr", "-z", "filter", "--", *(os.fsdecode(path) for path in chunk)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        parts = (proc.stdout or b"").split(b"\0")
        if parts and parts[-1] == b"":
            parts.pop()
        for index in range(0, len(parts) - 2, 3):
            path, attr, value = parts[index], parts[index + 1], parts[index + 2]
            if attr == b"filter" and value == b"lfs":
                marked.add(path)
    return marked


def _head_lfs_pointers(git: str, paths: list[bytes]) -> dict[bytes, tuple[str, int]]:
    """Return HEAD blobs that are Git LFS pointers, keyed by worktree path."""
    safe = [path for path in paths if b"\n" not in path]
    if not safe:
        return {}
    checked = subprocess.run(
        [git, *_LFS_FILTER_OFF, "cat-file", "--batch-check"],
        input=b"".join(b"HEAD:" + path + b"\n" for path in safe),
        check=True,
        capture_output=True,
        timeout=30,
    )
    small: list[bytes] = []
    for path, line in zip(safe, (checked.stdout or b"").splitlines(), strict=True):
        oid, kind, size_text = _batch_header(line)
        if kind != b"blob" or oid is None or size_text is None:
            continue
        try:
            size = int(size_text)
        except ValueError:
            continue
        if 0 <= size <= _LFS_POINTER_MAX_BYTES:
            small.append(path)
    if not small:
        return {}
    batched = subprocess.run(
        [git, *_LFS_FILTER_OFF, "cat-file", "--batch"],
        input=b"".join(b"HEAD:" + path + b"\n" for path in small),
        check=True,
        capture_output=True,
        timeout=30,
    )
    return _parse_pointer_batch(batched.stdout or b"", small)


def _batch_header(line: bytes) -> tuple[bytes | None, bytes | None, bytes | None]:
    parts = line.split()
    if len(parts) == 3 and parts[1] == b"blob":
        return parts[0], parts[1], parts[2]
    return None, None, None


def _parse_pointer_batch(stdout: bytes, paths: list[bytes]) -> dict[bytes, tuple[str, int]]:
    found: dict[bytes, tuple[str, int]] = {}
    offset = 0
    for path in paths:
        newline = stdout.find(b"\n", offset)
        if newline < 0:
            break
        _oid, kind, size_text = _batch_header(stdout[offset:newline])
        offset = newline + 1
        if kind != b"blob" or size_text is None:
            continue
        size = int(size_text)
        blob = stdout[offset : offset + size]
        offset += size
        if stdout[offset : offset + 1] == b"\n":
            offset += 1
        parsed = _parse_lfs_pointer(blob)
        if parsed is not None:
            found[path] = parsed
    return found


def _parse_lfs_pointer(blob: bytes) -> tuple[str, int] | None:
    if len(blob) > _LFS_POINTER_MAX_BYTES or b"\0" in blob:
        return None
    try:
        text = blob.decode("ascii")
    except UnicodeDecodeError:
        return None
    # Checked-out pointer files may carry CRLF endings (core.autocrlf).
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        return None
    oid: str | None = None
    size: int | None = None
    for line in lines[1:]:
        if not line:
            continue
        if line.startswith("oid sha256:"):
            candidate = line.removeprefix("oid sha256:")
            if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
                return None
            oid = candidate
        elif line.startswith("size "):
            raw_size = line.removeprefix("size ")
            if not raw_size.isdigit():
                return None
            size = int(raw_size)
    if oid is None or size is None:
        return None
    return oid, size


def _worktree_matches_pointer(raw_path: bytes, pointer: tuple[str, int]) -> bool:
    oid, size = pointer
    path = Path(os.fsdecode(raw_path))
    try:
        if not path.is_file() or path.stat().st_size != size:
            return False
    except OSError:
        return False
    return _file_sha256(path) == oid


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ContentAddress:
    """Real-file content identity, including a Git LFS pointer's object id.

    ``content_sha256`` is the hash of the file bytes. When the worktree file
    is an LFS pointer, that hash is the pointer's ``oid`` (the real object),
    not the hash of the pointer text. ``stored_sha256`` is always the hash of
    the bytes on disk, so a pointer and its smudged payload stay distinct.
    """

    content_sha256: str
    stored_sha256: str
    stored_size: int
    declared_size: int
    kind: str


def content_address(path: Path) -> ContentAddress:
    """Hash a file with the same chunked SHA-256 used for worktree fingerprints.

    A Git LFS pointer yields ``kind="lfs_pointer"`` and ``content_sha256``
    equal to the oid inside the pointer. Any other file yields ``kind="blob"``
    and both hashes equal ``_file_sha256``.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    stored_size = file_path.stat().st_size
    stored_sha256 = _file_sha256(file_path)
    if stored_size <= _LFS_POINTER_MAX_BYTES:
        parsed = _parse_lfs_pointer(file_path.read_bytes())
        if parsed is not None:
            oid, declared_size = parsed
            return ContentAddress(
                content_sha256=oid,
                stored_sha256=stored_sha256,
                stored_size=stored_size,
                declared_size=declared_size,
                kind="lfs_pointer",
            )
    return ContentAddress(
        content_sha256=stored_sha256,
        stored_sha256=stored_sha256,
        stored_size=stored_size,
        declared_size=stored_size,
        kind="blob",
    )


def _additions_blob(entries: list[tuple[bytes, bytes]]) -> bytes:
    """Length-frame untracked ``(path, bytes)`` pairs for the fingerprint.

    Plain ``path\\0bytes`` concatenation is not injective: file "a" holding
    ``x\\0b\\0y`` plus file "b" holding ``y`` produces the same blob as "a"=x,
    "b"=``y`` — a crafted tree could collide fingerprints. Framing each entry
    with its byte length (the same scheme tracked files use) removes it.
    Sorted by path so the blob is independent of ``ls-files`` ordering.
    """
    blob = bytearray()
    for raw_path, content in sorted(entries, key=lambda item: item[0]):
        blob.extend(raw_path)
        blob.extend(b"\0")
        blob.extend(str(len(content)).encode("ascii"))
        blob.extend(b"\0")
        blob.extend(content)
        blob.extend(b"\0")
    return bytes(blob)


def _fingerprint(tracked: list[tuple[bytes, bytes]], additions: bytes) -> str:
    """Hash path, status, and worktree bytes. An empty change set hashes ``b""``."""
    if not tracked:
        return hash_bytes(additions)
    tracked.sort(key=lambda item: (item[1], item[0]))
    hasher = hashlib.sha256()
    for status, raw_path in tracked:
        hasher.update(b"tracked\0")
        hasher.update(status)
        hasher.update(b"\0")
        hasher.update(raw_path)
        hasher.update(b"\0")
        path = Path(os.fsdecode(raw_path))
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if status == b"D" or not is_file:
            hasher.update(b"DELETED\0")
            continue
        hasher.update(str(path.stat().st_size).encode("ascii"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    if additions:
        hasher.update(additions)
    return hasher.hexdigest()
