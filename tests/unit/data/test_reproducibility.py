"""Worktree fingerprint hashes bytes, including through Git LFS."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_worktree_sha256


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "reproducibility@example.com")
    _git(repo, "config", "user.name", "reproducibility")
    _git(repo, "config", "commit.gpgsign", "false")


def test_worktree_fingerprint_tracks_real_bytes_not_lfs_clean_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LFS clean filter must not turn an unmodified blob into UNKNOWN.

    ``git diff HEAD --binary`` on a non-pointer blob with ``filter=lfs`` emits
    the filtered object. After the stat cache is invalidated (``touch``, or a
    racy checkout), that capture exceeds the old 5s timeout on the large
    parquet/npy files and the fingerprint collapses to ``UNKNOWN``. Phase-1
    index verification then rejects ``git_worktree_sha256``.
    """
    lfs = shutil.which("git-lfs")
    if lfs is None:
        pytest.skip("git-lfs is required to reproduce the clean-filter false dirty")
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitattributes").write_text("payload.bin filter=lfs -text\n", encoding="utf-8")
    payload = repo / "payload.bin"
    payload.write_bytes(b"same-bytes")
    _git(repo, "add", "--", ".gitattributes")
    # Store the raw bytes. A normal ``git add`` would run the LFS clean filter
    # and persist a pointer, which is not what this checkout contains.
    _git(
        repo,
        "-c",
        "filter.lfs.clean=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        "add",
        "--",
        "payload.bin",
    )
    _git(repo, "commit", "-qm", "init")
    _git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    _git(repo, "config", "filter.lfs.required", "true")
    # Matching stat information skips the filter. Touching forces the recheck
    # that a same-second checkout leaves racy.
    payload.touch()
    false_dirty = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert false_dirty.stdout, "LFS clean filter did not false-dirty the blob"

    monkeypatch.chdir(repo)
    fingerprint = git_worktree_sha256()
    assert fingerprint == hash_bytes(b"")
    assert len(fingerprint) == 64

    payload.write_bytes(b"same-bytes!")
    changed = git_worktree_sha256()
    assert changed != fingerprint
    assert len(changed) == 64
    assert all(char in "0123456789abcdef" for char in changed)


def _repo_with_file(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    target = repo / "f"
    target.write_bytes(content)
    _git(repo, "add", "--", "f")
    _git(repo, "commit", "-qm", "init")
    return repo, target


def _raw_diff(repo: Path) -> bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "diff",
            "--raw",
            "--no-renames",
            "HEAD",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def test_clean_worktree_fingerprint_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    first = git_worktree_sha256()
    second = git_worktree_sha256()
    assert first == second == hash_bytes(b"")
    assert len(first) == 64


def test_distinct_unstaged_edits_have_distinct_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``git diff --raw`` prints worktree oid ``0000000`` for every unstaged edit.

    Editing A→B and A→C therefore produces the same raw line. The fingerprint
    has to hash the worktree bytes, including bytes past the first read chunk.
    """
    repo, target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    clean = git_worktree_sha256()

    target.write_bytes(b"B")
    raw_b = _raw_diff(repo)
    edited_b = git_worktree_sha256()
    target.write_bytes(b"C")
    raw_c = _raw_diff(repo)
    edited_c = git_worktree_sha256()

    assert b"0000000" in raw_b
    assert raw_b == raw_c
    assert edited_b != edited_c
    assert edited_b != clean
    assert edited_c != clean

    past_chunk_x = b"B" * (1 << 20) + b"X"
    past_chunk_y = b"B" * (1 << 20) + b"Y"
    target.write_bytes(past_chunk_x)
    digest_x = git_worktree_sha256()
    target.write_bytes(past_chunk_y)
    digest_y = git_worktree_sha256()
    assert digest_x != digest_y


def test_deleted_file_changes_worktree_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    clean = git_worktree_sha256()
    target.unlink()
    deleted = git_worktree_sha256()
    assert deleted != clean
    assert len(deleted) == 64


def _filter_off_name_status(repo: Path) -> bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            "HEAD",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def test_smudged_lfs_file_matching_pointer_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching LFS smudge stays clean, including after the stat cache breaks.

    Git trusts the index stat cache. A quiescent smudge is absent from a
    filter-off diff once that cache is refreshed, and whether a just-written
    file is listed depends on racy-git timestamps (Git 2.55's nanosecond index
    often is not racy). Touching the file lists it, because the worktree bytes
    are not the pointer blob. The fingerprint still treats a matching oid as
    clean.
    """
    if shutil.which("git-lfs") is None:
        pytest.skip("git-lfs is required to smudge a real pointer")
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    _git(repo, "config", "filter.lfs.required", "true")
    (repo / ".gitattributes").write_text("payload.bin filter=lfs -text\n", encoding="utf-8")
    payload = repo / "payload.bin"
    original = b"hello-lfs-content-not-a-pointer"
    payload.write_bytes(original)
    _git(repo, "add", "--", ".gitattributes", "payload.bin")
    _git(repo, "commit", "-qm", "init")
    pointer = _git(repo, "cat-file", "blob", "HEAD:payload.bin").stdout
    assert pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n")
    assert payload.read_bytes() == original
    # Backdate, then refresh with the clean filter still on, so the cached
    # stat matches and the index timestamp is strictly newer than the file.
    past = time.time() - 30
    os.utime(payload, (past, past))
    _git(repo, "update-index", "--refresh")
    assert b"payload.bin" not in _filter_off_name_status(repo)

    monkeypatch.chdir(repo)
    assert git_worktree_sha256() == hash_bytes(b"")
    payload.touch()
    assert b"payload.bin" in _filter_off_name_status(repo)
    assert git_worktree_sha256() == hash_bytes(b"")
    payload.write_bytes(original + b"!")
    assert git_worktree_sha256() != hash_bytes(b"")


def test_git_revision_reuses_a_resolved_head_when_later_calls_time_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two seals must agree when a later rev-parse loses the index lock."""
    import quant_fund.utils.reproducibility as repro

    repro.clear_git_revision_cache()
    calls = {"n": 0}
    real_run = subprocess.run

    def flaky_run(cmd: list[str] | tuple[str, ...], *args: object, **kwargs: object):
        if isinstance(cmd, (list, tuple)) and "rev-parse" in cmd and "HEAD" in cmd:
            calls["n"] += 1
            if calls["n"] > 1:
                raise subprocess.TimeoutExpired(cmd, float(kwargs.get("timeout", 15)))
        return real_run(cmd, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repro.subprocess, "run", flaky_run)
    try:
        first = repro.git_revision()
        second = repro.git_revision()
    finally:
        repro.clear_git_revision_cache()
    assert first != "UNKNOWN"
    assert second == first
    assert calls["n"] == 1


def test_git_revision_does_not_cache_an_unknown_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fund.utils.reproducibility as repro

    repro.clear_git_revision_cache()
    monkeypatch.setattr(repro.time, "sleep", lambda _delay: None)
    real_run = subprocess.run

    def timed_out(cmd: list[str] | tuple[str, ...], *args: object, **kwargs: object):
        raise subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(repro.subprocess, "run", timed_out)
    try:
        assert repro.git_revision() == "UNKNOWN"
        monkeypatch.setattr(repro.subprocess, "run", real_run)
        resolved = repro.git_revision()
    finally:
        repro.clear_git_revision_cache()
    assert resolved != "UNKNOWN"
    assert len(resolved) >= 7


def test_additions_blob_is_injective_under_path_content_repacking() -> None:
    """Untracked entries are length-framed: a file's bytes can no longer be
    reparsed as another file's ``path\\0`` header — the old concat scheme
    collided on crafted trees."""
    from quant_fund.utils.reproducibility import _additions_blob

    set_a = _additions_blob([(b"a", b"x\x00b\x00y"), (b"b", b"y")])
    set_b = _additions_blob([(b"a", b"x"), (b"b", b"y")])
    assert set_a != set_b
    # Order-insensitive: sorted by path inside the helper.
    assert _additions_blob([(b"z", b"1"), (b"a", b"2")]) == _additions_blob(
        [(b"a", b"2"), (b"z", b"1")]
    )
