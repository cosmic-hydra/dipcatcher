"""SYNTHETIC release-integrity, bounded-I/O, and resource-lifetime regressions."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from fx1.serve import signing

_TEST_KEY = "SYNTHETIC-test-key"


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(signing.SIGNING_KEY_ENV, _TEST_KEY)
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"SYNTHETIC weights")
    (root / "config.json").write_text("{}", encoding="utf-8")
    signing.sign_release(root)
    return root


def _write_authenticated_manifest(root: Path, content: bytes) -> None:
    (root / signing.MANIFEST_FILENAME).write_bytes(content)
    signature = hmac.new(_TEST_KEY.encode(), content, hashlib.sha256).hexdigest()
    (root / signing.SIGNATURE_FILENAME).write_text(signature, encoding="utf-8")


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")


def test_signed_checkpoint_round_trips_and_is_relocatable(checkpoint: Path) -> None:
    assert signing.verify_release(checkpoint) is True
    moved = checkpoint.with_name("moved-checkpoint")
    shutil.copytree(checkpoint, moved)
    assert signing.verify_release(moved) is True


def test_manifest_is_deterministic(checkpoint: Path) -> None:
    before = (checkpoint / signing.MANIFEST_FILENAME).read_bytes()
    signing.sign_release(checkpoint)
    assert (checkpoint / signing.MANIFEST_FILENAME).read_bytes() == before
    manifest = signing.build_manifest(checkpoint)
    assert list(manifest.artifacts) == sorted(manifest.artifacts)


@pytest.mark.parametrize("name", ["extra.bin", ".hidden", "nested/weights.bin"])
def test_unlisted_artifact_is_rejected(checkpoint: Path, name: str) -> None:
    path = checkpoint / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"SYNTHETIC unlisted artifact")
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("name", [signing.SIGNATURE_FILENAME, signing.MANIFEST_FILENAME])
def test_only_root_metadata_is_excluded(checkpoint: Path, name: str) -> None:
    nested = checkpoint / "nested"
    nested.mkdir()
    artifact = nested / name
    artifact.write_bytes(b"SYNTHETIC nested artifact")
    signing.sign_release(checkpoint)
    assert str(artifact.relative_to(checkpoint)) in signing.build_manifest(checkpoint).artifacts
    assert signing.verify_release(checkpoint) is True
    artifact.write_bytes(b"changed")
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("action", ["tamper", "delete", "replace-with-directory"])
def test_changed_artifact_is_rejected(checkpoint: Path, action: str) -> None:
    weights = checkpoint / "weights.bin"
    if action == "tamper":
        weights.write_bytes(b"altered weights")
    else:
        weights.unlink()
        if action == "replace-with-directory":
            weights.mkdir()
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("signature", [b"", b" ", b"bad", b"f" * 64, b"\xff", "é".encode()])
def test_malformed_signature_returns_false(checkpoint: Path, signature: bytes) -> None:
    (checkpoint / signing.SIGNATURE_FILENAME).write_bytes(signature)
    assert signing.verify_release(checkpoint) is False


def test_signature_padding_remains_supported(checkpoint: Path) -> None:
    signature = checkpoint / signing.SIGNATURE_FILENAME
    signature.write_bytes(b" \n" + signature.read_bytes() + b"\t\n")
    assert signing.verify_release(checkpoint) is True


@pytest.mark.parametrize("content", [b"{", b"\xff", b"[]", b"null", b"{}"])
def test_authenticated_malformed_manifest_returns_false(checkpoint: Path, content: bytes) -> None:
    _write_authenticated_manifest(checkpoint, content)
    assert signing.verify_release(checkpoint) is False


def test_empty_authenticated_inventory_is_rejected(checkpoint: Path) -> None:
    content = json.dumps({"checkpoint_dir": str(checkpoint), "artifacts": {}}).encode()
    _write_authenticated_manifest(checkpoint, content)
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("absolute", [False, True])
def test_manifest_paths_are_not_opened(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    outside = checkpoint.parent / "outside.bin"
    outside.write_bytes(b"SYNTHETIC outside content")
    relative = str(outside) if absolute else "../outside.bin"
    content = json.dumps(
        {
            "checkpoint_dir": str(checkpoint),
            "artifacts": {relative: hashlib.sha256(outside.read_bytes()).hexdigest()},
        }
    ).encode()
    _write_authenticated_manifest(checkpoint, content)
    original_hash = signing._hash_file
    opened: list[Path] = []

    def checked_hash(path: Path) -> str:
        assert path.resolve().is_relative_to(checkpoint.resolve())
        opened.append(path)
        return original_hash(path)

    monkeypatch.setattr(signing, "_hash_file", checked_hash)
    assert signing.verify_release(checkpoint) is False
    assert outside not in opened


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "metadata"])
def test_symlinks_are_rejected(checkpoint: Path, kind: str) -> None:
    outside = checkpoint.parent / "external"
    if kind == "directory":
        outside.mkdir()
    elif kind != "dangling":
        outside.write_bytes(b"SYNTHETIC external content")
    link = checkpoint / "linked-artifact"
    if kind == "metadata":
        link = checkpoint / signing.MANIFEST_FILENAME
        outside.write_bytes(link.read_bytes())
        link.unlink()
    _symlink(link, outside, directory=kind == "directory")
    assert signing.verify_release(checkpoint) is False
    with pytest.raises(ValueError, match="regular files|symlinks"):
        signing.sign_release(checkpoint)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unsupported")
def test_special_files_are_rejected_without_blocking(checkpoint: Path) -> None:
    fifo = checkpoint / "pipe"
    os.mkfifo(fifo)
    assert signing.verify_release(checkpoint) is False
    with pytest.raises(ValueError, match="regular files"):
        signing._hash_file(fifo)


def test_missing_key_does_not_replace_existing_metadata(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [checkpoint / signing.MANIFEST_FILENAME, checkpoint / signing.SIGNATURE_FILENAME]
    before = [path.read_bytes() for path in paths]
    (checkpoint / "weights.bin").write_bytes(b"SYNTHETIC new weights")
    monkeypatch.delenv(signing.SIGNING_KEY_ENV)
    with pytest.raises(RuntimeError, match="is not set"):
        signing.sign_release(checkpoint)
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(RuntimeError, match="is not set"):
        signing.verify_release(checkpoint)


def test_missing_key_does_not_create_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "weights.bin").write_bytes(b"SYNTHETIC")
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="is not set"):
        signing.sign_release(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["weights.bin"]


def test_missing_metadata_is_false_even_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(signing.SIGNING_KEY_ENV, raising=False)
    assert signing.verify_release(tmp_path) is False


def test_hashing_does_not_read_whole_artifact(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"SYNTHETIC block" * 100_000
    weights = checkpoint / "weights.bin"
    weights.write_bytes(payload)

    def forbid_read_bytes(path: Path) -> bytes:
        pytest.fail(f"whole-file read attempted: {path}")

    monkeypatch.setattr(Path, "read_bytes", forbid_read_bytes)
    assert signing._hash_file(weights) == hashlib.sha256(payload).hexdigest()


def test_bad_hmac_does_not_hash_artifacts(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (checkpoint / signing.SIGNATURE_FILENAME).write_bytes(b"invalid")

    def forbid_hash(path: Path) -> str:
        pytest.fail(f"unauthenticated manifest triggered artifact hashing: {path}")

    monkeypatch.setattr(signing, "_hash_file", forbid_hash)
    assert signing.verify_release(checkpoint) is False


@pytest.mark.parametrize("filename", [signing.SIGNATURE_FILENAME, signing.MANIFEST_FILENAME])
def test_metadata_read_failures_return_false(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    original_open = signing.os.open

    def denied_open(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if Path(path) == checkpoint / filename:
            raise PermissionError("SYNTHETIC read fault")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(signing.os, "open", denied_open)
    assert signing.verify_release(checkpoint) is False


def test_walk_failures_are_not_ignored(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied_walk(
        root: Path, *, onerror: Callable[[OSError], None]
    ) -> Iterator[tuple[str, list[str], list[str]]]:
        onerror(PermissionError("SYNTHETIC scan fault"))
        return iter(())

    monkeypatch.setattr(signing.os, "walk", denied_walk)
    assert signing.verify_release(checkpoint) is False
    with pytest.raises(PermissionError, match="scan fault"):
        signing.build_manifest(checkpoint)


@pytest.mark.parametrize("kind", ["manifest", "signature"])
def test_metadata_reads_are_bounded(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    constant = "_MAX_MANIFEST_BYTES" if kind == "manifest" else "_MAX_SIGNATURE_BYTES"
    monkeypatch.setattr(signing, constant, 8)
    assert signing.verify_release(checkpoint) is False


def test_oversized_manifest_is_not_published(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = checkpoint / signing.MANIFEST_FILENAME
    before = path.read_bytes()
    monkeypatch.setattr(signing, "_MAX_MANIFEST_BYTES", 8)
    with pytest.raises(ValueError, match="byte limit"):
        signing.sign_release(checkpoint)
    assert path.read_bytes() == before


def _descriptor_dir() -> Path | None:
    """``/dev/fd`` exists on macOS and Linux; ``/proc/self/fd`` is Linux-only."""
    for candidate in (Path("/dev/fd"), Path("/proc/self/fd")):
        if candidate.is_dir():
            return candidate
    return None


def test_hash_failure_closes_descriptor(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc_fds = _descriptor_dir()
    if proc_fds is None:
        pytest.skip("descriptor-count probe requires /dev/fd or /proc/self/fd")

    def failed_digest(*args: object, **kwargs: object) -> object:
        raise OSError("SYNTHETIC digest failure")

    monkeypatch.setattr(signing.hashlib, "file_digest", failed_digest)
    before = len(list(proc_fds.iterdir()))
    for _ in range(20):
        with pytest.raises(OSError, match="digest failure"):
            signing._hash_file(checkpoint / "weights.bin")
    assert len(list(proc_fds.iterdir())) == before
