"""SYNTHETIC symlink traversal regressions for durable uploads."""

from __future__ import annotations

from pathlib import Path

import pytest

from fx1.serve.uploads import UploadStore


def _store(root: Path) -> UploadStore:
    return UploadStore(max_entries=4, max_bytes=100, state_dir=root)


def test_gc_unlinks_upload_directory_symlink_without_following_it(tmp_path: Path) -> None:
    store = _store(tmp_path / "state")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"SYNTHETIC-secret")
    assert store._dir is not None
    store._dir.mkdir(parents=True, exist_ok=True)
    link = store._dir / "upload_planted"
    link.symlink_to(outside, target_is_directory=True)

    store._gc_blobs()

    assert secret.read_bytes() == b"SYNTHETIC-secret"
    assert not link.exists()


def test_cancel_unlinks_upload_symlink_without_deleting_external_files(tmp_path: Path) -> None:
    store = _store(tmp_path / "state")
    meta = store.create(purpose="batch", filename="a.jsonl", nbytes=2, mime_type="text/plain")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.bin"
    secret.write_bytes(b"SYNTHETIC-secret")
    assert store._dir is not None
    link = store._dir / meta.upload_id
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)

    store._drop_parts(meta)

    assert secret.read_bytes() == b"SYNTHETIC-secret"
    assert not link.exists()


def test_blob_path_rejects_escape_and_symlinked_part(tmp_path: Path) -> None:
    store = _store(tmp_path / "state")
    meta = store.create(purpose="batch", filename="a.jsonl", nbytes=2, mime_type="text/plain")
    assert store._dir is not None
    upload_dir = store._dir / meta.upload_id
    upload_dir.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"SYNTHETIC-secret")
    (upload_dir / "part_planted.bin").symlink_to(outside)

    with pytest.raises(RuntimeError, match="escapes"):
        store._blob(meta.upload_id, "part_planted")
    with pytest.raises(RuntimeError, match="escapes"):
        store._blob("../other", "part_safe")
    with pytest.raises(RuntimeError, match="escapes"):
        store._blob(meta.upload_id, "../outside")
