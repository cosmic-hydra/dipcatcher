"""Edges for crash-safe writes the durable-path suite left closed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quant_fund.utils.atomicio import atomic_write_text, publish_text_once


def test_directory_fsync_failure_still_publishes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = os.open

    def _open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if flags == os.O_RDONLY and Path(str(path)).is_dir():
            raise OSError("directory fsync unsupported")
        return real_open(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", _open)
    dest = tmp_path / "receipt.json"
    atomic_write_text(dest, '{"ok": true}\n')
    assert dest.read_text(encoding="utf-8") == '{"ok": true}\n'


def test_publish_race_refuses_different_bytes_and_accepts_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _lose_the_race(source: Path, dest: Path, content: str) -> None:
        del source
        Path(dest).write_text(content, encoding="utf-8")
        raise FileExistsError

    dest = tmp_path / "sealed.json"
    monkeypatch.setattr(os, "link", lambda source, target: _lose_the_race(source, target, "other"))
    with pytest.raises(FileExistsError, match="different content"):
        publish_text_once(dest, "wanted")
    assert dest.read_text(encoding="utf-8") == "other"

    same = tmp_path / "same.json"
    monkeypatch.setattr(os, "link", lambda source, target: _lose_the_race(source, target, "wanted"))
    publish_text_once(same, "wanted")
    assert same.read_text(encoding="utf-8") == "wanted"
