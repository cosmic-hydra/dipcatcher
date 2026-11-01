"""Lake write-path hardening (P6.3 follow-up).

Same defect class as pooled_stream: the destination path must never exist
in truncated form, concurrent writers to one rel must not share a tmp file,
and ``rel`` must not escape the lake root.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.lake import Lake


def _frame(tag: int) -> pl.DataFrame:
    return pl.DataFrame({"security_id": [f"S{tag}"], "close": [float(tag)]})


def test_write_parquet_rejects_path_traversal(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    for bad in ("../outside.parquet", "/abs/x.parquet", "a/../b.parquet", "x\\y.parquet", ""):
        with pytest.raises(ValueError, match="clean relative path"):
            lake.write_parquet(_frame(1), bad)
    assert not (tmp_path / "outside.parquet").exists()


def test_read_and_exists_reject_traversal(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    with pytest.raises(ValueError):
        lake.read_parquet("../secrets.parquet")
    with pytest.raises(ValueError):
        lake.exists("..")


def test_write_parquet_leaves_no_tmp_litter(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    lake.write_parquet(_frame(1), "silver/bars.parquet")
    litter = list((tmp_path / "silver").glob("*.tmp")) + list((tmp_path / "silver").glob(".*.tmp"))
    assert litter == []
    back = lake.read_parquet("silver/bars.parquet")
    assert back["close"].to_list() == [1.0]


def test_write_parquet_failed_write_leaves_prior_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = Lake(tmp_path)
    lake.write_parquet(_frame(7), "silver/bars.parquet")

    def crash_write(self: pl.DataFrame, target: Path) -> None:
        # Partial bytes at the tmp path, then die — the torn tmp must be
        # cleaned up and must never reach the canonical path.
        Path(target).write_bytes(b"\x50\x41\x52\x31truncated")
        raise OSError("simulated mid-write crash")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", crash_write)
    with pytest.raises(OSError, match="simulated"):
        lake.write_parquet(_frame(9), "silver/bars.parquet")
    assert lake.read_parquet("silver/bars.parquet")["close"].to_list() == [7.0]
    assert not list((tmp_path / "silver").glob("*.tmp"))


def test_concurrent_writers_same_rel_never_share_tmp(tmp_path: Path) -> None:
    """Two threads racing one destination must each write a private tmp file —
    a shared `<name>.tmp` lets one writer rename the other's torn output."""
    lake = Lake(tmp_path)
    rel = "silver/raced.parquet"
    barrier = threading.Barrier(8)

    def write(tag: int) -> None:
        barrier.wait()
        lake.write_parquet(_frame(tag), rel)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    # Final file must be one complete frame — never a mix of two tmp writes.
    result = lake.read_parquet(rel)
    assert result.height == 1
    assert int(result["close"][0]) in range(8)
    assert result["security_id"][0] == f"S{int(result['close'][0])}"
    assert not list((tmp_path / "silver").glob("*.tmp"))


@pytest.mark.parametrize("alias", ["silver//bars.parquet", "silver/./bars.parquet"])
def test_rename_lock_uses_normalized_destination(tmp_path: Path, alias: str) -> None:
    lake = Lake(tmp_path)
    canonical = lake._resolve("silver/bars.parquet")
    assert lake._rename_lock(canonical) is lake._rename_lock(lake._resolve(alias))
    assert lake._rename_lock(canonical) is not lake._rename_lock(
        lake._resolve("silver/other.parquet")
    )


def test_rename_lock_applies_platform_case_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exercise Windows key policy on any host, without claiming Windows I/O.
    import ntpath

    monkeypatch.setattr(os.path, "normcase", ntpath.normcase)
    lake = Lake(tmp_path)
    assert lake._rename_lock(lake._resolve("silver/Bars.parquet")) is lake._rename_lock(
        lake._resolve("SILVER/bars.parquet")
    )


def test_concurrent_alias_writers_serialize_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    lake = Lake(tmp_path)
    aliases = ["silver/raced.parquet", "silver//raced.parquet", "silver/./raced.parquet"]
    barrier = threading.Barrier(len(aliases))
    guard = threading.Lock()
    active = 0
    maximum = 0
    original_replace = os.replace
    original_lock = lake._rename_lock

    def ready_lock(path: Path) -> threading.Lock:
        barrier.wait(timeout=10)
        return original_lock(path)

    def observed_replace(src: Path, dst: Path) -> None:
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            # Widen the critical section to expose differently keyed locks.
            time.sleep(0.05)
            original_replace(src, dst)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(lake, "_rename_lock", ready_lock)
    monkeypatch.setattr(os, "replace", observed_replace)
    with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        futures = [
            pool.submit(lake.write_parquet, _frame(tag), rel) for tag, rel in enumerate(aliases)
        ]
        for future in futures:
            future.result(timeout=15)

    assert maximum == 1
    result = lake.read_parquet(aliases[0])
    assert result.height == 1
    tag = int(result["close"][0])
    assert tag in range(len(aliases))
    assert result["security_id"][0] == f"S{tag}"
    assert not list((tmp_path / "silver").glob(".*.tmp"))


def test_rename_lock_preserves_posix_case_sensitivity(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("POSIX case-sensitive lock policy")
    lake = Lake(tmp_path)
    upper = lake._rename_lock(lake._resolve("silver/Bars.parquet"))
    lower = lake._rename_lock(lake._resolve("silver/bars.parquet"))
    # Case-sensitive volumes keep distinct keys. Case-insensitive volumes
    # (default APFS) must share one lock, or the two names race on one inode.
    if lake._folds_case():
        assert upper is lower
    else:
        assert upper is not lower


def test_rename_lock_folds_case_when_volume_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.data.lake as lake_module

    monkeypatch.setattr(lake_module, "_directory_folds_case", lambda directory: True)
    lake = Lake(tmp_path)
    assert lake._rename_lock(lake._resolve("silver/Bars.parquet")) is lake._rename_lock(
        lake._resolve("silver/bars.parquet")
    )


def test_fsync_dir_platform_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verify dispatch and cleanup with a fake OS; no Windows I/O is exercised.
    from types import SimpleNamespace

    import quant_fund.data.lake as module

    events: list[object] = []

    def open_directory(path: Path, flags: int) -> int:
        events.append((path, flags))
        return 42

    def fail_flush(fd: int) -> None:
        events.append(("fsync", fd))
        raise OSError("directory flush failed")

    fake_os = SimpleNamespace(
        name="nt",
        O_RDONLY=0,
        O_DIRECTORY=65536,
        open=open_directory,
        fsync=fail_flush,
        close=lambda fd: events.append(("close", fd)),
    )
    monkeypatch.setattr(module, "os", fake_os)
    module._fsync_dir(tmp_path)
    assert events == []
    fake_os.name = "posix"
    with pytest.raises(OSError, match="directory flush failed"):
        module._fsync_dir(tmp_path)
    assert events == [(tmp_path, 65536), ("fsync", 42), ("close", 42)]
