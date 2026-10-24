"""Lake path traversal guards and atomic publish — no network."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.lake import Lake


@pytest.mark.parametrize(
    "rel",
    [
        "",
        "/abs/x",
        "../escape",
        "a/../b",
        "a\\b",
    ],
)
def test_lake_rejects_path_traversal(tmp_path: Path, rel: str) -> None:
    lake = Lake(tmp_path / "lake")
    with pytest.raises(ValueError, match="clean relative path"):
        lake._resolve(rel)


def test_lake_write_read_roundtrip_and_exists(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "lake")
    frame = pl.DataFrame({"security_id": ["A"], "close": [1.0]})
    path = lake.write_parquet(frame, "silver/bars.parquet")
    assert path.is_file()
    assert lake.exists("silver/bars.parquet") is True
    assert lake.exists("silver/missing.parquet") is False
    got = lake.read_parquet("silver/bars.parquet", columns=["close"])
    assert got.columns == ["close"]
    assert float(got["close"][0]) == pytest.approx(1.0)
