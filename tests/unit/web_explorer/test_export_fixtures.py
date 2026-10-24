"""Integrity checks for the web explorer's committed fixtures.

The app serves verbatim copies of sealed receipts plus mechanical exports of
committed artifacts (see ``web/scripts/export_fixtures.py``). These tests pin
the fixture contract: copies stay byte-identical to the sealed originals
(receipts are immutable — a mismatch means someone edited a receipt or the
fixtures are stale), the index references resolve, and equity series are
well-formed. No quant_fund imports; runs under plain pytest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "web" / "public" / "fixtures"

SHA256_LEN = 64


def _load(name: str) -> dict:
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {name} — re-run export script"
    payload: object = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


def test_index_shape_and_references() -> None:
    index = _load("index.json")
    assert index["honesty"]["research_only"] is True
    assert index["honesty"]["live_pnl_claim"] is False
    assert index["strategies"], "index has no strategies"
    assert index["receipts"], "index has no receipts"
    for strategy in index["strategies"]:
        assert (FIXTURES / strategy["stats_file"]).is_file()
        if strategy["equity_file"] is not None:
            assert (FIXTURES / strategy["equity_file"]).is_file()
    for receipt in index["receipts"]:
        assert (FIXTURES / receipt["file"]).is_file()


def test_receipt_fixtures_are_byte_identical_to_sealed_sources() -> None:
    """Receipts are immutable: fixture copies must equal receipts/ exactly."""
    index = _load("index.json")
    for receipt in index["receipts"]:
        src = REPO_ROOT / receipt["file"]
        assert src.is_file(), f"source receipt missing: {receipt['file']}"
        fixture = FIXTURES / receipt["file"]
        assert fixture.read_bytes() == src.read_bytes(), (
            f"{receipt['file']} drifted — re-run "
            "web/scripts/export_fixtures.py (receipts are immutable)"
        )


def test_fixtures_cover_every_sealed_receipt() -> None:
    """Every committed receipts/*.json must be exported — a silent gap hides
    sealed evidence from the explorer (this caught a stale-fixture drift)."""
    index = _load("index.json")
    indexed = {receipt["file"] for receipt in index["receipts"]}
    sealed = {f"receipts/{p.name}" for p in (REPO_ROOT / "receipts").glob("*.json")}
    assert sealed == indexed, f"fixture/receipt mismatch: {sorted(sealed ^ indexed)}"
    assert {p.name for p in (FIXTURES / "receipts").glob("*.json")} == {
        p.name for p in (REPO_ROOT / "receipts").glob("*.json")
    }


def test_every_committed_receipt_is_exported() -> None:
    """A new receipts/*.json without a fixture export silently drops evidence
    from the explorer. Completeness is pinned both directions."""
    committed = {f"receipts/{p.name}" for p in (REPO_ROOT / "receipts").glob("*.json")}
    exported = {receipt["file"] for receipt in _load("index.json")["receipts"]}
    assert committed - exported == set(), (
        f"receipts missing from fixtures — re-run web/scripts/export_fixtures.py: "
        f"{sorted(committed - exported)}"
    )
    assert exported - committed == set(), (
        f"fixtures index points at deleted receipts: {sorted(exported - committed)}"
    )


def test_receipts_declare_non_live_evidence() -> None:
    """Receipts must never claim live P&L and must carry at least one
    non-live marker (research_only, a declared data_label, dev_only, or a
    research-level seal)."""
    for path in sorted((FIXTURES / "receipts").glob("*.json")):
        payload = json.loads(path.read_text())
        # A missing live_pnl_claim key is no claim; only an explicit True is a
        # live-P&L claim. Sealed real-corpus drills (data_label=yahoo_eod and
        # friends) and META audit envelopes are evidence without the flag.
        assert payload.get("live_pnl_claim") is not True, path.name
        # ``*.seal.json`` is a prereg tamper sidecar (schema + file hash),
        # not an evidence claim. Honesty lives on the sealed document.
        if path.name.endswith(".seal.json") or payload.get("schema") == "prereg_seal.v1":
            continue
        honesty = payload.get("honesty")
        # Forward pre-registrations keep the flag inside ``honesty``; the
        # content hash covers that block, so it is the marker.
        honesty_research = isinstance(honesty, dict) and honesty.get("research_only") is True
        assert (
            payload.get("research_only") is True
            or isinstance(payload.get("data_label"), str)
            or payload.get("dev_only") is True
            or payload.get("level") == "research"
            or honesty_research
        ), f"{path.name}: no non-live-evidence marker"


def test_equity_fixtures_are_finite_sorted_series() -> None:
    index = _load("index.json")
    for strategy in index["strategies"]:
        eq = strategy.get("equity_file")
        if eq is None:
            continue
        payload = _load(eq)
        points = payload["points"]
        assert payload["rows"] == len(points)
        dates = [p[0] for p in points]
        assert dates == sorted(dates), f"{eq}: dates out of order"
        for row in points:
            _, nav, gross, net, turnover = row
            for value in (nav, gross, net, turnover):
                assert math.isfinite(value), eq
        assert payload["source_file"].startswith("artifacts/")


def test_hash_match_table_points_at_committed_files() -> None:
    index = _load("index.json")
    for digest, rel in index["hash_matches"].items():
        assert len(digest) == SHA256_LEN
        int(digest, 16)  # must be hex
        canonical = rel.removesuffix(" (canonical-json)")
        assert (REPO_ROOT / canonical).is_file(), f"match target missing: {rel}"


def test_strategies_keep_honesty_flags() -> None:
    for path in sorted((FIXTURES / "strategies").glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload["research_only"] is True, path.name
        assert payload["live_pnl_claim"] is False, path.name
        assert payload["segments"], path.name


def _exporter():
    import importlib.util

    path = REPO_ROOT / "web/scripts/export_fixtures.py"
    spec = importlib.util.spec_from_file_location("web_fixture_export", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_hashes_ignore_dirty_untracked_and_symlink_bytes(tmp_path, monkeypatch) -> None:
    import hashlib
    import subprocess

    module = _exporter()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src/example.py"
    source.parent.mkdir()
    committed = b"print('committed')\n"
    source.write_bytes(committed)
    (source.parent / "link.py").symlink_to("example.py")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source.write_bytes(b"dirty\n")
    (source.parent / "untracked.py").write_bytes(b"untracked\n")
    matches = module._build_hash_index()
    assert matches == {hashlib.sha256(committed).hexdigest(): "src/example.py"}
    files = module._committed_files()
    assert module._committed_bytes(files["src/example.py"]) == committed


def test_export_hash_counts_match_fields_not_distinct_values() -> None:
    module = _exporter()
    values: list[str] = []
    module._collect_hashes(
        {"a_sha256": "a" * 64, "input_hashes": {"x": "a" * 64, "bad": "z" * 64}}, values
    )
    assert values == ["a" * 64, "a" * 64]


def test_check_mode_detects_stale_fixtures(tmp_path: Path, monkeypatch) -> None:
    """--check must fail closed on missing, drifted, or extra fixture files."""
    module = _exporter()
    assert module.main(["--check"]) == 0, "committed fixtures should be fresh"

    fresh = tmp_path / "fresh"
    module._export(fresh)
    assert module._stale_fixtures(fresh, FIXTURES) == []

    missing = tmp_path / "missing"
    module._export(missing)
    next((missing / "receipts").glob("*.json")).unlink()
    assert module._stale_fixtures(fresh, missing)

    drifted = tmp_path / "drifted"
    module._export(drifted)
    index = json.loads((drifted / "index.json").read_text())
    index["receipts"] = index["receipts"][:1]
    (drifted / "index.json").write_text(json.dumps(index, indent=1))
    assert module._stale_fixtures(fresh, drifted) == ["index.json (drifted)"]

    extra = tmp_path / "extra"
    module._export(extra)
    (extra / "receipts" / "ghost.json").write_text("{}")
    assert module._stale_fixtures(fresh, extra) == ["receipts/ghost.json (no longer produced)"]


def test_stale_fixture_check_ignores_documentation(tmp_path: Path) -> None:
    module = _exporter()
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    (actual / "README.md").write_text("fixture documentation\n", encoding="utf-8")

    assert module._stale_fixtures(expected, actual) == []
