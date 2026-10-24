"""Export committed research artifacts into ``web/public/fixtures/``.

The research explorer web app is a static, read-only bundle. This script
materializes its data fixtures from the repo's real, committed evidence:

- ``receipts/*.json``           -> copied verbatim into ``fixtures/receipts/``
- ``artifacts/carry_*.json``    -> strategy stat fixtures (verbatim copies)
- ``artifacts/*equity*.parquet``-> equity-curve fixtures (compact JSON rows)
- ``fixtures/index.json``       -> manifest the app loads first, including a
                                   sha256 -> repo-path match table so the UI
                                   can show which embedded receipt hashes
                                   resolve to committed files in this checkout.

Sources are read-only; originals are never modified. Re-run after any
(sealed, intentional) change to ``receipts/`` or the committed artifacts::

    uv run --no-sync python web/scripts/export_fixtures.py

``--check`` regenerates everything into a scratch directory and exits 1 when
any committed fixture is missing, stale, or extra — the same freshness gate
as ``scripts/gen_arch_diagrams.py --check``.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "web" / "public" / "fixtures"

# Directories scanned when resolving embedded sha256 fields to committed files.
# Bounded to keep the export fast; extend if new hash fields appear.
HASH_SCAN_ROOTS = ("src", "scripts", "configs", "examples", "receipts", "artifacts")
HASH_SCAN_MAX_BYTES = 20 * 1024 * 1024

EQUITY_SOURCES = (
    ("carry", "carry_equity_full.parquet", "carry_champion.json", "Carry champion"),
    (
        "carry_expanded",
        "carry_equity_expanded_full.parquet",
        "carry_champion_expanded.json",
        "Carry champion (expanded universe)",
    ),
)

# Pre-envelope artifact kept for provenance under receipts/legacy-unsealed/
# (unsealed; outside the verify-all audited set — see that dir's README).
ADAPTIVE_MIX_RECEIPT = "legacy-unsealed/adaptive_mix_20asset_1d_20260922.json"


def _canonical_json_bytes(value: Any) -> bytes:
    """Mirror ``quant_fund.utils.hashing.canonical_json_bytes``.

    Kept dependency-free so the script runs under the shared env without
    importing the package under test.
    """

    def canon(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): canon(sub) for key, sub in item.items()}
        if isinstance(item, (list, tuple)):
            return [canon(sub) for sub in item]
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, bytes):
            return item.hex()
        scalar = getattr(item, "item", None)
        if callable(scalar):
            try:
                return canon(scalar())
            except (TypeError, ValueError):
                pass
        return item

    return json.dumps(
        canon(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _committed_files(revision: str = "HEAD") -> dict[str, str]:
    """Bounded inventory of ordinary blobs at HEAD; never index working-tree bytes."""
    raw = subprocess.check_output(
        ["git", "ls-tree", "-r", "-l", "-z", revision, "--", *HASH_SCAN_ROOTS],
        cwd=REPO_ROOT,
    )
    files: dict[str, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        if kind == b"blob" and mode in (b"100644", b"100755") and int(size) <= HASH_SCAN_MAX_BYTES:
            files[name.decode("utf-8")] = oid.decode("ascii")
    return files


def _committed_bytes(oid: str) -> bytes:
    return subprocess.check_output(["git", "cat-file", "blob", oid], cwd=REPO_ROOT)


def _build_hash_index(files: dict[str, str] | None = None) -> dict[str, str]:
    """Map digests of committed blobs, including canonical JSON, to repo paths."""
    matches: dict[str, str] = {}
    for rel, oid in sorted((files if files is not None else _committed_files()).items()):
        raw = _committed_bytes(oid)
        matches.setdefault(hashlib.sha256(raw).hexdigest(), rel)
        if rel.endswith(".json"):
            try:
                canonical = hashlib.sha256(_canonical_json_bytes(json.loads(raw))).hexdigest()
            except (ValueError, UnicodeDecodeError):
                continue
            matches.setdefault(canonical, rel + " (canonical-json)")
    return matches


def _key_is_hashy(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered == "sha256" or lowered.endswith(("_sha256", "_sha", "_hashes")) or "hash" in lowered
    )


def _collect_hashes(value: Any, into: list[str], parent_hashy: bool = False) -> None:
    """Collect digest fields, retaining duplicate fields to match the UI count."""
    if isinstance(value, dict):
        for key, sub in value.items():
            hashy = _key_is_hashy(key)
            if (
                isinstance(sub, str)
                and re.fullmatch(r"[0-9a-f]{64}", sub)
                and (hashy or parent_hashy)
            ):
                into.append(sub)
            _collect_hashes(sub, into, hashy)
    elif isinstance(value, list):
        for sub in value:
            _collect_hashes(sub, into, False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _export_equity(parquet_path: Path, out_path: Path, raw: bytes) -> dict[str, Any]:
    """Convert an equity parquet to a compact JSON columnar fixture."""

    import pandas as pd  # shared env provides pandas/pyarrow

    frame = pd.read_parquet(io.BytesIO(raw))
    required = {"event_time", "nav", "gross", "net", "turnover"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{parquet_path.name} missing columns: {sorted(missing)}")
    if frame.empty or frame["event_time"].isna().any():
        raise ValueError(f"{parquet_path.name} has empty or invalid timestamps")
    for column in ("nav", "gross", "net", "turnover"):
        if not all(math.isfinite(float(v)) for v in frame[column]):
            raise ValueError(f"{parquet_path.name} has nonfinite {column}")
    frame = frame.sort_values("event_time").reset_index(drop=True)
    points = [
        [
            row.event_time.date().isoformat(),
            round(float(row.nav), 6),
            round(float(row.gross), 8),
            round(float(row.net), 8),
            round(float(row.turnover), 8),
        ]
        for row in frame.itertuples(index=False)
    ]
    payload = {
        "fields": ["date", "nav", "gross", "net", "turnover"],
        "points": points,
        "rows": len(points),
        "source_file": parquet_path.relative_to(REPO_ROOT).as_posix(),
        "units": {"nav": "USD", "gross": "fraction", "net": "fraction", "turnover": "fraction"},
    }
    _write_json(out_path, payload)
    return {
        "rows": len(points),
        "first_date": points[0][0],
        "last_date": points[-1][0],
        "nav_final": points[-1][1],
    }


def _normalize_carry_stats(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Carry champion artifacts store dev/holdout/full segment blobs."""

    out: dict[str, dict[str, float]] = {}
    for segment in ("dev", "holdout", "full"):
        blob = raw.get(segment)
        if isinstance(blob, dict):
            out[segment] = {k: v for k, v in blob.items() if isinstance(v, (int, float))}
    return out


def _strategy_entry(
    strategy_id: str,
    name: str,
    *,
    kind: str,
    data_source: str,
    provenance: dict[str, Any],
    segments: dict[str, dict[str, float]],
    extras: dict[str, Any] | None = None,
    stats_file: str | None = None,
    equity_file: str | None = None,
    fixtures_dir: Path = FIXTURES_DIR,
) -> tuple[dict[str, Any], str]:
    """Emit a normalized strategy stats fixture and return its index entry."""

    stats_rel = stats_file or f"strategies/{strategy_id}.json"
    stats_payload = {
        "id": strategy_id,
        "name": name,
        "kind": kind,
        "data_source": data_source,
        "research_only": True,
        "live_pnl_claim": False,
        "provenance": provenance,
        "segments": segments,
        "extras": extras or {},
        "equity_file": equity_file,
    }
    _write_json(fixtures_dir / stats_rel, stats_payload)
    entry = {
        "id": strategy_id,
        "name": name,
        "kind": kind,
        "data_source": data_source,
        "stats_file": stats_rel,
        "equity_file": equity_file,
        "has_equity": equity_file is not None,
        "segments": sorted(segments.keys()),
    }
    return entry, stats_rel


def _export(fixtures_dir: Path) -> dict[str, int]:
    """Materialize every fixture under ``fixtures_dir``; return row counts."""
    receipts_dir = REPO_ROOT / "receipts"
    artifacts_dir = REPO_ROOT / "artifacts"
    if not receipts_dir.is_dir():
        raise SystemExit("receipts/ missing — run from the repo checkout")

    generated_at = datetime.now(UTC).isoformat()
    revision = _git_revision()
    if revision is None:
        raise SystemExit("committed Git revision is required for evidence export")
    files = _committed_files(revision)
    hash_index = _build_hash_index(files)

    # ---- receipts: verbatim copies + index rows ---------------------------
    receipt_entries: list[dict[str, Any]] = []
    receipt_hash_matches: dict[str, str] = {}
    for name in sorted(files):
        if not name.startswith("receipts/") or name.count("/") != 1 or not name.endswith(".json"):
            continue
        src = REPO_ROOT / name
        rel = f"receipts/{src.name}"
        dst = fixtures_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = _committed_bytes(files[name])
        dst.write_bytes(raw)
        payload = json.loads(raw)
        hashes: list[str] = []
        _collect_hashes(payload, hashes)
        for digest in hashes:
            if digest in hash_index:
                receipt_hash_matches[digest] = hash_index[digest]
        # receipt.v2 envelopes carry the honesty flag as data_label: anything
        # that is not REAL is research-only by contract. A receipt declaring
        # neither keeps None — unknown stays blank.
        research_only = payload.get("research_only")
        if research_only is None and "data_label" in payload:
            research_only = payload["data_label"] != "REAL"
        receipt_entries.append(
            {
                "id": src.stem,
                "file": rel,
                "schema": payload.get("schema"),
                "evidence_level": payload.get("evidence_level"),
                "research_only": research_only,
                "live_pnl_claim": payload.get("live_pnl_claim"),
                "created_at": payload.get("created_at") or payload.get("generated_at"),
                "n_hashes": len(hashes),
                "n_hash_matches": sum(1 for h in hashes if h in hash_index),
            }
        )

    # ---- strategies --------------------------------------------------------
    strategies: list[dict[str, Any]] = []

    for strategy_id, parquet_name, champion_name, display in EQUITY_SOURCES:
        parquet_path = artifacts_dir / parquet_name
        if f"artifacts/{parquet_name}" not in files or f"artifacts/{champion_name}" not in files:
            raise SystemExit(f"missing artifacts for {strategy_id}")
        equity_rel = f"equity/{strategy_id}.json"
        equity_meta = _export_equity(
            parquet_path,
            fixtures_dir / equity_rel,
            _committed_bytes(files[f"artifacts/{parquet_name}"]),
        )
        raw_stats = json.loads(_committed_bytes(files[f"artifacts/{champion_name}"]))
        segments = _normalize_carry_stats(raw_stats)
        entry, _ = _strategy_entry(
            strategy_id,
            display,
            kind="equity_backed",
            data_source=str(raw_stats.get("data_source", "UNVERIFIED")),
            provenance={
                "source_files": [
                    f"artifacts/{champion_name}",
                    f"artifacts/{parquet_name}",
                ],
                "note": "Committed research artifact; simulated NAV path, "
                "not live trading performance.",
                "equity": equity_meta,
            },
            segments=segments,
            equity_file=equity_rel,
            fixtures_dir=fixtures_dir,
        )
        strategies.append(entry)

    if f"receipts/{ADAPTIVE_MIX_RECEIPT}" in files:
        receipt = json.loads(_committed_bytes(files[f"receipts/{ADAPTIVE_MIX_RECEIPT}"]))
        receipt_rel = f"receipts/{ADAPTIVE_MIX_RECEIPT}"
        results = receipt.get("results", {})
        for allocator_id in ("adaptive", "equal"):
            blob = results.get(allocator_id)
            if not isinstance(blob, dict):
                continue
            segments = {
                name: {k: v for k, v in seg.items() if isinstance(v, (int, float))}
                for name, seg in (blob.get("segments") or {}).items()
                if isinstance(seg, dict)
            }
            extras = {
                key: blob[key] for key in blob if key.startswith("full_path") and key != "segments"
            }
            entry, _ = _strategy_entry(
                f"amix_{allocator_id}",
                f"Adaptive mix replay — {allocator_id} allocator",
                kind="receipt_stats",
                data_source=str(receipt.get("data_source", "unknown")),
                provenance={
                    "receipt": receipt_rel,
                    "pointer": f"results.{allocator_id}",
                    "note": "Segment stats recorded in sealed receipt "
                    f"{ADAPTIVE_MIX_RECEIPT}; no equity series exported.",
                },
                segments=segments,
                extras=extras,
                fixtures_dir=fixtures_dir,
            )
            strategies.append(entry)
        sleeves = receipt.get("paper_sleeves", {})
        for sleeve in sorted(sleeves):
            blob = sleeves[sleeve]
            if not isinstance(blob, dict):
                continue
            segments = {
                name: {k: v for k, v in seg.items() if isinstance(v, (int, float))}
                for name, seg in blob.items()
                if isinstance(seg, dict)
            }
            entry, _ = _strategy_entry(
                f"amix_sleeve_{sleeve}",
                f"Adaptive mix sleeve — {sleeve}",
                kind="receipt_stats",
                data_source=str(receipt.get("data_source", "unknown")),
                provenance={
                    "receipt": receipt_rel,
                    "pointer": f"paper_sleeves.{sleeve}",
                    "note": "Paper-sleeve segment stats recorded in sealed "
                    "receipt; diagnostic only, no equity series exported.",
                },
                segments=segments,
                fixtures_dir=fixtures_dir,
            )
            strategies.append(entry)

    index = {
        "generated_at": generated_at,
        "generator": "web/scripts/export_fixtures.py",
        "repo_revision": revision,
        "honesty": {
            "research_only": True,
            "live_pnl_claim": False,
            "note": "All fixtures are verbatim copies or mechanical exports of "
            "committed research evidence. The app renders stored fields; it "
            "makes no performance or live-trading claims.",
        },
        "strategies": strategies,
        "receipts": receipt_entries,
        "hash_matches": receipt_hash_matches,
    }
    _write_json(fixtures_dir / "index.json", index)
    return {
        "receipts": len(receipt_entries),
        "strategies": len(strategies),
        "hash_matches": len(receipt_hash_matches),
    }


def _index_payload(path: Path) -> dict[str, Any]:
    """index.json minus the volatile stamp fields."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("generated_at", "repo_revision"):
        payload.pop(key, None)
    return payload


def _stale_fixtures(expected: Path, actual: Path) -> list[str]:
    """Fixture paths whose committed bytes differ from a fresh export."""
    stale: list[str] = []
    documentation = {Path("README.md")}
    if actual.is_dir():
        for path in sorted(actual.rglob("*")):
            if path.is_file():
                rel = path.relative_to(actual)
                if rel not in documentation and not (expected / rel).is_file():
                    stale.append(f"{rel} (no longer produced)")
    for path in sorted(expected.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(expected)
        committed = actual / rel
        if not committed.is_file():
            stale.append(f"{rel} (missing)")
        elif rel.name == "index.json":
            if _index_payload(path) != _index_payload(committed):
                stale.append(f"{rel} (drifted)")
        elif not filecmp.cmp(path, committed, shallow=False):
            stale.append(f"{rel} (drifted)")
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed fixtures are fresh; do not write",
    )
    args = parser.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory(
            prefix=".fixture-check-", dir=FIXTURES_DIR.parent
        ) as scratch:
            fresh = Path(scratch) / "fixtures"
            _export(fresh)
            stale = _stale_fixtures(fresh, FIXTURES_DIR)
        if stale:
            print("stale fixtures — run `python web/scripts/export_fixtures.py`:")
            for rel in stale:
                print(f"  {rel}")
            return 1
        print("fixtures are fresh")
        return 0

    counts = _export(FIXTURES_DIR)
    print(
        f"exported {counts['receipts']} receipts, "
        f"{counts['strategies']} strategies, "
        f"{counts['hash_matches']} hash matches -> {FIXTURES_DIR}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
