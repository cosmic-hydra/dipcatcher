"""Verify the completed, sealed Phase-1 forecast and tournament runs.

These are separate research protocols from the canonical notebook schema in
``research.verify``. A passing result attests to the bytes and declared
provenance available in this checkout; the receipts are hashes, not signatures.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from quant_fund.metrics import inference, snooping
from quant_fund.research import cost_allocation, net_replay, net_tournament, real_benchmark
from quant_fund.research.catalog import BENCHMARK_CATALOG_VERSION
from quant_fund.utils.hashing import SHA256_HEX_LENGTH
from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON value: {token}")


def _receipt(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        compressed = path.with_name(path.name + ".gz")
        if path.exists() and compressed.exists():
            raise ValueError("ambiguous raw and gzip receipts")
        raw = (
            gzip.open(compressed, "rt", encoding="utf-8")
            if compressed.exists()
            else path.open("rt", encoding="utf-8")
        )
        with raw as handle:
            contents = handle.read()
        value = json.loads(
            contents,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        if not isinstance(value, dict):
            raise ValueError("receipt is not a JSON object")
        digest = value.get("receipt_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("missing or malformed receipt_sha256")
        content = {k: v for k, v in value.items() if k != "receipt_sha256"}
        expected = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if digest != expected:
            raise ValueError("receipt SHA-256 mismatch")
        return value
    except (OSError, UnicodeError, ValueError, TypeError, EOFError) as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def _receipt_exists(path: Path) -> bool:
    return path.exists() or path.with_name(path.name + ".gz").exists()


def _assert(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def _timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _sha256(value: Any, *, length: int = SHA256_HEX_LENGTH) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _committed_code_hashes(revision: str, errors: list[str]) -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    try:
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        errors.append("index: git_revision does not identify a local commit")
        return {}
    modules = (
        real_benchmark,
        net_tournament,
        net_replay,
        cost_allocation,
        inference,
        snooping,
    )
    hashes: dict[str, str] = {}
    for module in modules:
        source_name = module.__file__
        if source_name is None:
            errors.append("index: source module lacks a file")
            continue
        source = Path(source_name).resolve()
        try:
            relative = source.relative_to(root).as_posix()
            blob = subprocess.run(
                ["git", "-C", str(root), "show", f"{revision}:{relative}"],
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout
            hashes[source.name] = hashlib.sha256(blob).hexdigest()
        except (OSError, ValueError, subprocess.SubprocessError):
            errors.append(f"index: source {source.name} unavailable at git_revision")
    return hashes


_BENCHMARK_RUNTIME_KEYS = ("python", "numpy", "polars")
_TOURNAMENT_RUNTIME_KEYS = (*_BENCHMARK_RUNTIME_KEYS, "cvxpy", "clarabel", "scipy")


def _committed_runtime(revision: str, errors: list[str]) -> dict[str, str]:
    """Read the dependency versions and Python series frozen at the indexed commit."""
    root = Path(__file__).resolve().parents[3]
    try:
        lock_bytes = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:uv.lock"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        python_bytes = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:.python-version"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
        python_series = python_bytes.decode("utf-8").strip()
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        errors.append(f"index: historical runtime lock unavailable: {exc}")
        return {}
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", python_series):
        errors.append("index: invalid historical .python-version")
        return {}
    packages = lock.get("package")
    if not isinstance(packages, list):
        errors.append("index: historical uv.lock lacks packages")
        return {}
    runtime = {"python": python_series}
    for package in packages:
        if not isinstance(package, dict):
            errors.append("index: invalid historical locked package")
            return {}
        name, version = package.get("name"), package.get("version")
        if name in _TOURNAMENT_RUNTIME_KEYS[1:]:
            if name in runtime or not isinstance(version, str) or not version:
                errors.append(f"index: invalid or duplicate historical locked package {name}")
                return {}
            runtime[name] = version
    if set(runtime) != set(_TOURNAMENT_RUNTIME_KEYS):
        errors.append("index: historical uv.lock lacks required runtime packages")
        return {}
    return runtime


def _runtime_matches(recorded: Any, expected: Mapping[str, object], *, historical: bool) -> bool:
    """Historical Python patch is unpinned; every package version is pinned by uv.lock."""
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        return False
    if not historical:
        return recorded == expected
    python_version = recorded.get("python")
    python_series = expected.get("python")
    if (
        not isinstance(python_version, str)
        or not isinstance(python_series, str)
        or not re.fullmatch(r"\d+\.\d+\.\d+", python_version)
        or not (
            python_version == python_series
            if python_series.count(".") == 2
            else python_version.startswith(python_series + ".")
        )
    ):
        return False
    return all(recorded[key] == value for key, value in expected.items() if key != "python")


def _honesty(value: dict[str, Any], errors: list[str], label: str, *, report: bool) -> None:
    _assert(value.get("research_only") is True, errors, f"{label}: research_only must be true")
    _assert(value.get("live_pnl_claim") is False, errors, f"{label}: live_pnl_claim must be false")
    if report:
        _assert(value.get("promote") is False, errors, f"{label}: promote must be false")


def _dataset(path: Path, expected: Any, errors: list[str]) -> None:
    if not _sha256(expected):
        errors.append("protocol: invalid dataset_sha256")
        return
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        _assert(digest.hexdigest() == expected, errors, "source dataset SHA-256 mismatch")
    except OSError as exc:
        errors.append(f"source dataset unreadable: {exc}")


def _benchmark_manifest(
    run_dir: Path,
    errors: list[str],
    *,
    committed_code_hashes: dict[str, str] | None = None,
    committed_runtime: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    manifest = _receipt(run_dir / "manifest.json", errors)
    if manifest is None:
        return None
    _assert(manifest.get("schema_version") == 1, errors, "benchmark: unsupported schema")
    _assert(_timestamp(manifest.get("created_at")), errors, "benchmark: invalid timestamp")
    _honesty(manifest, errors, "benchmark manifest", report=False)
    expected_code_sha = (
        committed_code_hashes.get("real_benchmark.py")
        if committed_code_hashes is not None
        else real_benchmark._code_sha()
    )
    _assert(
        manifest.get("code_sha256") == expected_code_sha,
        errors,
        "benchmark: code SHA-256 differs from "
        + ("indexed Git revision" if committed_code_hashes is not None else "this checkout"),
    )
    expected_runtime = (
        {key: committed_runtime.get(key) for key in _BENCHMARK_RUNTIME_KEYS}
        if committed_runtime is not None
        else real_benchmark._runtime()
    )
    _assert(
        _runtime_matches(
            manifest.get("runtime"),
            expected_runtime,
            historical=committed_runtime is not None,
        ),
        errors,
        "benchmark: runtime differs from "
        + ("indexed Git lock" if committed_runtime is not None else "this environment"),
    )
    try:
        raw = manifest["protocol"]
        if not isinstance(raw, dict):
            raise TypeError("protocol is not an object")
        protocol = real_benchmark.BenchmarkProtocol(**raw)
        protocol.validate()
        source = Path(protocol.dataset_path)
        _dataset(
            source if source.is_absolute() else run_dir / source, protocol.dataset_sha256, errors
        )
        _assert(
            manifest.get("holdout_status")
            == (
                "previously_inspected"
                if protocol.holdout_previously_inspected
                else "uninspected_by_declaration"
            ),
            errors,
            "benchmark: holdout disclosure differs from protocol",
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"benchmark: invalid protocol: {exc}")
    _assert(
        isinstance(manifest.get("limitations"), list)
        and all(isinstance(item, str) and item for item in manifest["limitations"]),
        errors,
        "benchmark: limitations missing",
    )
    return manifest


def _benchmark_report(
    run_dir: Path, phase: str, manifest: dict[str, Any], errors: list[str]
) -> None:
    report = _receipt(run_dir / f"{phase}.json", errors)
    if report is None:
        return
    _assert(report.get("phase") == phase, errors, f"{phase}: phase mismatch")
    _assert(_timestamp(report.get("created_at")), errors, f"{phase}: invalid timestamp")
    _assert(
        report.get("manifest_sha256") == manifest["receipt_sha256"],
        errors,
        f"{phase}: manifest link mismatch",
    )
    _honesty(report, errors, phase, report=True)
    _assert(
        report.get("claim") == "fixed_split_forecast_diagnostic",
        errors,
        f"{phase}: invalid research claim",
    )
    for field in ("holdout_status", "limitations"):
        _assert(
            report.get(field) == manifest.get(field),
            errors,
            f"{phase}: {field} differs from manifest",
        )
    scores = report.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(real_benchmark.MODELS):
        errors.append(f"{phase}: matched baseline scores missing")
        return
    counts: set[tuple[int, int]] = set()
    for model, values in scores.items():
        if not isinstance(values, dict):
            errors.append(f"{phase}: invalid {model} scores")
            continue
        for metric in ("date_equal_weight_mse", "date_equal_weight_mae"):
            score = values.get(metric)
            numeric_score = (
                math.isfinite(float(score)) and float(score) >= 0
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else False
            )
            _assert(
                numeric_score,
                errors,
                f"{phase}: invalid {model} {metric}",
            )
        rows, dates = values.get("n_rows"), values.get("n_dates")
        if type(rows) is int and type(dates) is int and rows >= dates > 0:
            counts.add((rows, dates))
        else:
            errors.append(f"{phase}: invalid {model} sample counts")
    _assert(len(counts) == 1, errors, f"{phase}: baseline sample counts differ")
    audit = manifest.get("audit")
    if isinstance(audit, dict) and isinstance(audit.get("split_counts"), dict) and len(counts) == 1:
        rows, dates = next(iter(counts))
        _assert(
            audit["split_counts"].get(phase) == {"rows": rows, "dates": dates},
            errors,
            f"{phase}: scores differ from eligibility audit counts",
        )
    else:
        errors.append("benchmark: invalid eligibility audit")


def _benchmark(
    run_dir: Path,
    errors: list[str],
    *,
    committed_code_hashes: dict[str, str] | None = None,
    committed_runtime: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    manifest = _benchmark_manifest(
        run_dir,
        errors,
        committed_code_hashes=committed_code_hashes,
        committed_runtime=committed_runtime,
    )
    if manifest is not None:
        for phase in ("validation", "test"):
            _benchmark_report(run_dir, phase, manifest, errors)
    return manifest


def _tournament_manifest(
    run_dir: Path,
    errors: list[str],
    *,
    committed_code_hashes: dict[str, str] | None = None,
    committed_runtime: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    manifest = _receipt(run_dir / "manifest.json", errors)
    if manifest is None:
        return None
    _assert(manifest.get("schema_version") == 1, errors, "tournament: unsupported schema")
    _assert(_timestamp(manifest.get("created_at")), errors, "tournament: invalid timestamp")
    _honesty(manifest, errors, "tournament manifest", report=False)
    expected_code_hashes = (
        committed_code_hashes
        if committed_code_hashes is not None
        else net_tournament._code_hashes()
    )
    _assert(
        manifest.get("code_sha256") == expected_code_hashes,
        errors,
        "tournament: code hashes differ from "
        + ("indexed Git revision" if committed_code_hashes is not None else "this checkout"),
    )
    expected_runtime = (
        {key: committed_runtime.get(key) for key in _TOURNAMENT_RUNTIME_KEYS}
        if committed_runtime is not None
        else net_tournament._tournament_runtime()
    )
    _assert(
        _runtime_matches(
            manifest.get("runtime"),
            expected_runtime,
            historical=committed_runtime is not None,
        ),
        errors,
        "tournament: runtime differs from "
        + ("indexed Git lock" if committed_runtime is not None else "this environment"),
    )
    parent = manifest.get("benchmark_run")
    if not isinstance(parent, str) or not parent.strip():
        errors.append("tournament: benchmark_run path missing")
    else:
        parent_dir = run_dir / parent
        parent_errors: list[str] = []
        parent_manifest = _benchmark(
            parent_dir,
            parent_errors,
            committed_code_hashes=committed_code_hashes,
            committed_runtime=committed_runtime,
        )
        errors.extend(f"parent benchmark: {error}" for error in parent_errors)
        if parent_manifest is not None:
            _assert(
                manifest.get("benchmark_manifest") == parent_manifest,
                errors,
                "tournament: embedded benchmark manifest differs from parent",
            )
    try:
        raw = manifest["spec"]
        if not isinstance(raw, dict):
            raise TypeError("spec is not an object")
        _, trials, baseline = net_tournament._spec(raw)
        _assert(
            manifest.get("candidate_count") == len(trials),
            errors,
            "tournament: candidate_count mismatch",
        )
        _assert(
            manifest.get("candidates") == [asdict(t) for t in trials],
            errors,
            "tournament: candidate slate mismatch",
        )
        _assert(
            manifest.get("benchmark") == asdict(baseline), errors, "tournament: baseline mismatch"
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"tournament: invalid specification: {exc}")
    _assert(
        manifest.get("selection_rule")
        == "highest validation mean net return, name ascending for ties",
        errors,
        "tournament: selection rule mismatch",
    )
    _assert(
        manifest.get("scenarios") == {"configured": 1.0, "double_impact": 2.0},
        errors,
        "tournament: impact scenarios mismatch",
    )
    _assert(
        isinstance(manifest.get("limitations"), list)
        and all(isinstance(item, str) and item for item in manifest["limitations"]),
        errors,
        "tournament: limitations missing",
    )
    return manifest


def _tournament_phase(
    run_dir: Path,
    phase: str,
    manifest: dict[str, Any],
    validation: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any] | None:
    candidates = manifest.get("candidates")
    names = (
        [
            item["name"]
            for item in candidates
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if isinstance(candidates, list)
        else []
    )
    benchmark = manifest.get("benchmark")
    baseline = benchmark.get("name") if isinstance(benchmark, dict) else None
    attempt = _receipt(run_dir / f"{phase}.attempt.json", errors)
    if attempt is not None:
        _assert(attempt.get("phase") == phase, errors, f"{phase}: attempt phase mismatch")
        _assert(
            _timestamp(attempt.get("created_at")), errors, f"{phase}: invalid attempt timestamp"
        )
        _assert(
            attempt.get("manifest_sha256") == manifest["receipt_sha256"],
            errors,
            f"{phase}: attempt manifest link mismatch",
        )
        _assert(
            attempt.get("candidates") == names, errors, f"{phase}: attempt candidate slate mismatch"
        )
    report = _receipt(run_dir / f"{phase}.json", errors)
    if report is None:
        return None
    _assert(report.get("phase") == phase, errors, f"{phase}: phase mismatch")
    _assert(_timestamp(report.get("created_at")), errors, f"{phase}: invalid timestamp")
    _assert(
        report.get("manifest_sha256") == manifest["receipt_sha256"],
        errors,
        f"{phase}: manifest link mismatch",
    )
    _honesty(report, errors, phase, report=True)
    _assert(
        report.get("claim") == "simulated_net_price_return_tournament",
        errors,
        f"{phase}: invalid research claim",
    )
    for field in ("holdout_status", "limitations"):
        parent = manifest.get("benchmark_manifest")
        reference = (
            parent.get("holdout_status")
            if field == "holdout_status" and isinstance(parent, dict)
            else manifest.get(field)
        )
        _assert(report.get(field) == reference, errors, f"{phase}: {field} differs from manifest")
    expected_link = validation["receipt_sha256"] if phase == "test" and validation else None
    _assert(
        report.get("validation_receipt_sha256") == expected_link,
        errors,
        f"{phase}: validation link mismatch",
    )
    selected = report.get("selected")
    if phase == "test":
        _assert(
            validation is not None and selected == validation.get("selected"),
            errors,
            "test: validation selection changed",
        )
    elif selected is not None:
        _assert(selected in names, errors, "validation: selected candidate outside frozen slate")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != {"configured", "double_impact"}:
        errors.append(f"{phase}: scenarios missing or changed")
        return report
    complete = True
    liquidated = True
    for scenario, case in scenarios.items():
        if not isinstance(case, dict):
            errors.append(f"{phase}/{scenario}: invalid scenario")
            continue
        outcomes = case.get("trials")
        if not isinstance(outcomes, dict) or set(outcomes) != {baseline, *names}:
            errors.append(f"{phase}/{scenario}: trial slate incomplete")
            complete = False
            liquidated = False
            continue
        for name, outcome in outcomes.items():
            if not isinstance(outcome, dict) or outcome.get("status") not in {
                "completed",
                "failed",
            }:
                errors.append(f"{phase}/{scenario}/{name}: invalid trial status")
                complete = False
                liquidated = False
                continue
            if outcome["status"] == "failed":
                complete = False
                liquidated = False
                _assert(
                    isinstance(outcome.get("error"), str),
                    errors,
                    f"{phase}/{scenario}/{name}: failure reason missing",
                )
            else:
                for key in ("daily", "fills", "rejections", "allocations"):
                    _assert(
                        isinstance(outcome.get(key), list),
                        errors,
                        f"{phase}/{scenario}/{name}: {key} ledger missing",
                    )
                _assert(
                    type(outcome.get("liquidation_complete")) is bool,
                    errors,
                    f"{phase}/{scenario}/{name}: liquidation status missing",
                )
                liquidated &= outcome.get("liquidation_complete") is True
        comparison = case.get("comparison")
        if not isinstance(comparison, dict):
            errors.append(f"{phase}/{scenario}: comparison missing")
        else:
            _assert(
                comparison.get("tested_candidates") == names,
                errors,
                f"{phase}/{scenario}: inference slate mismatch",
            )
            if any(
                outcome.get("status") == "failed"
                for outcome in outcomes.values()
                if isinstance(outcome, dict)
            ):
                _assert(
                    comparison.get("status") == "incomplete_trials",
                    errors,
                    f"{phase}/{scenario}: failed trial hidden from inference",
                )
        _assert(
            isinstance(case.get("allocation_ablations"), list),
            errors,
            f"{phase}/{scenario}: ablations missing",
        )
    _assert(report.get("complete") is complete, errors, f"{phase}: completeness flag mismatch")
    _assert(
        report.get("all_terminal_liquidations_complete") is (complete and liquidated),
        errors,
        f"{phase}: liquidation flag mismatch",
    )
    if phase == "validation":
        configured = scenarios["configured"]
        compared = configured.get("comparison", {}) if isinstance(configured, dict) else {}
        if not complete or not isinstance(compared, dict) or compared.get("status") != "tested":
            _assert(
                selected is None, errors, "validation: selection without completed tested slate"
            )
    _assert(
        type(report.get("selected_holdout_adjusted_rejection")) is bool,
        errors,
        f"{phase}: adjusted rejection flag missing",
    )
    _assert(
        type(report.get("selected_survives_double_impact")) is bool,
        errors,
        f"{phase}: impact stress flag missing",
    )
    evidence = (
        report.get("selected_holdout_adjusted_rejection") is True
        and report.get("all_terminal_liquidations_complete") is True
        and report.get("selected_survives_double_impact") is True
    )
    _assert(
        report.get("economic_evidence_gate") is evidence,
        errors,
        f"{phase}: economic evidence gate mismatch",
    )
    return report


def verify_phase1_run(
    run_dir: Path,
    *,
    committed_code_hashes: dict[str, str] | None = None,
    committed_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Verify a completed run or a tournament blocked by frozen validation."""
    run_dir = Path(run_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
    except (OSError, ValueError, UnicodeError) as exc:
        return {
            "valid": False,
            "path": str(run_dir),
            "kind": None,
            "errors": [f"manifest.json: {exc}"],
        }
    kind = (
        "net_tournament"
        if "benchmark_manifest" in manifest
        else "real_benchmark"
        if "protocol" in manifest
        else None
    )
    state = "complete"
    if kind == "real_benchmark":
        _benchmark(
            run_dir,
            errors,
            committed_code_hashes=committed_code_hashes,
            committed_runtime=committed_runtime,
        )
    elif kind == "net_tournament":
        sealed = _tournament_manifest(
            run_dir,
            errors,
            committed_code_hashes=committed_code_hashes,
            committed_runtime=committed_runtime,
        )
        if sealed is not None:
            validation = _tournament_phase(run_dir, "validation", sealed, None, errors)
            if validation is not None and validation.get("selected") is None:
                state = "blocked"
                scenarios = validation.get("scenarios")
                configured = scenarios.get("configured") if isinstance(scenarios, dict) else None
                comparison = configured.get("comparison") if isinstance(configured, dict) else None
                _assert(
                    validation.get("complete") is False
                    or not isinstance(comparison, dict)
                    or comparison.get("status") != "tested",
                    errors,
                    "validation: null selection lacks a recorded failure or untestable comparison",
                )
                for name in ("test.attempt.json", "test.json"):
                    _assert(
                        not _receipt_exists(run_dir / name),
                        errors,
                        f"blocked tournament: unexpected {name}",
                    )
            else:
                _tournament_phase(run_dir, "test", sealed, validation, errors)
    else:
        errors.append("unsupported Phase-1 manifest")
    return {
        "valid": not errors,
        "path": str(run_dir),
        "kind": kind,
        "state": state,
        "errors": errors,
    }


def _path_is_absolute(value: str) -> bool:
    """True for a POSIX or Windows absolute path, on either host.

    ``Path`` follows the host. ``/abs/run`` is relative on Windows, so an
    index that names it would be accepted there and rejected on Linux.
    """
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def verify_phase1_index(path: Path) -> dict[str, Any]:
    """Verify an index binding run receipts to source configs and code provenance."""
    path = Path(path)
    errors: list[str] = []
    index = _receipt(path, errors)
    if index is None:
        return {
            "valid": False,
            "path": str(path),
            "kind": "phase1_evidence_index",
            "errors": errors,
        }
    _assert(index.get("kind") == "phase1_evidence_index", errors, "index: kind mismatch")
    _assert(index.get("schema_version") == 1, errors, "index: unsupported schema")
    _assert(_timestamp(index.get("created_at")), errors, "index: invalid timestamp")
    _assert(
        _sha256(index.get("git_revision"), length=40),
        errors,
        "index: git_revision must be a commit SHA",
    )
    _assert(_sha256(index.get("git_worktree_sha256")), errors, "index: git_worktree_sha256 missing")
    _assert(
        index.get("benchmark_catalog_version") == BENCHMARK_CATALOG_VERSION,
        errors,
        "index: benchmark catalog version differs from this checkout",
    )
    revision = index.get("git_revision")
    committed = (
        _committed_code_hashes(revision, errors)
        if isinstance(revision, str) and _sha256(revision, length=40)
        else {}
    )
    committed_runtime = (
        _committed_runtime(revision, errors)
        if isinstance(revision, str) and _sha256(revision, length=40)
        else {}
    )
    same_dirty_checkout = (
        revision == git_revision() and index.get("git_worktree_sha256") == git_worktree_sha256()
    )
    runs = index.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append("index: runs must be a nonempty list")
        return {
            "valid": False,
            "path": str(path),
            "kind": "phase1_evidence_index",
            "errors": errors,
        }
    seen: set[tuple[str, str]] = set()
    for number, entry in enumerate(runs):
        label = f"index run {number}"
        if not isinstance(entry, dict):
            errors.append(f"{label}: entry is not an object")
            continue
        kind, run_path, config_path = (
            entry.get("kind"),
            entry.get("path"),
            entry.get("config_path"),
        )
        if kind not in {"real_benchmark", "net_tournament"}:
            errors.append(f"{label}: unknown run kind")
            continue
        if not isinstance(run_path, str) or not run_path or _path_is_absolute(run_path):
            errors.append(f"{label}: run path must be relative to index")
            continue
        if not isinstance(config_path, str) or not config_path or _path_is_absolute(config_path):
            errors.append(f"{label}: config path must be relative to index")
            continue
        identity = (kind, run_path)
        _assert(identity not in seen, errors, f"{label}: duplicate run")
        seen.add(identity)
        run_dir = (path.parent / run_path).resolve()
        config_file = (path.parent / config_path).resolve()
        result = verify_phase1_run(
            run_dir,
            committed_code_hashes=committed if committed and not same_dirty_checkout else None,
            committed_runtime=committed_runtime if not same_dirty_checkout else None,
        )
        errors.extend(f"{label}: {error}" for error in result["errors"])
        _assert(result["kind"] == kind, errors, f"{label}: run kind mismatch")
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("validation.json", "validation_sha256"),
            ("test.json", "test_sha256"),
        ):
            recorded = entry.get(field)
            if field == "test_sha256" and recorded is None and result.get("state") == "blocked":
                _assert(
                    not _receipt_exists(run_dir / filename),
                    errors,
                    f"{label}: blocked run contains test receipt",
                )
                continue
            if not _sha256(recorded):
                errors.append(f"{label}: {field} missing")
                continue
            linked = _receipt(run_dir / filename, errors)
            if linked is not None:
                _assert(
                    linked["receipt_sha256"] == recorded, errors, f"{label}: {field} link mismatch"
                )
        expected_config = entry.get("config_sha256")
        if not _sha256(expected_config):
            errors.append(f"{label}: config_sha256 missing")
            continue
        try:
            config_bytes = config_file.read_bytes()
            _assert(
                hashlib.sha256(config_bytes).hexdigest() == expected_config,
                errors,
                f"{label}: config SHA-256 mismatch",
            )
            config = json.loads(
                config_bytes,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
            manifest = _receipt(run_dir / "manifest.json", errors)
            if manifest is None or not isinstance(config, dict):
                raise ValueError("config or run manifest is not an object")
            source_hashes = (
                {"real_benchmark.py": manifest.get("code_sha256")}
                if kind == "real_benchmark"
                else manifest.get("code_sha256")
            )
            if not isinstance(source_hashes, dict):
                errors.append(f"{label}: source code hashes missing")
            elif committed and not same_dirty_checkout:
                for name, digest in source_hashes.items():
                    _assert(
                        committed.get(name) == digest,
                        errors,
                        f"{label}: {name} differs from indexed git revision",
                    )
            elif not committed:
                errors.append(f"{label}: indexed git revision unavailable")
            if kind == "net_tournament":
                _assert(
                    config == manifest.get("spec"),
                    errors,
                    f"{label}: config differs from frozen slate",
                )
            else:
                frozen = manifest.get("protocol")
                if not isinstance(frozen, dict):
                    raise ValueError("frozen protocol is not an object")
                declared = config.get("dataset_path")
                actual = frozen.get("dataset_path")
                if not isinstance(declared, str) or not isinstance(actual, str):
                    raise ValueError("dataset path missing")
                frozen_source = Path(actual)
                frozen_source = (
                    frozen_source if frozen_source.is_absolute() else run_dir / frozen_source
                )
                _assert(
                    (config_file.parent / declared).resolve() == frozen_source.resolve(),
                    errors,
                    f"{label}: config dataset path differs from frozen dataset",
                )
                declared_protocol = asdict(real_benchmark.BenchmarkProtocol(**config))
                _assert(
                    {k: v for k, v in declared_protocol.items() if k != "dataset_path"}
                    == {k: v for k, v in frozen.items() if k != "dataset_path"},
                    errors,
                    f"{label}: config differs from frozen protocol",
                )
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            errors.append(f"{label}: config unavailable or invalid: {exc}")
    return {
        "valid": not errors,
        "path": str(path),
        "kind": "phase1_evidence_index",
        "errors": errors,
    }
