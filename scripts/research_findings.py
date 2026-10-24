"""Regenerate docs/research/findings.md from recorded trial artifacts.

The page copies figures that are already in trial ledgers, research receipts,
and reality-filter outputs. A missing field stays missing. Draft PR #202 is
named only when its survivorship-corrected batch is not in the tree, and that
row has no measured figures.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

PENDING = "pending, not merged"
NOT_IN_ARTIFACT = "not in artifact"
SURVIVORSHIP_PR = "draft PR #202 survivorship-corrected batch"

_METRIC_COLUMNS = (
    "date",
    "trials",
    "dsr_vs_bar",
    "pbo",
    "best_vs_baselines",
    "verdict",
    "artifact",
)


@dataclass(frozen=True)
class BatchRow:
    """One recorded batch, or the pending row for an unmerged draft."""

    batch: str
    date: str
    trials: str
    dsr_vs_bar: str
    pbo: str
    best_vs_baselines: str
    verdict: str
    artifact: str
    pending: bool = False


def show(value: object) -> str:
    """Render a JSON value without rounding or substituting a default."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return json.dumps(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "[" + ", ".join(show(item) for item in value) + "]"
    if isinstance(value, dict):
        parts: list[str] = []
        for key in sorted(value):
            if isinstance(key, str):
                parts.append(f"{key}: {show(value[key])}")
        return "{" + ", ".join(parts) + "}"
    return NOT_IN_ARTIFACT


def as_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            out[key] = item
    return out


def as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return list(value)
    return None


def lookup(mapping: dict[str, object], key: str) -> tuple[bool, object]:
    if key in mapping:
        return True, mapping[key]
    return False, None


def field(mapping: dict[str, object] | None, key: str) -> str:
    if mapping is None:
        return NOT_IN_ARTIFACT
    present, value = lookup(mapping, key)
    if not present:
        return NOT_IN_ARTIFACT
    return show(value)


def load_json(path: Path) -> object:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            parsed: object = json.load(handle)
            return parsed
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def pending_row() -> BatchRow:
    """Name the unmerged draft without copying any measured figure."""
    return BatchRow(
        batch=SURVIVORSHIP_PR,
        date=PENDING,
        trials=PENDING,
        dsr_vs_bar=PENDING,
        pbo=PENDING,
        best_vs_baselines=PENDING,
        verdict=PENDING,
        artifact=PENDING,
        pending=True,
    )


def _is_survivorship(row: BatchRow) -> bool:
    blob = f"{row.batch} {row.artifact}".lower()
    return "survivorship" in blob and not row.pending


def _baseline_names(prereg: dict[str, object] | None) -> set[str] | None:
    if prereg is None:
        return None
    why = as_dict(prereg.get("why_these_strategies"))
    if why is None:
        return None
    names: set[str] = set()
    for name, text in why.items():
        if isinstance(text, str) and "baseline" in text.lower():
            names.add(name)
    return names


def _trial_periodic_ratio(trial: dict[str, object]) -> tuple[bool, object]:
    windows = as_dict(trial.get("windows"))
    if windows is None:
        return False, None
    validation = as_dict(windows.get("validation"))
    if validation is None:
        return False, None
    return lookup(validation, "periodic_ratio")


def _best_vs_baselines(
    receipt: dict[str, object],
    prereg: dict[str, object] | None,
) -> str:
    gate = as_dict(receipt.get("reality_gate"))
    selected = field(receipt, "selected_trial_id")
    if gate is not None:
        best_present, best_id = lookup(gate, "best_trial_id")
        if best_present and show(best_id) != selected:
            selected = f"{selected} (reality_gate.best_trial_id {show(best_id)})"
    trials = as_list(receipt.get("trials"))
    chosen: dict[str, object] | None = None
    if trials is not None:
        for item in trials:
            trial = as_dict(item)
            if trial is None:
                continue
            if field(trial, "trial_id") == field(receipt, "selected_trial_id"):
                chosen = trial
                break
    if chosen is None:
        return f"selected trial {selected}; trial record {NOT_IN_ARTIFACT}"
    strategy = field(chosen, "strategy")
    present, ratio = _trial_periodic_ratio(chosen)
    ratio_text = show(ratio) if present else NOT_IN_ARTIFACT
    parts = [
        f"gate best trial {selected}",
        f"strategy {strategy}",
        f"validation periodic_ratio {ratio_text}",
    ]
    baselines = _baseline_names(prereg)
    if baselines is None:
        parts.append(f"baseline labels {NOT_IN_ARTIFACT}")
        return "; ".join(parts)
    if strategy in baselines:
        parts.append(f"preregistration calls {strategy} a baseline")
    else:
        parts.append(f"preregistration does not call {strategy} a baseline")
    if trials is None:
        return "; ".join(parts)
    best_ratio: float | None = None
    best_shown = ""
    best_ids: list[str] = []
    best_strategy = ""
    for item in trials:
        trial = as_dict(item)
        if trial is None:
            continue
        name = trial.get("strategy")
        if not isinstance(name, str) or name in baselines:
            continue
        ratio_present, raw_ratio = _trial_periodic_ratio(trial)
        if (
            not ratio_present
            or isinstance(raw_ratio, bool)
            or not isinstance(raw_ratio, (int, float))
        ):
            continue
        numeric = float(raw_ratio)
        trial_id = field(trial, "trial_id")
        if best_ratio is None or numeric > best_ratio:
            best_ratio = numeric
            best_shown = show(raw_ratio)
            best_ids = [trial_id]
            best_strategy = name
        elif numeric == best_ratio:
            best_ids.append(trial_id)
    if best_ratio is None:
        parts.append(f"non-baseline validation periodic_ratio {NOT_IN_ARTIFACT}")
    else:
        ids = ", ".join(best_ids)
        parts.append(
            "highest recorded validation periodic_ratio among trials whose "
            f"strategy is not labeled baseline: {best_shown} ({best_strategy} {ids})"
        )
    return "; ".join(parts)


def _dsr_vs_bar(receipt: dict[str, object], prereg: dict[str, object] | None) -> str:
    gate = as_dict(receipt.get("reality_gate"))
    dsr = field(gate, "dsr")
    bar = NOT_IN_ARTIFACT
    if prereg is not None:
        reality_filter = as_dict(prereg.get("reality_filter"))
        bar = field(reality_filter, "dsr_pass")
    if dsr == NOT_IN_ARTIFACT and bar == NOT_IN_ARTIFACT:
        text = NOT_IN_ARTIFACT
    elif bar == NOT_IN_ARTIFACT:
        text = f"gate dsr {dsr}; bar {NOT_IN_ARTIFACT}"
    elif dsr == NOT_IN_ARTIFACT:
        text = f"gate dsr {NOT_IN_ARTIFACT}; bar {bar}"
    else:
        text = f"gate dsr {dsr} vs bar {bar}"
    raw_present, raw = lookup(receipt, "deflated_probability_raw_count")
    if raw_present:
        text += f"; deflated_probability_raw_count {show(raw)} (stored separately from gate dsr)"
    return text


def _pbo_cell(receipt: dict[str, object]) -> str:
    gate = as_dict(receipt.get("reality_gate"))
    parts = [f"reality_gate.pbo {field(gate, 'pbo')}"]
    present, top = lookup(receipt, "pbo")
    if not present:
        parts.append(f"cscv pbo {NOT_IN_ARTIFACT}")
    else:
        block = as_dict(top)
        if block is None:
            parts.append(f"pbo {show(top)}")
        else:
            parts.append(f"cscv pbo {field(block, 'pbo')}")
    return "; ".join(parts)


def _reality_trials_cell(receipt: dict[str, object], ledger: Path) -> str:
    gate = as_dict(receipt.get("reality_gate"))
    parts = [f"n_trials {field(gate, 'n_trials')}"]
    if gate is not None and "n_effective_trials" in gate:
        parts.append(f"n_effective_trials {field(gate, 'n_effective_trials')}")
    counted = count_jsonl(ledger)
    if counted is None:
        parts.append("sibling trials.jsonl absent")
    else:
        parts.append(f"ledger rows {counted}")
    return "; ".join(parts)


def _reality_verdict(receipt: dict[str, object]) -> str:
    gate = as_dict(receipt.get("reality_gate"))
    parts = [f"verdict {field(gate, 'verdict')}"]
    if "live_pnl_claim" in receipt:
        parts.append(f"live_pnl_claim {field(receipt, 'live_pnl_claim')}")
    return "; ".join(parts)


def reality_row(root: Path, receipt_path: Path) -> BatchRow | None:
    payload = as_dict(load_json(receipt_path))
    if payload is None:
        return None
    schema = payload.get("schema")
    if schema != "dipcatcher.reality_sweep_receipt.v1" and "reality_gate" not in payload:
        return None
    prereg_path = receipt_path.with_name("preregistration.json")
    prereg = as_dict(load_json(prereg_path)) if prereg_path.is_file() else None
    study = payload.get("study_id")
    batch = study if isinstance(study, str) and study else receipt_path.parent.name
    artifacts = [rel(root, receipt_path)]
    if prereg is not None:
        artifacts.append(rel(root, prereg_path))
    ledger = receipt_path.with_name("trials.jsonl")
    if ledger.is_file():
        artifacts.append(rel(root, ledger))
    gate = as_dict(payload.get("reality_gate"))
    return BatchRow(
        batch=batch,
        date=field(gate, "created_utc"),
        trials=_reality_trials_cell(payload, ledger),
        dsr_vs_bar=_dsr_vs_bar(payload, prereg),
        pbo=_pbo_cell(payload),
        best_vs_baselines=_best_vs_baselines(payload, prereg),
        verdict=_reality_verdict(payload),
        artifact="; ".join(artifacts),
    )


def _phase_file(run_dir: Path, phase: str) -> Path | None:
    plain = run_dir / f"{phase}.json"
    packed = run_dir / f"{phase}.json.gz"
    if plain.is_file():
        return plain
    if packed.is_file():
        return packed
    return None


def _benchmark_name(manifest: dict[str, object]) -> str:
    present, value = lookup(manifest, "benchmark")
    if not present:
        return NOT_IN_ARTIFACT
    if isinstance(value, str):
        return value
    block = as_dict(value)
    if block is None:
        return NOT_IN_ARTIFACT
    return field(block, "name")


def _selected_excess(comparison: dict[str, object], selected: object) -> str:
    present, excess = lookup(comparison, "mean_excess_net_return")
    if not present:
        return f"mean_excess_net_return {NOT_IN_ARTIFACT}"
    block = as_dict(excess)
    if block is not None and isinstance(selected, str) and selected in block:
        return f"selected mean_excess_net_return {show(block[selected])}"
    return f"mean_excess_net_return {show(excess)}"


def _selected_stepm(comparison: dict[str, object], selected: object) -> str:
    present, stepm = lookup(comparison, "stepm_adjusted_p")
    if not present:
        return f"stepm_adjusted_p {NOT_IN_ARTIFACT}"
    block = as_dict(stepm)
    if block is not None and isinstance(selected, str) and selected in block:
        return f"selected stepm_adjusted_p {show(block[selected])}"
    return f"stepm_adjusted_p {show(stepm)}"


def _scenario_phrase(
    phase: str, scenario: str, comparison: dict[str, object], selected: object
) -> str:
    bits = [
        f"{phase}/{scenario} status {field(comparison, 'status')}",
        _selected_excess(comparison, selected),
        _selected_stepm(comparison, selected),
        f"reality_check_p {field(comparison, 'reality_check_p')}",
        f"spa_consistent_p {field(comparison, 'spa_consistent_p')}",
    ]
    return "; ".join(bits)


def _tournament_comparison(
    phase_payload: dict[str, object], phase: str, selected: object
) -> list[str]:
    scenarios = as_dict(phase_payload.get("scenarios"))
    if scenarios is None:
        return [f"{phase} scenarios {NOT_IN_ARTIFACT}"]
    phrases: list[str] = []
    for name in sorted(scenarios):
        scenario = as_dict(scenarios[name])
        if scenario is None:
            phrases.append(f"{phase}/{name} {NOT_IN_ARTIFACT}")
            continue
        comparison = as_dict(scenario.get("comparison"))
        if comparison is None:
            phrases.append(f"{phase}/{name} comparison {NOT_IN_ARTIFACT}")
            continue
        phrases.append(_scenario_phrase(phase, name, comparison, selected))
    return phrases


def _configured_trial_names(phase_payload: dict[str, object]) -> str:
    scenarios = as_dict(phase_payload.get("scenarios"))
    if scenarios is None or "configured" not in scenarios:
        return NOT_IN_ARTIFACT
    configured = as_dict(scenarios["configured"])
    if configured is None:
        return NOT_IN_ARTIFACT
    trials = as_dict(configured.get("trials"))
    if trials is None:
        return NOT_IN_ARTIFACT
    names = ", ".join(sorted(trials))
    return f"{len(trials)} ({names})"


def tournament_row(root: Path, manifest_path: Path) -> BatchRow | None:
    manifest = as_dict(load_json(manifest_path))
    if manifest is None or "candidate_count" not in manifest or "benchmark" not in manifest:
        return None
    run_dir = manifest_path.parent
    phases: list[tuple[str, Path, dict[str, object]]] = []
    for phase in ("validation", "test"):
        path = _phase_file(run_dir, phase)
        if path is None:
            continue
        payload = as_dict(load_json(path))
        if payload is not None:
            phases.append((phase, path, payload))
    validation = next((item for item in phases if item[0] == "validation"), None)
    selected_source = validation[2] if validation is not None else phases[0][2] if phases else None
    selected_present = False
    selected: object = None
    if selected_source is not None:
        selected_present, selected = lookup(selected_source, "selected")
    selected_text = show(selected) if selected_present else NOT_IN_ARTIFACT
    trial_names = (
        _configured_trial_names(validation[2]) if validation is not None else NOT_IN_ARTIFACT
    )
    trials = f"candidate_count {field(manifest, 'candidate_count')}; configured trial outcomes {trial_names}"
    if not any(phase == "test" for phase, _path, _payload in phases):
        trials += "; test phase receipt absent"
    comparison_bits: list[str] = []
    verdict_bits = [f"selected {selected_text}"]
    for phase, _path, payload in phases:
        comparison_bits.extend(_tournament_comparison(payload, phase, selected))
        verdict_bits.append(
            f"{phase} economic_evidence_gate {field(payload, 'economic_evidence_gate')}"
        )
        verdict_bits.append(f"{phase} complete {field(payload, 'complete')}")
        if "promote" in payload:
            verdict_bits.append(f"{phase} promote {field(payload, 'promote')}")
        if "live_pnl_claim" in payload:
            verdict_bits.append(f"{phase} live_pnl_claim {field(payload, 'live_pnl_claim')}")
    if not comparison_bits:
        comparison_bits.append(NOT_IN_ARTIFACT)
    verdict_bits.append("no verdict field")
    artifacts = [rel(root, manifest_path)]
    artifacts.extend(rel(root, path) for _phase, path, _payload in phases)
    return BatchRow(
        batch=rel(root, run_dir),
        date=field(manifest, "created_at"),
        trials=trials,
        dsr_vs_bar=NOT_IN_ARTIFACT,
        pbo=NOT_IN_ARTIFACT,
        best_vs_baselines=(f"benchmark {_benchmark_name(manifest)}; " + "; ".join(comparison_bits)),
        verdict="; ".join(verdict_bits),
        artifact="; ".join(artifacts),
    )


def _band_search_row(root: Path, path: Path, payload: dict[str, object]) -> BatchRow | None:
    candidates = as_list(payload.get("candidates"))
    if candidates is None:
        return None
    eligible_values: list[str] = []
    for item in candidates:
        candidate = as_dict(item)
        if candidate is None or "eligible" not in candidate:
            eligible_values.append(NOT_IN_ARTIFACT)
        else:
            eligible_values.append(show(candidate["eligible"]))
    unique_eligible = sorted(set(eligible_values))
    eligible_text = ", ".join(unique_eligible) if unique_eligible else NOT_IN_ARTIFACT
    verdict_bits = [
        "no verdict field",
        f"selected_band {field(payload, 'selected_band')}",
        f"eligible values {eligible_text}",
    ]
    if "live_pnl_claim" in payload:
        verdict_bits.append(f"live_pnl_claim {field(payload, 'live_pnl_claim')}")
    baselines = (
        NOT_IN_ARTIFACT
        if "baseline" not in payload and "baselines" not in payload
        else field(payload, "baseline")
    )
    return BatchRow(
        batch=path.stem,
        date=field(payload, "created_at"),
        trials=f"{len(candidates)} (length of candidates; no n_trials field)",
        dsr_vs_bar=NOT_IN_ARTIFACT,
        pbo=NOT_IN_ARTIFACT,
        best_vs_baselines=(
            f"baseline field {baselines}; selected_band {field(payload, 'selected_band')}"
        ),
        verdict="; ".join(verdict_bits),
        artifact=rel(root, path),
    )


def _overfitting_row(root: Path, path: Path, payload: dict[str, object]) -> BatchRow | None:
    block = as_dict(payload.get("backtest_overfitting"))
    if block is None:
        return None
    return BatchRow(
        batch=path.stem,
        date=field(payload, "created_at"),
        trials=f"n_trials {field(block, 'n_trials')}",
        dsr_vs_bar=f"dsr {field(block, 'dsr')}; bar {NOT_IN_ARTIFACT}",
        pbo=field(block, "pbo"),
        best_vs_baselines=NOT_IN_ARTIFACT,
        verdict=f"metrics_status {field(block, 'metrics_status')}",
        artifact=rel(root, path),
    )


def receipt_row(root: Path, path: Path) -> BatchRow | None:
    payload = as_dict(load_json(path))
    if payload is None:
        return None
    if payload.get("schema") == "dipcatcher.reality_sweep_receipt.v1" or "reality_gate" in payload:
        return reality_row(root, path)
    if "backtest_overfitting" in payload:
        return _overfitting_row(root, path, payload)
    if "candidates" in payload and "selected_band" in payload:
        return _band_search_row(root, path, payload)
    return None


def _note_for_receipt(root: Path, path: Path) -> str:
    payload = as_dict(load_json(path))
    if payload is None:
        return f"`{rel(root, path)}`: not a JSON object"
    schema = payload.get("schema")
    label = schema if isinstance(schema, str) else "no schema"
    return (
        f"`{rel(root, path)}`: {label}; no reality_gate, backtest_overfitting block, "
        "or candidate band search"
    )


def collect_batches(root: Path) -> tuple[list[BatchRow], list[str]]:
    """Read the three artifact classes and return rows plus scan notes."""
    rows: list[BatchRow] = []
    notes: list[str] = []
    seen: set[str] = set()

    studies = root / "research" / "reality"
    pending_ledger = studies / "trials.jsonl"
    if pending_ledger.is_file():
        counted = count_jsonl(pending_ledger)
        notes.append(
            f"`research/reality/trials.jsonl`: present ({counted} non-blank lines); "
            "a study receipt in the same directory is scored instead of this export alone"
        )
    else:
        notes.append("`research/reality/trials.jsonl`: absent")

    if studies.is_dir():
        for receipt_path in sorted(studies.glob("**/receipt.json")):
            row = reality_row(root, receipt_path)
            if row is None:
                notes.append(
                    f"`{rel(root, receipt_path)}`: receipt was not a reality-filter output"
                )
                continue
            rows.append(row)
            seen.add(rel(root, receipt_path))

    metadata = root / "data" / "metadata"
    if metadata.is_dir():
        for manifest_path in sorted(metadata.glob("**/manifest.json")):
            # Windows glob is case-insensitive, so MANIFEST.json would
            # otherwise become a scan note Linux never emits.
            if manifest_path.name != "manifest.json":
                continue
            row = tournament_row(root, manifest_path)
            if row is None:
                notes.append(
                    f"`{rel(root, manifest_path)}`: no candidate_count and benchmark; not a trial batch"
                )
                continue
            rows.append(row)
            seen.add(rel(root, manifest_path))

    receipts = root / "receipts"
    if receipts.is_dir():
        for path in sorted(receipts.glob("*.json")):
            row = receipt_row(root, path)
            if row is None:
                notes.append(_note_for_receipt(root, path))
                continue
            key = rel(root, path)
            if key in seen:
                continue
            rows.append(row)
            seen.add(key)

    if not any(_is_survivorship(row) for row in rows):
        rows.append(pending_row())
        notes.append(
            "draft PR #202 survivorship-corrected batch: not on this tree; "
            "listed as pending, not merged, with no measured figures"
        )
    rows.sort(key=lambda row: (row.pending, row.date, row.batch))
    return rows, notes


def render(root: Path) -> str:
    """Build the findings page for ``root``."""
    rows, notes = collect_batches(root)
    lines = [
        "# Recorded trial batches",
        "",
        "This page is generated by `scripts/research_findings.py` from trial",
        "ledgers, research receipts, and reality-filter outputs already in the",
        "tree. It does not rerun trials. A cell says `not in artifact` when the",
        "cited file has no such field. `null` means the field is present and",
        "empty. Numbers are copied from those files. Nothing here is estimated.",
        "",
        "Research diagnostics only. Not a live-trading claim and not a promotion.",
        "DSR and PBO are overfitting diagnostics. A bar copied from a",
        "preregistration is the bar already stored there.",
        "",
        "Artifacts under `artifacts/` (carry grids, equity-momentum sweeps,",
        "hedge-lab books) are outside those three source classes, so this page",
        "does not score them.",
        "",
        "## Batches",
        "",
        "| batch | date | trials | DSR vs bar | PBO | best candidate vs baselines | verdict | artifact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cells = [
            row.batch,
            row.date,
            row.trials,
            row.dsr_vs_bar,
            row.pbo,
            row.best_vs_baselines,
            row.verdict,
            row.artifact,
        ]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    lines.extend(
        [
            "",
            "## Sources scanned that are not a batch",
            "",
        ]
    )
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def assert_pending_has_no_measured_figures(rows: list[BatchRow]) -> None:
    """The unmerged draft row must not carry a measured figure."""
    for row in rows:
        if row.batch != SURVIVORSHIP_PR:
            continue
        for column in _METRIC_COLUMNS:
            text = getattr(row, column)
            if any(character.isdigit() for character in text):
                raise ValueError(f"pending row {column} contains a digit: {text}")
            if text != PENDING:
                raise ValueError(f"pending row {column} is {text!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate docs/research/findings.md")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    text = render(root)
    rows, _notes = collect_batches(root)
    assert_pending_has_no_measured_figures(rows)
    out = args.out if args.out is not None else root / "docs" / "research" / "findings.md"
    if args.check:
        current = out.read_text(encoding="utf-8") if out.is_file() else ""
        if current != text:
            print(f"{out} does not match scripts/research_findings.py")
            return 1
        print(f"{out} matches")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
