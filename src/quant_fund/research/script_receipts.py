"""Contract checks for script-emitted receipt schemas.

The ``scripts/`` lanes write ``receipts/*.json`` evidence artifacts with their
own ``schema`` tags (``adaptive_mix_replay.v1``, ``basis_reversion_screen.v1``,
``incumbent_bench.v1``, ``basis_pair_candidate.v1``,
``adaptive_mix_band_search.v1``, ``fx1.dip_bench/v1``). These are research
receipts: every check below is a *re-derivation or bound* computed purely from
the sealed body — a headline scalar that disagrees with the embedded detail is
forged or mislabeled, and the verifier must say so.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _finite_number(value: object) -> bool:
    return _as_finite_float(value) is not None


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fraction(value: object) -> bool:
    f = _as_finite_float(value)
    return f is not None and 0.0 <= f <= 1.0


def _segment_metrics(metrics: object) -> list[str]:
    """Required segment statistics block: n + finite return/drawdown fields."""
    if not isinstance(metrics, Mapping):
        return ["segment_metrics_not_object"]
    errors: list[str] = []
    if not _positive_int(metrics.get("n")):
        errors.append("segment_n_not_positive_int")
    for key in ("net_return", "cagr", "sharpe", "max_drawdown"):
        if not _finite_number(metrics.get(key)):
            errors.append(f"segment_{key}_not_finite")
    return errors


def _allocation_simplex_errors(name: str, weights: object, sleeves: set[str]) -> list[str]:
    if not isinstance(weights, Mapping):
        return [f"{name}:allocation_not_object"]
    errors: list[str] = []
    total = 0.0
    for sleeve, w in weights.items():
        if sleeve not in sleeves:
            errors.append(f"{name}:unknown_sleeve:{sleeve}")
        if not _fraction(w):
            errors.append(f"{name}:weight_out_of_bounds:{sleeve}")
        else:
            total += float(w)
    if not errors and not (1.0 - 1e-6 <= total <= 1.0 + 1e-6):
        errors.append(f"{name}:allocation_not_simplex:{total}")
    return errors


def band_search_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``adaptive_mix_band_search.v1``: re-derive eligibility and the selected band.

    The embedded ``selection_rule`` is executable: a candidate is eligible iff
    ``cagr > 0``, ``sharpe > 0`` and ``max_drawdown > -0.05`` on development;
    ``selected_band`` must equal the eligible candidate with the highest
    development Sharpe, or ``null`` when none qualify.
    """
    errors: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ["candidates_missing"]
    eligible: list[tuple[float, object]] = []
    for i, cand in enumerate(candidates):
        if not isinstance(cand, Mapping):
            errors.append(f"candidate_{i}_not_object")
            continue
        if not _finite_number(cand.get("band")):
            errors.append(f"candidate_{i}_band_not_finite")
        errors.extend(f"candidate_{i}:{e}" for e in _segment_metrics(cand.get("development")))
        dev = cand.get("development") or {}
        if isinstance(dev, Mapping) and all(
            _finite_number(dev.get(k)) for k in ("cagr", "sharpe", "max_drawdown")
        ):
            cagr = _as_finite_float(dev.get("cagr"))
            sharpe = _as_finite_float(dev.get("sharpe"))
            drawdown = _as_finite_float(dev.get("max_drawdown"))
            if cagr is None or sharpe is None or drawdown is None:
                continue
            is_eligible = cagr > 0.0 and sharpe > 0.0 and drawdown > -0.05
            if cand.get("eligible") is not is_eligible:
                errors.append(f"candidate_{i}_eligible_mismatch")
            if is_eligible:
                eligible.append((sharpe, cand.get("band")))
    if not errors:
        expected = max(eligible, key=lambda t: t[0])[1] if eligible else None
        if payload.get("selected_band") != expected:
            errors.append("selected_band_not_argmax_dev_sharpe")
    return errors


def adaptive_mix_replay_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``adaptive_mix_replay.v1``: sleeve set agreement + simplex allocations."""
    errors: list[str] = []
    sleeves_obj = payload.get("paper_sleeves")
    if not isinstance(sleeves_obj, Mapping) or not sleeves_obj:
        return ["paper_sleeves_missing"]
    sleeves = set(sleeves_obj)
    for sleeve, seg in sleeves_obj.items():
        if not isinstance(seg, Mapping) or not seg:
            errors.append(f"paper_sleeve_{sleeve}_empty")
            continue
        for seg_name, metrics in seg.items():
            errors.extend(
                f"paper_sleeve_{sleeve}.{seg_name}:{e}" for e in _segment_metrics(metrics)
            )
    alloc = payload.get("mean_adaptive_allocation")
    if not isinstance(alloc, Mapping) or not alloc:
        errors.append("mean_adaptive_allocation_missing")
    else:
        for seg_name, weights in alloc.items():
            errors.extend(_allocation_simplex_errors(str(seg_name), weights, sleeves))
    if not _positive_int(payload.get("n_assets")):
        errors.append("n_assets_not_positive_int")
    if not _positive_int(payload.get("bar_count")):
        errors.append("bar_count_not_positive_int")
    return errors


def basis_reversion_screen_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``basis_reversion_screen.v1``: bounded fractions + four fold means."""
    errors: list[str] = []
    for key in ("positive_day_fraction", "hedged_pair_active_date_fraction"):
        if not _fraction(payload.get(key)):
            errors.append(f"{key}_out_of_bounds")
    for key in ("four_chronological_fold_means", "hedged_pair_four_fold_means"):
        folds = payload.get(key)
        if (
            not isinstance(folds, list)
            or len(folds) != 4
            or not all(_finite_number(v) for v in folds)
        ):
            errors.append(f"{key}_not_four_finite_means")
    for key in ("n_assets", "n_dates"):
        if not _positive_int(payload.get(key)):
            errors.append(f"{key}_not_positive_int")
    for key in (
        "mean_daily_gross_spread",
        "median_daily_gross_spread",
        "hedged_pair_mean_daily_gross_return",
    ):
        if not _finite_number(payload.get(key)):
            errors.append(f"{key}_not_finite")
    return errors


def basis_pair_candidate_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``basis_pair_candidate.v1``: development metrics + eligibility flag."""
    errors: list[str] = []
    if not isinstance(payload.get("development_eligible"), bool):
        errors.append("development_eligible_not_bool")
    dev = payload.get("development")
    if not isinstance(dev, Mapping):
        errors.append("development_missing")
    else:
        for key in ("total_return", "cagr", "sharpe", "max_drawdown"):
            if not _finite_number(dev.get(key)):
                errors.append(f"development_{key}_not_finite")
    if not _positive_int(payload.get("n_target_events")):
        errors.append("n_target_events_not_positive_int")
    if not _positive_int(payload.get("n_assets")):
        errors.append("n_assets_not_positive_int")
    return errors


def incumbent_bench_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``incumbent_bench.v1``: re-derive the reported medians from the raw runs."""
    errors: list[str] = []
    latency = payload.get("latency")
    if not isinstance(latency, Mapping):
        return ["latency_missing"]
    for side in ("dipcatcher", "qlib"):
        runs = latency.get(f"{side}_ms_all")
        median = latency.get(f"{side}_ms_median")
        if not isinstance(runs, list) or not runs or not all(_finite_number(v) for v in runs):
            errors.append(f"{side}_ms_all_not_finite_list")
            continue
        if not _finite_number(median):
            errors.append(f"{side}_ms_median_not_finite")
            continue
        expected = statistics.median(_as_finite_float(v) or 0.0 for v in runs)
        claimed = _as_finite_float(median)
        if claimed is None or abs(claimed - expected) > 0.51:
            errors.append(f"{side}_ms_median_rederive_mismatch")
    correctness = payload.get("correctness")
    if not isinstance(correctness, Mapping):
        errors.append("correctness_missing")
    else:
        if not isinstance(correctness.get("common_dates"), int) or correctness["common_dates"] < 0:
            errors.append("common_dates_not_nonneg_int")
        for key, value in correctness.items():
            if isinstance(value, float) and not _finite_number(value):
                errors.append(f"correctness_{key}_not_finite")
    for key in ("environment", "incumbent", "workload"):
        if not isinstance(payload.get(key), Mapping):
            errors.append(f"{key}_missing")
    return errors


def dip_bench_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``fx1.dip_bench/v1``: n_events must equal the sum of per-asset events."""
    errors: list[str] = []
    per_asset = payload.get("per_asset")
    if not isinstance(per_asset, Mapping) or not per_asset:
        return ["per_asset_missing"]
    total = 0
    for asset, info in per_asset.items():
        if not isinstance(info, Mapping):
            errors.append(f"per_asset_{asset}_not_object")
            continue
        bars, events = info.get("bars"), info.get("events")
        if not _positive_int(bars):
            errors.append(f"per_asset_{asset}_bars_invalid")
        if not isinstance(events, int) or isinstance(events, bool) or events < 0:
            errors.append(f"per_asset_{asset}_events_invalid")
            continue
        if isinstance(bars, int) and not isinstance(bars, bool) and events > bars:
            errors.append(f"per_asset_{asset}_events_exceed_bars")
        total += events
    n_events = payload.get("n_events")
    if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
        errors.append("n_events_invalid")
    elif not errors and n_events != total:
        errors.append("n_events_not_sum_of_per_asset")
    baseline = payload.get("baseline_recovery")
    if not isinstance(baseline, Mapping) or not all(_fraction(v) for v in baseline.values()):
        errors.append("baseline_recovery_not_fractions")
    params = payload.get("params")
    if isinstance(params, Mapping):
        horizons = params.get("horizons_bars")
        if not isinstance(horizons, Mapping) or not all(
            _positive_int(v) for v in horizons.values()
        ):
            errors.append("horizons_bars_not_positive_ints")
        if not _fraction(params.get("threshold")) or float(params.get("threshold", 1)) >= 1.0:
            errors.append("threshold_out_of_bounds")
    else:
        errors.append("params_missing")
    return errors


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def deps_hygiene_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``deps_hygiene.v1``: the audit's internal claims must be coherent.

    The artifacts themselves aren't committed, so the check binds what is
    re-derivable from the sealed body: digest shapes, finding counts that
    are non-negative ints, scanned/audited denominators that are positive,
    and the lock-check's agreement with the duplicate-name count.
    """
    errors: list[str] = []
    artifact_hashes = payload.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        errors.append("artifact_sha256_missing")
    else:
        for name, digest in artifact_hashes.items():
            if not _is_hex64(digest):
                errors.append(f"artifact_sha256:{name}_not_hex64")
    base_commit = payload.get("base_commit")
    if not (
        isinstance(base_commit, str)
        and len(base_commit) == 40
        and all(c in "0123456789abcdef" for c in base_commit)
    ):
        errors.append("base_commit_not_git_sha")
    for key in ("ci_gates", "commands", "environment"):
        if not isinstance(payload.get(key), Mapping) or not payload[key]:
            errors.append(f"{key}_missing")
    docs = payload.get("docs")
    if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
        errors.append("docs_not_string_list")
    results = payload.get("results")
    if not isinstance(results, Mapping) or not results:
        errors.append("results_missing")
        return errors
    count_keys = (
        "vulnerabilities",
        "leaks",
        "severity_medium_or_high_findings",
        "gpl_family_runtime",
        "duplicate_names_in_export",
        "adverse_statuses",
        "eval_exec_hits",
        "findings_total",
    )
    denominator_keys = (
        "audited_packages",
        "audited_dependencies",
        "installed_scanned",
        "pins_covered",
        "loc",
        "bytes_scanned",
        "commits_scanned",
    )
    for tool, block in results.items():
        if not isinstance(block, Mapping):
            errors.append(f"results:{tool}_not_object")
            continue
        for key, value in block.items():
            if key in count_keys and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                errors.append(f"results:{tool}.{key}_not_nonneg_int")
            if key in denominator_keys and not _positive_int(value):
                errors.append(f"results:{tool}.{key}_not_positive")
    pin_audit = results.get("pin_audit")
    if (
        isinstance(pin_audit, Mapping)
        and pin_audit.get("uv_lock_check") == "pass"
        and pin_audit.get("duplicate_names_in_export") != 0
    ):
        errors.append("pin_audit:pass_with_duplicates")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    return errors


def _divergence_entry_ok(d: object) -> bool:
    if isinstance(d, str):
        return True
    # ``{arm, diverges}`` rows (markout.v1 and later lanes).
    return (
        isinstance(d, Mapping)
        and isinstance(d.get("arm"), str)
        and isinstance(d.get("diverges"), bool)
    )


def measurement_receipt_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-check the wave-21b measurement receipts (tape/sim lanes).

    These receipts share a common envelope — ``kind``, ``data_label``,
    ``research_only``, ``git_revision``, optional ``real``/``sim_arms``/
    ``divergences`` sections — rather than a lane-specific claim
    structure. The contract re-derives the envelope invariants that make
    the sealed body admissible evidence: honesty markers present and
    correctly typed, no forbidden headline metric key at top level, and
    the evidence sections, when present, well-formed.
    """
    from quant_fund.research.catalog.registry import FORBIDDEN_RESEARCH_METRIC_KEYS

    errors: list[str] = []
    if not isinstance(payload.get("kind"), str) or not payload["kind"]:
        errors.append("kind_missing")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    label = payload.get("data_label")
    if label not in ("SYNTHETIC", "MIXED", "REAL"):
        errors.append(f"data_label_bad:{label}")
    rev = payload.get("git_revision")
    if not isinstance(rev, str) or not rev:
        errors.append("git_revision_missing")
    seal = payload.get("receipt_sha256")
    if (
        not isinstance(seal, str)
        or len(seal) != 64
        or any(c not in "0123456789abcdef" for c in seal)
    ):
        errors.append("receipt_sha256_not_sha256_hex")
    for key in payload:
        if str(key).lower() in FORBIDDEN_RESEARCH_METRIC_KEYS:
            errors.append(f"forbidden_headline_metric:{key}")
    divergences = payload.get("divergences")
    if divergences is not None and (
        not isinstance(divergences, list) or any(not _divergence_entry_ok(d) for d in divergences)
    ):
        errors.append("divergences_bad_entry")
    real = payload.get("real")
    if real is not None and not isinstance(real, Mapping):
        errors.append("real_not_object")
    sim_arms = payload.get("sim_arms")
    if sim_arms is not None and not isinstance(sim_arms, (Mapping, list)):
        errors.append("sim_arms_bad_type")
    return errors


def joint_tune_v2_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Check the frozen v2 numerical/provenance contract; v1 is unchanged."""
    from quant_fund.microstructure.joint_tune_contract import contract_errors

    return measurement_receipt_contract_errors(payload) + contract_errors(payload)


def _sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _git_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    )


def _forbidden_headline_errors(payload: Mapping[str, Any]) -> list[str]:
    from quant_fund.research.catalog.registry import FORBIDDEN_RESEARCH_METRIC_KEYS

    return [
        f"forbidden_headline_metric:{key}"
        for key in payload
        if str(key).lower() in FORBIDDEN_RESEARCH_METRIC_KEYS
    ]


def _lobster_match_errors(recon: Mapping[str, Any]) -> tuple[list[str], float | None, object]:
    """Re-derive ``match_rate`` from ``n_match / n_compared``."""
    errors: list[str] = []
    match_rate = _as_finite_float(recon.get("match_rate"))
    n_compared = recon.get("n_compared")
    n_match = recon.get("n_match")
    counts_ok = (
        match_rate is not None
        and _positive_int(n_compared)
        and isinstance(n_match, int)
        and not isinstance(n_match, bool)
    )
    if not counts_ok:
        errors.append("reconstruction_counts_bad")
    elif match_rate != (n_match / n_compared):
        errors.append("match_rate_not_n_match_over_n_compared")
    return errors, match_rate, n_match


def _lobster_resync_errors(recon: Mapping[str, Any], n_match: object) -> list[str]:
    """Re-derive ``resync_rate`` and the match/resync partition."""
    n_compared = recon.get("n_compared")
    n_resync = recon.get("n_resync_events")
    resync = _as_finite_float(recon.get("resync_rate"))
    resync_ok = (
        isinstance(n_resync, int)
        and not isinstance(n_resync, bool)
        and _positive_int(n_compared)
        and resync is not None
        and resync == (n_resync / n_compared)
    )
    if not resync_ok:
        return ["resync_rate_not_n_resync_over_n_compared"]
    if (
        isinstance(n_match, int)
        and not isinstance(n_match, bool)
        and n_match + n_resync != n_compared
    ):
        return ["compared_events_not_partitioned"]
    return []


def _lobster_engine_flag_errors(
    payload: Mapping[str, Any], recon: Mapping[str, Any], match_rate: float | None
) -> list[str]:
    """``engine_valid_on_emitted`` is the writer's predicate on the counts."""
    max_div = recon.get("max_divergence_len")
    if not isinstance(max_div, int) or isinstance(max_div, bool) or max_div < 0:
        return ["max_divergence_len_bad"]
    if match_rate is None:
        return []
    expected = match_rate > 0.90 and max_div <= 5
    if payload.get("engine_valid_on_emitted") is not expected:
        return ["engine_valid_on_emitted_mismatch"]
    return []


def lobster_replay_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``lobster_replay.v1``: re-derive match rate and the engine-valid flag.

    The sample label is the tape identity (``LOBSTER-{ticker}-…``), not the
    measurement envelope's SYNTHETIC/MIXED/REAL tag. ``engine_valid_on_emitted``
    is the writer's predicate on the embedded reconstruction counts.
    """
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") not in (None, False):
        errors.append("live_pnl_claim_not_false")
    if not _git_sha(payload.get("git_revision")):
        errors.append("git_revision_missing")
    if not _sha256_hex(payload.get("receipt_sha256")) or not _sha256_hex(
        payload.get("tape_sha256")
    ):
        errors.append("sha256_missing")
    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        errors.append("ticker_missing")
    elif payload.get("data_label") != f"LOBSTER-{ticker}-2012-06-21-sample":
        errors.append("data_label_not_lobster_sample")
    recon = payload.get("reconstruction")
    if not isinstance(recon, Mapping):
        return errors + ["reconstruction_missing"]
    match_errors, match_rate, n_match = _lobster_match_errors(recon)
    errors.extend(match_errors)
    errors.extend(_lobster_resync_errors(recon, n_match))
    errors.extend(_lobster_engine_flag_errors(payload, recon, match_rate))
    errors.extend(_forbidden_headline_errors(payload))
    return errors


def queue_priority_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``queue_priority.v1``: headline comparison must copy the arm curves.

    The producer stamps ``code_revision`` (not ``git_revision``) and seals
    ``comparison`` by reading each arm's fill curve and survival table.
    """
    errors: list[str] = []
    if payload.get("data_label") != "SYNTHETIC":
        errors.append("data_label_not_synthetic")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if payload.get("claim") != "synthetic_simulator_diagnostic_only":
        errors.append("claim_not_synthetic_diagnostic")
    if not _git_sha(payload.get("code_revision")):
        errors.append("code_revision_missing")
    if not _sha256_hex(payload.get("receipt_sha256")):
        errors.append("receipt_sha256_not_sha256_hex")
    arms = payload.get("arms")
    comparison = payload.get("comparison")
    if not isinstance(arms, Mapping) or not isinstance(comparison, Mapping):
        return errors + ["arms_or_comparison_missing"]
    errors.extend(_queue_comparison_errors(arms, comparison))
    errors.extend(_forbidden_headline_errors(payload))
    return errors


def _queue_comparison_errors(arms: Mapping[str, Any], comparison: Mapping[str, Any]) -> list[str]:
    """Each comparison headline must copy the named arm curve field."""
    errors: list[str] = []
    pairs = (
        ("iid", "p_queue_ahead_zero_iid", "fill_curve", "p_queue_ahead_zero"),
        ("markov_regime", "p_queue_ahead_zero_markov", "fill_curve", "p_queue_ahead_zero"),
        ("iid", "p_fill_overall_iid", "queue_survival", "p_fill_overall"),
        ("markov_regime", "p_fill_overall_markov", "queue_survival", "p_fill_overall"),
        ("iid", "mean_fill_wait_seconds_iid", "fill_curve", "mean_fill_wait_seconds"),
        ("markov_regime", "mean_fill_wait_seconds_markov", "fill_curve", "mean_fill_wait_seconds"),
    )
    for arm_name, headline, section, field in pairs:
        arm = arms.get(arm_name)
        block = arm.get(section) if isinstance(arm, Mapping) else None
        detail = block.get(field) if isinstance(block, Mapping) else None
        if not _finite_number(detail) or detail != comparison.get(headline):
            errors.append(f"comparison_mismatch:{headline}")
    return errors


def calibration_audit_drill_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``calibration_audit.drill.v1``: shard label and train/eval echo.

    The drill stamps ``data_label`` from the Yahoo shard (``yahoo_eod``) and
    ``generated_at_commit`` rather than the measurement envelope's
    ``git_revision``. Train and eval lengths must match between ``drill``
    and ``params`` — those are the sizes the e-process actually ran.
    """
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if payload.get("kind") != "calibration_audit.drill.v1":
        errors.append("kind_mismatch")
    if not _git_sha(payload.get("generated_at_commit")):
        errors.append("generated_at_commit_not_sha")
    if not _sha256_hex(payload.get("receipt_sha256")) or not _sha256_hex(
        payload.get("inputs_sha256")
    ):
        errors.append("sha256_missing")
    drill = payload.get("drill")
    params = payload.get("params")
    if not isinstance(drill, Mapping) or not isinstance(params, Mapping):
        return errors + ["drill_or_params_missing"]
    errors.extend(_calibration_shard_errors(payload, drill))
    errors.extend(_calibration_size_errors(payload, drill, params))
    errors.extend(_forbidden_headline_errors(payload))
    return errors


def _calibration_shard_errors(payload: Mapping[str, Any], drill: Mapping[str, Any]) -> list[str]:
    """``data_label`` must echo the Yahoo shard the drill names."""
    shard = drill.get("shard")
    if not isinstance(shard, Mapping):
        return ["shard_missing"]
    errors: list[str] = []
    label = payload.get("data_label")
    if not isinstance(label, str) or not label or label != shard.get("data_label"):
        errors.append("data_label_not_shard_label")
    if shard.get("source") != "yahoo":
        errors.append("shard_source_not_yahoo")
    return errors


def _calibration_size_errors(
    payload: Mapping[str, Any], drill: Mapping[str, Any], params: Mapping[str, Any]
) -> list[str]:
    """Train/eval lengths, model counts, alpha, and the evidence list."""
    errors: list[str] = []
    for key in ("n_train", "n_eval"):
        if drill.get(key) != params.get(key) or not _positive_int(drill.get(key)):
            errors.append(f"{key}_not_echoed")
    if not _positive_int(payload.get("n_models")) or not _positive_int(payload.get("n_shards")):
        errors.append("counts_not_positive_int")
    alpha = _as_finite_float(params.get("alpha"))
    if alpha is None or not 0.0 < alpha <= 1.0:
        errors.append("alpha_not_in_unit_interval")
    evidence = payload.get("evidence")
    evidence_ok = isinstance(evidence, list) and bool(evidence)
    if evidence_ok and any(not isinstance(item, str) or not item for item in evidence):
        evidence_ok = False
    if not evidence_ok:
        errors.append("evidence_bad")
    return errors


#: The wave-21b tape/sim measurement lanes emit ``schema``-tagged receipts
#: with a shared envelope; each registers the measurement contract so no
#: committed receipt verifies on its seal alone.
_MEASUREMENT_SCHEMAS = (
    "abc_calibrate.v1",
    "aftermath_flow.v1",
    "anchor_scan.v1",
    "band_occupancy.v1",
    "band_shape.v1",
    "cancel_cluster.v1",
    "cancel_gradient_bench.v1",
    "churn_reseed.v1",
    "churn_stability.v1",
    "closure_fit.v1",
    "closure_stack.v1",
    "continuation_attr.v1",
    "crown_behind.v1",
    "crown_density.v1",
    "crown_join.v1",
    "crown_size.v1",
    "cxl_shield.v1",
    "deep_book_bench.v1",
    "deep_microprice.v1",
    "depth_consumption.v1",
    "depth_tilt.v1",
    "empirical_flow.v1",
    "hawkes_clock.v1",
    "event_burst.v1",
    "event_granger.v1",
    "event_matrix.v1",
    "forecast_pipeline_audit.v1",
    "exec_cost_real.v1",
    "exec_cost_split.v1",
    "flee_wide.v1",
    "floor_compose.v1",
    "floor_pins.v1",
    "floor_rate.v1",
    "floor_reseed.v1",
    "floor_stability.v1",
    "flow_couple.v1",
    "full_impact.v1",
    "full_stack.v1",
    "gap_close.v1",
    "iid_floor.v1",
    "glft_bench.v1",
    "hawkes_mv.v1",
    "hawkes_real.v1",
    "hidden_depth.v1",
    "hidden_depth_bench.v1",
    "hit_flee.v1",
    "hit_starve.v1",
    "ice_budget.v1",
    "ice_crown.v1",
    "iceberg.v1",
    "impact_instant.v1",
    "impact_persist.v1",
    "improve_flow.v1",
    "instant_decomp.v1",
    "intraday_shape.v1",
    "joint_fit.v1",
    "grammar_map.v1",
    "joint_tune.v1",
    "joint_stability.v1",
    "level_gap.v1",
    "lob_exec.v1",
    "lob_resilience.v1",
    "lo_response.v1",
    "marketable_limit.v1",
    "maker_age.v1",
    "markout.v1",
    "metaorder_detect.v1",
    "mid_dark.v1",
    "mid_jump.v1",
    "order_lifetime.v1",
    "order_revision.v1",
    "mortal_repost.v1",
    "pin_stability.v1",
    "place_law.v1",
    "place_mix.v1",
    "post_trade_drift.v1",
    "price_clustering.v1",
    "price_improvement.v1",
    "propagator_real.v1",
    "quote_floor.v1",
    "quote_place.v1",
    "refill_hazard.v1",
    "regime_clock.v1",
    "release_chase.v1",
    "reload_gate.v1",
    "repost_frontier.v1",
    "repost_latency.v1",
    "reseed_hazard.v1",
    "round_lot.v1",
    "sign_autocorr_real.v1",
    "sign_predict.v1",
    "sim_real_ledger.v1",
    "shield_decay.v1",
    "split_flow.v1",
    "spread_dynamics.v1",
    "spread_floor.v1",
    "spread_reopen.v1",
    "spread_response.v1",
    "spread_response_bench.v1",
    "stale_quote.v1",
    "streak_calibrate.v1",
    "streak_stats.v1",
    "sweep_crown.v1",
    "sweep_width_bench.v1",
    "tape_digest.v1",
    "tick_rule.v1",
    "touch_empty.v1",
    "touch_follow.v1",
    "unhit_chase.v1",
    "vac_chase.v1",
    "vol_signature.v1",
    "vpin.v1",
    "wave23_map.v1",
    "wave24_map.v1",
    "zone_embargo.v1",
    "zone_card.v1",
    "zone_map.v1",
    "zone_churn.v1",
    "zone_ttl.v1",
    "zone_stability.v1",
)


SCRIPT_RECEIPT_CONTRACTS: dict[str, Any] = {
    "adaptive_mix_band_search.v1": band_search_contract_errors,
    "adaptive_mix_replay.v1": adaptive_mix_replay_contract_errors,
    "basis_reversion_screen.v1": basis_reversion_screen_contract_errors,
    "basis_pair_candidate.v1": basis_pair_candidate_contract_errors,
    "incumbent_bench.v1": incumbent_bench_contract_errors,
    "fx1.dip_bench/v1": dip_bench_contract_errors,
    "deps_hygiene.v1": deps_hygiene_contract_errors,
    "joint_tune.v2": joint_tune_v2_contract_errors,
    "lobster_replay.v1": lobster_replay_contract_errors,
    "queue_priority.v1": queue_priority_contract_errors,
    "calibration_audit.drill.v1": calibration_audit_drill_contract_errors,
    **{s: measurement_receipt_contract_errors for s in _MEASUREMENT_SCHEMAS},
}


def script_receipt_contract_errors(schema: object, payload: Mapping[str, Any]) -> list[str]:
    """Re-derive a script receipt's headline claims; ``[]`` when unknown schema.

    A numeric ``schema`` is a format version (the three ``schema: 1`` tape
    receipts). Those dispatch on ``kind``, which is the lane id the writer
    actually stamped.
    """
    key: object = schema if isinstance(schema, str) and schema else payload.get("kind")
    check = SCRIPT_RECEIPT_CONTRACTS.get(str(key))
    if check is None:
        return []
    try:
        result: list[str] = check(payload)
        return result
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError):
        # A contract-check crash must never read as a verified receipt.
        return ["contract_check_error"]
