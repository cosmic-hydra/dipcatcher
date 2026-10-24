"""Research command.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import typer

from .app import app
from .support import _cfg, format_data_label, format_fdr_families


def _harvest_p_values(findings: Iterable[Any]) -> list[float]:
    """Pull the ``stat == "p"`` values out of one harvest_findings result.

    Findings are dicts on the corpus-inference lane; attribute access is
    tolerated so the command survives a dataclass-shaped finding.
    """
    out: list[float] = []
    for finding in findings:
        if isinstance(finding, Mapping):
            stat, value = finding.get("stat"), finding.get("value")
        else:
            stat = getattr(finding, "stat", None)
            value = getattr(finding, "value", None)
        if stat == "p" and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


@app.command()
def research(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Run Dipcatcher's proprietary research benches and write a labeled notebook."""
    from quant_fund.research.agent import format_p_value, run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    # Always print DATA_LABEL + FDR families (regression-tested).
    typer.echo(format_data_label(synthetic=bool(nb.synthetic), data_source=str(nb.data_source)))
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(f"{nb.product} — {nb.firm}'s proprietary research lab {nb.version}")
    typer.echo(format_fdr_families(list(nb.hypotheses)))
    typer.echo(nb.disclaimer)
    for h in nb.hypotheses:
        typer.echo(f"{h.id}: {h.decision} (p={format_p_value(h.p_value)})")
    for r in nb.rankers:
        if str(r.get("name", "")).startswith("_"):
            dm_all = r.get("pairwise_dm_all") or []
            for d in dm_all:
                typer.echo(
                    f"DM {d.get('a')} vs {d.get('b')}: "
                    f"stat={d.get('statistic')} p={d.get('p_value')} preferred={d.get('preferred')}"
                )
            continue
        typer.echo(
            f"{r['name']}: IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
            f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
        )
    vol = nb.families.get("volatility", {})
    if vol:
        typer.echo(
            f"volatility QLIKE ewma={vol.get('qlike_ewma')} rolling={vol.get('qlike_rolling')}"
        )
    rl = nb.families.get("reinforcement", {})
    if rl:
        typer.echo(
            f"linucb reward={rl.get('mean_policy_reward')} "
            f"vs_uniform={rl.get('mean_advantage_vs_uniform')} "
            f"regret={rl.get('mean_regret_vs_oracle')}"
        )
    conf = nb.families.get("conformal", {})
    if conf:
        aci = conf.get("aci", {})
        raw = conf.get("gaussian_raw", {})
        wrap = conf.get("wrappee", "scaled_gaussian")
        typer.echo(
            f"conformal wrappee={wrap} "
            f"raw_cov={raw.get('coverage')} "
            f"cqr_raw_cov={conf.get('cqr_raw', {}).get('coverage')} "
            f"scaled_cov={conf.get('scaled', conf.get('scaled_gaussian', {})).get('coverage')} "
            f"cqr_cov={conf.get('cqr', {}).get('coverage')} "
            f"aci_cov={aci.get('coverage')} "
            f"aci_width={aci.get('mean_width')} "
            f"mondrian_cov={conf.get('mondrian_aci', {}).get('coverage')} "
            f"high_x={conf.get('mondrian_aci', {}).get('high_x_coverage')} "
            f"worst_x={conf.get('mondrian_aci', {}).get('worst_x_coverage')}"
        )
    ev = nb.families.get("evalues", {})
    if ev:
        typer.echo(
            f"evalues cov={ev.get('coverage')} e_final={ev.get('e_final')} "
            f"ever_cross={ev.get('ever_cross')}"
        )
    jp = nb.families.get("jackknife_plus", {})
    if jp:
        typer.echo(
            f"jackknife_plus cov={jp.get('coverage')} width={jp.get('mean_width')} "
            f"floor={jp.get('coverage_floor')}"
        )
    crc = nb.families.get("crc", {})
    if crc:
        typer.echo(
            f"crc wrappee={crc.get('wrappee')} risk={crc.get('risk')} "
            f"crc_stat={crc.get('crc_stat')} lambda={crc.get('lambda_hat')} "
            f"high_vol_bound={crc.get('high_vol_mean_bound')} "
            f"low_vol_bound={crc.get('low_vol_mean_bound')}"
        )
    wcqr = nb.families.get("weighted_conformal", {})
    if wcqr:
        typer.echo(
            f"weighted_conformal cov={wcqr.get('coverage')} "
            f"width={wcqr.get('mean_width')} "
            f"unweighted_cov={wcqr.get('unweighted_coverage')}"
        )
    caps = nb.families.get("interval_risk", {})
    if caps:
        typer.echo(
            f"interval_risk mean_cap={caps.get('mean_cap')} "
            f"frac_binding={caps.get('frac_binding')} "
            f"mean_width={caps.get('mean_width')}"
        )
    qb = nb.families.get("quantile_bandit", {})
    if qb:
        typer.echo(
            f"quantile_bandit reward={qb.get('mean_policy_reward')} "
            f"vs_uniform={qb.get('mean_advantage_vs_uniform')} "
            f"regret={qb.get('mean_regret_vs_oracle')}"
        )
    ns = nb.families.get("northset", {})
    if ns:
        typer.echo(
            f"northset ohlc_ok={ns.get('ohlc_identity_rate')} "
            f"book_ok={ns.get('book_uncrossed_rate')} "
            f"session_ok={ns.get('session_reconstructs_daily_rate')} "
            f"imb_ic={ns.get('imbalance_top_mean_ic')} "
            f"ofi_ic={ns.get('ofi_mean_ic')} "
            f"kyle_lambda={ns.get('kyle_lambda')} "
            f"park_qlike={ns.get('parkinson_qlike_vs_cc')} "
            f"gk_qlike={ns.get('garman_klass_qlike_vs_cc')} "
            f"cs_spread={ns.get('corwin_schultz_spread')} "
            f"chain_ok={ns.get('session_chain_rate')} "
            f"jump={ns.get('session_mean_jump_ratio')} "
            f"vpin={ns.get('vpin_mean')} "
            f"sweep_rate={ns.get('sweep_any_rate')} "
            f"sweep_rej_ic={ns.get('sweep_reject_signed_mean_ic')} "
            f"sweep_rej_event={ns.get('sweep_reject_event_mean_bps')}bps "
            f"sweep_follow_event={ns.get('sweep_follow_event_mean_bps')}bps "
            f"sweep_follow_costed={ns.get('sweep_follow_cost_adjusted_mean_bps')}bps "
            f"mean_session_spread_bps_mean={ns.get('mean_session_spread_bps_mean')} "
            f"mean_session_close_spread_bps={ns.get('mean_session_close_spread_bps')} "
            f"mean_session_close_imbalance={ns.get('mean_session_close_imbalance')} "
            f"mean_session_close_micro_bps={ns.get('mean_session_close_micro_bps')} "
            f"mean_session_close_mid={ns.get('mean_session_close_mid')} "
            f"mean_session_close_bid_depth={ns.get('mean_session_close_bid_depth')} "
            f"mean_session_close_ask_depth={ns.get('mean_session_close_ask_depth')} "
            f"mean_session_imbalance_std={ns.get('mean_session_imbalance_std')} "
            f"mean_session_imbalance_mean={ns.get('mean_session_imbalance_mean')} "
            f"session_ofi_sum_mean={ns.get('session_ofi_sum_mean')} "
            f"mean_session_ofi_abs_sum={ns.get('mean_session_ofi_abs_sum')} "
            f"session_book_vpin_mean={ns.get('session_book_vpin_mean')} "
            f"mean_session_book_snaps={ns.get('mean_session_book_snaps')}"
        )
    typer.echo(f"json={nb.artifacts.get('json')}")
    typer.echo(f"markdown={nb.artifacts.get('markdown')}")


@app.command("execution-sensitivity")
def execution_sensitivity_cmd(
    config: Path = typer.Option(Path("configs/backtest.yaml")),
    max_dates: int | None = typer.Option(
        None,
        help=(
            "Use only the first N decision dates present on both the feature "
            "panel and the causal gold panel."
        ),
    ),
    signal_latencies: str = typer.Option(
        "0,1", help="Comma-separated signal-to-order delays in bars."
    ),
    exchange_latencies: str = typer.Option(
        "0,1", help="Comma-separated order-to-exchange delays in bars."
    ),
    impact: str = typer.Option("1,2", help="Comma-separated square-root impact multipliers."),
    seed: int = typer.Option(0, help="Deterministic simulator seed."),
    initial_nav: float = typer.Option(1_000_000.0),
) -> None:
    """Latency/impact grid for one strategy. Execution diagnostic, not a live P&L claim."""
    import polars as pl

    from quant_fund.backtest.event_sim import execution_sensitivity, format_sensitivity_table
    from quant_fund.features.engine import build_features
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel, decision_dates

    def _ints(raw: str) -> tuple[int, ...]:
        parts = tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of bar counts")
        return parts

    def _floats(raw: str) -> tuple[float, ...]:
        parts = tuple(float(piece.strip()) for piece in raw.split(",") if piece.strip())
        if not parts:
            raise typer.BadParameter("expected a comma-separated list of multipliers")
        return parts

    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    feat = build_features(bars, cfg)
    # Gold drops warmup bars (history / label horizon). optimize_asof refuses
    # a decision date with no panel row, so the grid uses the overlap only.
    dates = decision_dates(cfg, feat["event_time"].unique().sort().to_list())
    if len(dates) < 2:
        raise typer.BadParameter(
            "need at least 2 decision dates on both the feature panel and the causal gold panel"
        )
    if max_dates is not None:
        if max_dates < 2:
            raise typer.BadParameter("--max-dates must be at least 2")
        dates = dates[:max_dates]
    feat = feat.filter(pl.col("event_time").is_in(dates))
    weights = build_causal_weight_panel(cfg, dates)
    report = execution_sensitivity(
        feat,
        weights,
        cfg,
        signal_latencies=_ints(signal_latencies),
        exchange_latencies=_ints(exchange_latencies),
        impact_multipliers=_floats(impact),
        initial_nav=initial_nav,
        seed=seed,
    )
    rows = report.get("rows")
    source = cfg.data.source
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        source = str(rows[0].get("source", source))
    synthetic = source.upper() == "SYNTHETIC" or "synthetic" in source.lower()
    typer.echo(format_data_label(synthetic=synthetic, data_source=source))
    if synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(format_sensitivity_table(report))


@app.command("verify-identities")
def verify_identities(
    out: Path = typer.Option(
        Path("data/metadata/research/identity_sweep.json"), "--out", help="Receipt JSON path."
    ),
    trials: int = typer.Option(8, "--trials", help="Seeded SYNTHETIC draws per identity."),
    seed: int = typer.Option(7, "--seed", help="Base seed for the synthetic generators."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = identity_sweep v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Prove catalog/northset microstructure identities on SYNTHETIC draws.

    Prints the identity/residual table and writes an immutable receipt.
    Exits non-zero when any identity's max-abs residual exceeds its
    tolerance — fail-closed, CI gate candidate.
    """
    from quant_fund.research.identity_sweep import (
        format_identity_table,
        run_identity_sweep,
        write_identity_receipt,
    )

    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    receipt = run_identity_sweep(n_trials=int(trials), seed=int(seed))
    write_identity_receipt(out, receipt, receipt_version=receipt_version)
    typer.echo("SYNTHETIC")
    typer.echo(format_identity_table(receipt))
    typer.echo(f"receipt={out}")
    raise typer.Exit(code=0 if receipt["all_passed"] else 1)


@app.command()
def fleet(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = fleet_eval.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC distribution-challenger fleet and write a receipt.

    Proper scores only (pinball/CRPS/PIT-KS/coverage) on labeled synthetic
    shards — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
        run_distribution_fleet,
        write_fleet_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved_shards = resolve_shard_generators(None if shards is None else shards.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = run_distribution_fleet(
        factories,
        resolved_shards,
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        taus=cfg.quantiles.levels,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_fleet_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command("calibration-eval")
def calibration_eval_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    pit_bins: int = typer.Option(10, help="PIT histogram bins (5-50)."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = calibration_eval.v1 (default), "
        "2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC distribution-fleet calibration lane and write a receipt.

    PIT histograms, central-interval coverage, and reliability/calibration
    slopes on the same origins as ``fleet`` — calibration diagnostics only,
    never market or live-P&L claims.
    """
    from quant_fund.research.calibration_eval import (
        run_calibration_eval,
        write_calibration_receipt,
    )
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved_shards = resolve_shard_generators(None if shards is None else shards.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = run_calibration_eval(
        factories,
        resolved_shards,
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        taus=cfg.quantiles.levels,
        pit_bins=pit_bins,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_calibration_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command("verify-receipt")
def verify_receipt_cmd(
    path: Path = typer.Argument(..., help="Receipt JSON file to verify."),
) -> None:
    """Verify a sealed receipt: structure plus hash consistency.

    ``receipt.v2`` envelopes are validated against the pydantic schema and
    their sealed digest, environment fingerprint, code-map digest, and (for
    known kinds) dataset/params digests are re-derived. Older v1 receipts get
    a ``receipt_sha256`` seal check (canonical or strict JSON convention);
    ``fleet_eval.v1`` payloads get their writer's contract too. Exits non-zero
    on any violation — fail-closed.
    """
    import json

    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    typer.echo(json.dumps(result, indent=2))
    raise typer.Exit(code=0 if result["valid"] else 1)


@app.command("verify-all")
def verify_all_cmd(
    receipts_dir: Path = typer.Option(Path("receipts"), help="Receipt directory to audit."),
    check_index: bool = typer.Option(
        True, help="Byte-compare docs/evidence/index.md against a fresh regen."
    ),
    write: bool = typer.Option(True, help="Write the sealed evidence_audit receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Audit receipt output dir."),
) -> None:
    """Chain-of-custody audit over the whole receipts directory.

    Per-file verification via ``verify_receipt_file`` (v2 deep verification,
    v1 seal check); set-level checks a single-file verifier cannot express:
    filename↔digest binding, duplicate seals across files, unsealed-legacy
    accounting, and evidence-index staleness. Writes a sealed
    ``evidence_audit_<digest16>.json`` receipt. Exits non-zero on any hard
    finding — sealed-invalid, unparseable, filename mismatch, duplicate seal,
    or stale index.
    """
    from quant_fund.research.evidence_audit import (
        format_evidence_audit_table,
        run_evidence_audit,
        write_evidence_audit_receipt,
    )

    try:
        rows, receipt = run_evidence_audit(receipts_dir, check_index=check_index, root=Path.cwd())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_evidence_audit_table(rows))
    payload = receipt["payload"]
    typer.echo(
        f"files={payload['n_files']} sealed={payload['n_sealed']} "
        f"unsealed={payload['n_unsealed']} index_fresh={payload['index_fresh']}"
    )
    if payload["findings"]:
        typer.echo("findings: " + "; ".join(payload["findings"]))
    if write:
        path = write_evidence_audit_receipt(receipt, out_dir)
        typer.echo(f"receipt={path}")
    raise typer.Exit(code=0 if receipt["verdict"] == "pass" else 1)


@app.command("replay")
def replay_cmd(
    receipt_path: Path = typer.Argument(
        ..., help="Replay-declared receipt JSON to re-execute and prove."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Proof receipt path (default: receipts/replay_proof_<digest16>.json).",
    ),
    timeout: float = typer.Option(
        120.0, "--timeout", help="Subprocess timeout in seconds for the replayed lane."
    ),
) -> None:
    """Re-execute a receipt's declared lane argv and seal a ``replay_proof.v1`` receipt.

    Reads the optional ``replay`` manifest ``{argv, artifacts, cwd?}``, runs
    argv under the repo root (``dipcatcher``/``quant`` resolve to this
    interpreter's ``quant_fund.cli.main``), re-hashes each declared artifact
    file, and compares observed bytes against the pinned digests. The
    ``replay_proof.v1`` body is wrapped in a sealed ``receipt.v2`` envelope;
    the envelope verdict is pass iff the lane exits 0 AND every artifact
    matches — fail closed on any deviation. Exits non-zero on a fail
    verdict or a non-declared receipt.
    """
    import quant_fund.research.replay_proof as _replay_proof_mod
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text

    root = Path.cwd()
    try:
        body = _replay_proof_mod.run_replay(receipt_path, root=root, timeout_s=timeout)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    document = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(_replay_proof_mod.__file__),),
            verdict=body["verdict"],
            dataset={
                "receipt": body["receipt"],
                "source_receipt_sha256": body["source_receipt_sha256"],
            },
            params={"argv": body["argv"], "timeout_s": body["timeout_s"]},
        )
    )
    digest = str(document["receipt_sha256"])
    out_path = out if out is not None else Path("receipts") / f"replay_proof_{digest[:16]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    typer.echo(
        format_data_label(
            synthetic=body["data_label"] == "SYNTHETIC", data_source=body["data_label"]
        )
    )
    typer.echo(
        f"argv={' '.join(body['argv'])} exit_code={body['exit_code']} "
        f"timed_out={body['timed_out']} elapsed_s={body['elapsed_s']}"
    )
    for row in body["artifacts"]:
        typer.echo(
            f"artifact {row['path']}: match={row['match']} "
            f"expected={row['expected_sha256'][:16]} observed={(row['observed_sha256'] or '-')[:16]}"
        )
    typer.echo(f"all_match={body['all_match']} verdict={body['verdict']}")
    typer.echo(f"receipt={out_path}")
    raise typer.Exit(code=0 if body["verdict"] == "pass" else 1)


@app.command("replay-all")
def replay_all_cmd(
    carriers_dir: Path = typer.Option(
        Path("data/manifests/replay"),
        "--dir",
        help="Directory of replay_manifest.v1 carriers to sweep.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Coverage receipt path (default: receipts/replay_coverage_<digest16>.json).",
    ),
    timeout: float = typer.Option(
        120.0, "--timeout", help="Subprocess timeout in seconds per replayed lane."
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit 1 on any failed replay, or if nothing ran (all skipped). "
        "Carriers skipped for unverified inputs (e.g. tape absent in CI) "
        "are tolerated but reported.",
    ),
) -> None:
    """Sweep every committed replay carrier and seal a ``replay_coverage.v1``.

    Runs ``run_replay`` over each ``replay_manifest.v1`` in the carriers
    directory, aggregates per-carrier verdicts and claim-equality counts
    into one sealed coverage receipt — the corpus's single reproducibility
    number. ``skipped`` (inputs unverified / committed-overwrite guard)
    is honest, never counted as a failure; ``fail`` means the lane ran
    and diverged. ``--strict`` exits non-zero when any carrier fails or
    when every carrier skipped — an all-skip run proves nothing.
    """
    import quant_fund.research.replay_sweep as _replay_sweep_mod
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text

    root = Path.cwd()
    try:
        body = _replay_sweep_mod.run_replay_sweep(carriers_dir, root=root, timeout_s=timeout)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    document = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(_replay_sweep_mod.__file__),),
            verdict=body["verdict"],
            dataset={"carriers_dir": body["carriers_dir"]},
            params={"timeout_s": body["timeout_s"]},
        )
    )
    digest = str(document["receipt_sha256"])
    out_path = out if out is not None else Path("receipts") / f"replay_coverage_{digest[:16]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    for row in body["rows"]:
        typer.echo(f"carrier {row['carrier']}: verdict={row['verdict']}")
    typer.echo(
        f"carriers={body['n_carriers']} pass={body['n_pass']} "
        f"fail={body['n_fail']} skipped={body['n_skipped']} "
        f"coverage={body['coverage']:.4f} "
        f"claims_byte_equal={body['n_claims_byte_equal']}/{body['n_claims_total']}"
    )
    typer.echo(f"verdict={body['verdict']} receipt={out_path}")
    if strict and (body["n_fail"] > 0 or body["n_pass"] == 0):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0 if body["verdict"] == "pass" else 1)


@app.command("vol-bench")
def vol_bench(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None,
        help="Comma-separated vol model names (default: rv_roll, rv_ewma, har, "
        "realized_garch, dip_garch_t).",
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all SYNTHETIC vol shards)."
    ),
    horizons: str = typer.Option(
        "1,5", "--horizons", help="Comma-separated forecast horizons in bars."
    ),
    min_history: int = typer.Option(300, help="Leading fit bars per origin."),
    n_origins: int = typer.Option(24, help="Scored origins per shard/horizon (>=10)."),
    stride: int | None = typer.Option(None, help="Origin spacing in bars (default: max horizon)."),
    n_bars: int | None = typer.Option(
        None, help="Shard length (default: minimal for the origin schedule)."
    ),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = vol_bench.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Run the SYNTHETIC volatility bench and write a sealed receipt.

    QLIKE + MSE on next-bar/h-step cumulative realized variance for HAR,
    realized-GARCH, dip_garch_t and RV baselines over seeded synthetic vol
    shards (GARCH clustering, rough vol, structural breaks). Proper scores
    only — correctness evidence, never market or live-P&L claims.
    """
    from quant_fund.research.vol_bench import (
        resolve_vol_models,
        resolve_vol_shard_generators,
        run_vol_bench,
        write_vol_bench_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        horizon_set = tuple(int(h.strip()) for h in horizons.split(",") if h.strip())
        forecasters = resolve_vol_models(None if models is None else models.split(","))
        resolved_shards = resolve_vol_shard_generators(
            None if shards is None else shards.split(",")
        )
        frame, receipt = run_vol_bench(
            forecasters,
            resolved_shards,
            horizons=horizon_set,
            min_history=int(min_history),
            n_origins=int(n_origins),
            stride=None if stride is None else int(stride),
            n_bars=None if n_bars is None else int(n_bars),
            seed=int(base_seed),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_vol_bench_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(frame)
    typer.echo(f"receipt={path}")


@app.command()
def rankic(
    panels: str | None = typer.Option(
        None, help="Comma-separated panel names (default: all synthetic panels)."
    ),
    challengers: str | None = typer.Option(
        None, help="Comma-separated challenger names (default: all transforms)."
    ),
    n_assets: int = typer.Option(32, help="Assets per date."),
    n_dates: int = typer.Option(96, help="Panel length in dates."),
    horizons: str = typer.Option("1,5,20", help="Comma-separated forward horizons."),
    seed: int = typer.Option(11, help="Base seed for the synthetic panels."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = cross_sectional_rankic.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Cross-sectional rank-IC bench on SYNTHETIC planted-signal panels (P3.4).

    Per-date Spearman rank-IC between each challenger transform and h-step
    forward returns, summarized with a Newey-West mean-IC t-stat. Proper-score
    framing only — never a P&L or live-trading claim.
    """
    from quant_fund.research.cross_sectional import (
        format_rankic_table,
        resolve_panels,
        run_cross_sectional_bench,
        write_rankic_receipt,
    )

    def _names(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        return [piece.strip() for piece in raw.split(",") if piece.strip()]

    try:
        resolved = resolve_panels(_names(panels))
        horizon_tuple = tuple(int(piece.strip()) for piece in horizons.split(",") if piece.strip())
        if not horizon_tuple:
            raise typer.BadParameter("--horizons must be nonempty")
        frame, receipt = run_cross_sectional_bench(
            resolved,
            challengers=_names(challengers),
            horizons=horizon_tuple,
            n_assets=n_assets,
            n_dates=n_dates,
            seed=seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_rankic_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(format_rankic_table(frame))
    typer.echo(f"receipt={path}")


@app.command()
def capacity(
    books: str | None = typer.Option(
        None, help="Comma-separated synthetic book names (default: all)."
    ),
    n_dates: int = typer.Option(126, help="Dates per synthetic book."),
    n_names: int = typer.Option(32, help="Names per synthetic book."),
    seed: int = typer.Option(11, help="Base seed."),
    aum_grid: str = typer.Option(
        "1e6,1e7,5e7,1e8,5e8,1e9", help="Comma-separated AUM levels in dollars."
    ),
    participation_cap: float = typer.Option(
        0.10, help="Max share of dollar ADV a rebalance may consume per name-day."
    ),
    impact_coeff: float = typer.Option(0.1, help="Square-root impact coefficient."),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = capacity_overlay.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """P5.6 participation-capacity bench on SYNTHETIC books (dev-only).

    Feasibility fractions, days-to-trade, and sqrt-impact cost in bps per
    book x AUM cell. Sealed `capacity_overlay_eval` receipt. Size bounds
    only — never a market or live-P&L claim.
    """
    if not dev:
        raise typer.BadParameter(
            "capacity is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.capacity_overlay import (
        format_capacity_table,
        resolve_books,
        run_capacity_bench,
        write_capacity_receipt,
    )

    try:
        grid = tuple(float(x) for x in aum_grid.split(","))
        resolved = resolve_books(None if books is None else books.split(","))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    book_objs = [gen(n_dates, n_names, seed + i) for i, gen in enumerate(resolved.values())]
    frame, receipt = run_capacity_bench(
        book_objs,
        seed=seed,
        aum_grid=grid,
        participation_cap=participation_cap,
        impact_coeff=impact_coeff,
    )
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    path = write_capacity_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(format_capacity_table(frame))
    typer.echo(f"receipt={path}")


@app.command()
def pairs(
    n_assets: int = typer.Option(8, help="Assets in the synthetic panel."),
    n_dates: int = typer.Option(600, help="Panel length in dates."),
    seed: int = typer.Option(0, help="Seed for the synthetic panel."),
    window: int = typer.Option(120, help="Trailing hedge-ratio window."),
    z_window: int = typer.Option(60, help="Trailing z-score window."),
    min_corr: float = typer.Option(0.5, help="Correlation pre-filter threshold."),
    alpha: float = typer.Option(0.05, help="BH rejection level."),
    entry: float = typer.Option(2.0, help="Z-score entry band."),
    exit_band: float = typer.Option(0.5, help="Z-score exit band."),
    hedge_method: str = typer.Option("ols", help="Hedge estimator: ols or kalman."),
    eval_horizon: int = typer.Option(1, help="Forward spread-change horizon."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Stat-arb pairs screen + PIT signal eval on a SYNTHETIC planted panel.

    Engle-Granger residual-ADF screen with a correlation pre-filter and
    BH/Bonferroni multiple-testing control, then a point-in-time z-score
    signal scored against forward spread changes. Proper-score framing
    only — detection truth and IC alignment, never a P&L claim.
    """
    from quant_fund.research.pairs import (
        format_pairs_table,
        run_pairs_eval,
        write_pairs_receipt,
    )

    try:
        frame, receipt = run_pairs_eval(
            seed=seed,
            n_assets=n_assets,
            n_dates=n_dates,
            window=window,
            z_window=z_window,
            min_corr=min_corr,
            alpha=alpha,
            entry=entry,
            exit=exit_band,
            hedge_method=hedge_method,
            eval_horizon=eval_horizon,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_pairs_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_pairs_table(frame))
    typer.echo(
        "planted_detected="
        f"{receipt['planted']['detected']} rank={receipt['planted']['rank_by_p_bh']} "
        f"spearman_ic={receipt['alignment']['spearman_ic']:+.4f}"
    )
    typer.echo(f"receipt={path}")


@app.command()
def race(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(256, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(128, help="Trailing eval rows, sliced into chunks."),
    n_chunks: int = typer.Option(8, help="Ordered eval chunks per shard (>=4, divides n_eval)."),
    alpha: float = typer.Option(0.05, help="Anytime-valid promotion level."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = fleet_race.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential fleet elimination race on SYNTHETIC shards.

    Each head fits the leading slice and predicts the eval slice once;
    eval rows split into time-ordered chunks and two e-processes per head
    vs the chunk-0 incumbent give anytime-valid promotion/elimination
    verdicts (proper scores only — correctness evidence, never P&L).
    """
    from quant_fund.research.fleet_eval import (
        fleet_head_factories,
        resolve_shard_generators,
    )
    from quant_fund.research.fleet_race import fleet_race, write_race_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
        resolved = resolve_shard_generators(None if shards is None else shards.split(","))
        frame, receipt = fleet_race(
            factories,
            resolved,
            n_train=n_train,
            n_eval=n_eval,
            n_chunks=n_chunks,
            alpha=alpha,
            seed=base_seed,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_race_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(
        frame.select(
            "shard", "model", "status", "promoted_at", "eliminated_at", "shard_winner", "verdict"
        )
    )
    typer.echo(f"receipt={path}")


@app.command("lane-power")
def lane_power(
    defects: str = typer.Option("0.0,0.1,0.25,0.5,1.0", help="Comma-separated defect sizes."),
    n_steps: int = typer.Option(400, help="Stream length per run."),
    n_seeds: int = typer.Option(20, help="Seeds per (lane, defect) cell."),
    alpha: float = typer.Option(0.05, help="Alarm threshold."),
    lanes: str | None = typer.Option(
        None, help="Comma-separated lane names (default: all available)."
    ),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = lane_power.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential power bench — measured alarm rate/time per monitor lane.

    Injects controlled defects into synthetic streams and records how fast
    each anytime-valid lane alarms; the defect=0 row bounds each lane's
    false-alarm rate by alpha. Lanes whose modules are unmerged show
    ``lane_missing`` on the receipt.
    """
    from quant_fund.research.lane_power import lane_power_bench, write_lane_power_receipt

    try:
        defect_grid = tuple(float(x) for x in defects.split(","))
    except ValueError as exc:
        raise typer.BadParameter(f"defects must be numeric: {exc}") from exc
    frame, receipt = lane_power_bench(
        defects=defect_grid,
        n_steps=n_steps,
        n_seeds=n_seeds,
        alpha=alpha,
        lanes=None if lanes is None else tuple(lanes.split(",")),
    )
    try:
        path = write_lane_power_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"lanes_ok={receipt['n_lanes_ok']} rows={frame.height} "
        f"null_alarm_rate={receipt['null_alarm_rate']}"
    )
    typer.echo(f"receipt={path}")


@app.command("suite-health")
def suite_health_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), help="Directory of committed receipts to re-verify."
    ),
    alpha: float = typer.Option(0.05, help="Pooled-evidence alarm threshold."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = suite_health.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit 1 unless every failing receipt is a byte-pinned legacy "
        "unsealed artifact (CI evidence-audit gate).",
    ),
) -> None:
    """Re-verify every receipt in a directory + pool evidence → sealed summary.

    One command audits the whole evidence trail: each file gets a fresh
    verify-receipt pass (seal + kind contract), harvestable p-values /
    e-values are pooled under arbitrary dependence, and a corrupt artifact
    withholds the pooled claim — never asserted over partial evidence.
    """
    from quant_fund.research.suite_health import suite_health, write_suite_health_receipt

    try:
        frame, receipt = suite_health(receipts_dir, alpha=alpha)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_suite_health_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=receipt["data_label"],
        )
    )
    typer.echo(
        f"receipts={receipt['n_receipts']} ok={receipt['n_ok']} "
        f"failed={receipt['n_failed']} pooled_evalue={receipt['pooled_evalue']}"
    )
    typer.echo(f"receipt={path}")
    if strict:
        import polars as pl

        from quant_fund.research.legacy_unsealed import (
            is_known_contract_legacy,
            is_known_unsealed,
        )
        from quant_fund.research.receipt_v2 import verify_receipt_file

        bad: list[str] = []
        for row in frame.filter(~pl.col("valid")).iter_rows(named=True):
            p = receipts_dir / str(row["file"])
            result = verify_receipt_file(p)
            if not (
                is_known_unsealed(p, result["errors"])
                or is_known_contract_legacy(p, result["errors"])
            ):
                bad.append(f"{row['file']}: {result['errors']}")
        epoch_state = receipt.get("corpus_epoch")
        if isinstance(epoch_state, dict) and epoch_state.get("chain_ok") is False:
            for err in epoch_state.get("errors") or []:
                bad.append(f"corpus_epoch: {err}")
        if bad:
            typer.echo("STRICT FAILURE — unverifiable receipts:", err=True)
            for line in bad:
                typer.echo(f"  {line}", err=True)
            raise typer.Exit(code=1)


@app.command()
def mcs(
    streams: Path = typer.Argument(
        ...,
        help="Loss streams: JSON {head: [per-origin losses]} or parquet/csv "
        "(numeric column per head). Lower loss = better head.",
    ),
    alpha: float = typer.Option(0.05, help="Coverage failure level."),
    lam: float = typer.Option(0.5, help="Betting fraction per pair-process."),
    data_label: str = typer.Option("UNKNOWN", help="Provenance label stamped on the receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = mcs_seq.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Sequential model confidence set over per-origin proper losses.

    Maintains a survivor set containing an optimal head with probability
    >= 1 - alpha *uniformly over time* (Ville + union bound over pairwise
    e-processes; permanent elimination). Writes a sealed mcs_seq.v1
    receipt — proper-score evidence only.
    """
    from quant_fund.research.mcs_seq import load_loss_streams, mcs_report, write_mcs_receipt

    try:
        loss_streams = load_loss_streams(streams)
        receipt = mcs_report(loss_streams, alpha=alpha, lam=lam, data_label=data_label)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        path = write_mcs_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    typer.echo(f"survivors={receipt['survivors']}")
    typer.echo(f"eliminated={receipt['eliminated']}")
    typer.echo(f"champion={receipt['champion']}")
    typer.echo(f"receipt={path}")


@app.command("serial-watch")
def serial_watch_cmd(
    pits: Path = typer.Argument(..., help="JSON file: a list of PITs in (0,1), or {name: [pits]}."),
    n_lags: int = typer.Option(5, help="Max lag for the sign-product families."),
    alpha: float = typer.Option(0.05, help="Per-family claim level."),
    lam: float = typer.Option(0.5, help="Bet cap λ ∈ (0,1)."),
    data_label: str = typer.Option("UNKNOWN", help="Provenance label stamped on each receipt."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = serial_watch.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Anytime-valid PIT serial-independence audit; sealed receipt per stream.

    Proper-scores evidence only — reports the per-lag family claim and the
    pooled e-value claim separately (never one merged flag; the joint claim
    boundary is per-family level α AND pooled level α, not one shared level).
    """
    import json

    from quant_fund.research.serial_watch import serial_report, write_serial_receipt

    try:
        raw = json.loads(pits.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"cannot read PIT file {pits}: {exc}") from exc
    streams: dict[str, object] = (
        {"stream": raw}
        if isinstance(raw, list)
        else {str(k): v for k, v in raw.items()}
        if isinstance(raw, dict)
        else {}
    )
    if not streams:
        raise typer.BadParameter("PIT file must be a JSON list or {name: [pits]} mapping")
    try:
        for name, stream in streams.items():
            receipt = serial_report(
                stream, n_lags=n_lags, alpha=alpha, lam=lam, data_label=data_label
            )
            path = write_serial_receipt(receipt, out_dir, receipt_version=receipt_version)
            typer.echo(
                format_data_label(synthetic=data_label == "SYNTHETIC", data_source=data_label)
            )
            typer.echo(
                f"{name}: alarmed_lags={receipt['alarmed_lags']} "
                f"(per-family claim, level {alpha}) pooled_evalue={receipt['pooled_evalue']:.4g} "
                f"pooled_alarmed={receipt['pooled_alarmed']} (separate pooled claim, level {alpha})"
            )
            typer.echo(f"receipt={path}")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("cost-calibration")
def cost_calibration(
    half_spread_bps: float = typer.Option(1.0, help="Flat half-spread floor in bps."),
    lookback: int = typer.Option(20, help="Trailing OHLC window for estimators."),
    n_dates: int = typer.Option(40, help="Dates in the SYNTHETIC panel."),
    n_names: int = typer.Option(4, help="Names in the SYNTHETIC panel."),
    seed: int = typer.Option(7, help="Panel seed."),
    planted_rel_spread: float = typer.Option(
        0.002, help="Planted high-low relative full spread for the SYNTHETIC book."
    ),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    report_path: Path = typer.Option(
        Path("reports/cost_calibration_flat_vs_ohlc.md"),
        help="Markdown report path (flat vs calibrated trial table).",
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = cost_calibration.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Flat vs OHLC-calibrated cost trials (dev-only SYNTHETIC diagnostic).

    Matched books under flat half-spread and Corwin–Schultz / Abdi–Ranaldo /
    Roll. Reports decomposed costs only — never Sharpe or live P&L.
    """
    if not dev:
        raise typer.BadParameter(
            "cost-calibration is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.cost_calibration import (
        format_cost_calibration_table,
        run_cost_calibration_trials,
        write_cost_calibration_receipt,
        write_cost_calibration_report,
    )

    frame, receipt = run_cost_calibration_trials(
        half_spread_bps=half_spread_bps,
        lookback=lookback,
        n_dates=n_dates,
        n_names=n_names,
        seed=seed,
        planted_rel_spread=planted_rel_spread,
    )
    path = write_cost_calibration_receipt(receipt, out_dir, receipt_version=receipt_version)
    report = write_cost_calibration_report(frame, receipt, report_path)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_cost_calibration_table(frame))
    typer.echo(f"receipt={path}")
    typer.echo(f"report={report}")


@app.command()
def verdict(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    alpha: float = typer.Option(0.05, help="Confidence level for the verdict lanes."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples for winner's-curse."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = honest_verdict.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Fleet tournament → composite honest verdict → sealed receipt.

    Runs every head on the synthetic shards, keeps the per-origin loss
    and PIT streams (not just aggregates), and asks `honest_verdict`
    whether the winner's claim holds: promotion e-process, winner's-curse
    correction, drift alarm, magnitude CS, PIT calibration, changepoint
    localization. Verdicts: confirmed / supported_with_caveats /
    not_supported / inconclusive — inconclusive is a valid answer, never
    forced into a binary.
    """
    from quant_fund.research.fleet_eval import fleet_head_factories
    from quant_fund.research.verdict_run import run_verdict, write_verdict_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    report, status = run_verdict(
        factories,
        None if shards is None else shards.split(","),
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        alpha=alpha,
        n_boot=n_boot,
        taus=cfg.quantiles.levels,
    )
    path = write_verdict_receipt(report, out_dir, receipt_version=receipt_version)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(f"verdict={report['verdict']} winner={report.get('winner')}")
    excluded = report["run"]["excluded_heads"]
    if excluded:
        typer.echo(f"excluded_heads={','.join(excluded)}")
    for name, detail in sorted(report["components"].items()):
        typer.echo(f"  {name}: {json.dumps(detail)[:200]}")
    typer.echo(f"receipt={path}")


@app.command(name="fleet-monitor")
def monitor(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    seed: int | None = typer.Option(
        None, help="Base seed (default: train.random_seed from config)."
    ),
    alpha: float = typer.Option(0.05, help="Alarm threshold (anytime-valid)."),
    level: float = typer.Option(0.9, help="Central interval for coverage lane."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = monitor_run.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Fleet tournament → every anytime-valid monitor lane → sealed receipt.

    Re-runs the tournament keeping per-origin quantiles, then streams each
    (shard, head) cell through the monitor family: coverage breach rate,
    nested tail depth, PIT calibration, conformal exchangeability, and
    loss drift vs the fleet median. Lanes whose modules are not merged
    report lane_missing on the receipt rather than failing silently.
    """
    from quant_fund.research.fleet_eval import fleet_head_factories
    from quant_fund.research.monitor_run import monitor_fleet, write_monitor_receipt

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    try:
        factories = fleet_head_factories(
            cfg.quantiles.levels,
            base_seed,
            None if models is None else models.split(","),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    frame, receipt = monitor_fleet(
        factories,
        None if shards is None else shards.split(","),
        n_train=n_train,
        n_eval=n_eval,
        seed=base_seed,
        alpha=alpha,
        level=level,
        taus=cfg.quantiles.levels,
    )
    try:
        path = write_monitor_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"rows={receipt['n_rows']} alarms={receipt['n_alarm_rows']} "
        f"lanes={sum(receipt['lanes_available'].values())}/5"
    )
    typer.echo(f"receipt={path}")


@app.command()
def corpus(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of committed receipts to audit."
    ),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level for the pooled corpus family."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = corpus_inference.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Pool claims into a BH-FDR corpus_inference.v1 receipt.

    Correctness evidence, never a market or P&L claim."""
    try:
        from quant_fund.research.corpus_inference import corpus_audit, write_corpus_receipt
    except ImportError as exc:
        raise typer.BadParameter("requires corpus_inference (PR #382)") from exc

    try:
        receipt = corpus_audit(receipts_dir, q=q)
    except ValueError as exc:
        message = str(exc)
        # Lead with the phrase so Rich cannot split "does not exist".
        if message.endswith("does not exist"):
            message = f"does not exist: {receipts_dir}"
        raise typer.BadParameter(message) from exc
    try:
        path = write_corpus_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"corpus receipts={receipt.get('n_receipts')} "
        f"p_findings={receipt.get('n_p_findings')} "
        f"survivors={receipt.get('n_survivors')} "
        f"corpus_evalue={receipt.get('corpus_evalue')} "
        f"parse_errors={receipt.get('n_parse_errors')}"
    )
    typer.echo(f"receipt={path}")


@app.command("online-fdr")
def online_fdr_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of committed receipts to replay."
    ),
    level: float = typer.Option(0.05, "--level", help="Target mFDR bound for the stream."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    per_receipt: bool = typer.Option(
        False,
        "--per-receipt",
        help="Treat each receipt as ONE test (min harvested p); default tests per finding.",
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = online_fdr.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
) -> None:
    """Replay committed receipts through Foster–Stine alpha-investing.

    Receipts replay in mtime/name order; every harvested p-value is one
    test (or one per receipt under ``--per-receipt``) fed to ``OnlineFDR``,
    whose wealth budget bounds the stream mFDR at ``--level`` at every
    arrival. Writes a sealed ``online_fdr.v1`` receipt. Correctness
    evidence, never a market or P&L claim.
    """
    try:
        from quant_fund.research.corpus_inference import harvest_findings
    except ImportError as exc:
        raise typer.BadParameter("requires corpus_inference (PR #382)") from exc
    try:
        from quant_fund.research.online_fdr import OnlineFDR, write_online_fdr_receipt
    except ImportError as exc:
        raise typer.BadParameter("requires online_fdr (PR #383)") from exc
    import json

    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    root = Path(receipts_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"does not exist: {root}")
    try:
        controller = OnlineFDR(level=level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    from quant_fund.utils.receipt import verified_corpus_files

    files = sorted(
        verified_corpus_files(root),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    digests: dict[str, str] = {}
    skipped: list[str] = []
    n_p_findings = 0
    for path in files:
        try:
            raw = path.read_bytes()
            payload: object = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            skipped.append(path.name)
            continue
        if not isinstance(payload, Mapping):
            skipped.append(path.name)
            continue
        digests[path.name] = hash_bytes(raw)
        ps = _harvest_p_values(harvest_findings(payload, path.name))
        n_p_findings += len(ps)
        if per_receipt:
            if ps:
                controller.update(min(ps))
        else:
            for p in ps:
                controller.update(p)

    receipt = {
        "kind": "online_fdr.v1",
        "schema": "online_fdr.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "generated_at_commit": git_revision(),
        "inputs_sha256": hash_bytes(
            canonical_json_bytes({"digests": digests, "level": level, "per_receipt": per_receipt})
        ),
        "params": {
            "level": level,
            "per_receipt": per_receipt,
            "receipts_dir": str(root),
        },
        "n_receipts": len(files),
        "n_skipped": len(skipped),
        "skipped_files": skipped,
        "n_p_findings": n_p_findings,
        **controller.stream_report(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = write_online_fdr_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"online_fdr tests={receipt['n_tests']} rejections={receipt['n_rejections']} "
        f"final_wealth={receipt['final_wealth']:.6g} level={receipt['level']} "
        f"skipped={receipt['n_skipped']}"
    )
    typer.echo(f"receipt={path}")


@app.command("tape-pin")
def tape_pin_cmd(
    source_label: str = typer.Option(
        ...,
        "--source-label",
        help="Label the manifest is filed under (e.g. yahoo_eod, synthetic_bench)",
    ),
    tape: list[Path] = typer.Option(
        [],
        "--tape",
        help="Tape parquet to pin (repo-relative; repeatable, concatenated in order)",
    ),
    promotion_receipt: Path | None = typer.Option(
        None,
        "--promotion-receipt",
        help="Optional bar_promotion.v1 receipt whose bytes are bound into the manifest",
    ),
    out_dir: Path = typer.Option(
        Path("data") / "manifests", "--out-dir", help="Manifest output directory"
    ),
) -> None:
    """Pin raw tape bytes into the committed ``data/manifests`` registry.

    Writes ``<out_dir>/<source_label>.json`` (``tape_manifest.v1``): sealed
    sha256/byte counts per tape file plus the canonical CSV digest lanes
    hash as ``inputs_sha256``. Tapes stay gitignored — the manifest attests
    bytes, it does not ship them.
    """
    from quant_fund.research.tape_registry import pin_tape

    if not tape:
        raise typer.BadParameter("pass at least one --tape parquet")
    for path in tape:
        if not path.is_file():
            raise typer.BadParameter(f"tape file not found: {path}")
    try:
        result = pin_tape(
            source_label,
            list(tape),
            out_dir=out_dir,
            promotion_receipt=promotion_receipt,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    manifest = result["manifest"]
    typer.echo(
        format_data_label(synthetic=source_label.upper() == "SYNTHETIC", data_source=source_label)
    )
    typer.echo(
        f"tape_files={len(manifest['tape_files'])} n_rows={manifest['n_rows']} "
        f"n_names={manifest['n_names']} frame_csv_sha256={manifest['frame_csv_sha256'][:12]}…"
    )
    typer.echo(f"receipt_sha256={manifest['receipt_sha256']}")
    typer.echo(f"manifest={result['path']}")


@app.command("tape-verify")
def tape_verify_cmd(
    manifest: Path = typer.Option(..., "--manifest", help="tape_manifest.v1 JSON to verify"),
    root: Path = typer.Option(
        Path("."), "--root", help="Repository root the manifest's tape paths resolve under"
    ),
) -> None:
    """Re-hash a pinned manifest against the tape bytes on this machine.

    Exits non-zero on any drift: stale seal, missing/tampered tape files, or
    a profile/CSV digest that no longer re-derives — fail-closed.
    """
    from quant_fund.research.tape_registry import verify_manifest

    errors = verify_manifest(manifest, root)
    typer.echo(json.dumps({"manifest": str(manifest), "valid": not errors, "errors": errors}))
    raise typer.Exit(code=0 if not errors else 1)


__all__ = [
    "capacity",
    "cost_calibration",
    "corpus",
    "execution_sensitivity_cmd",
    "fleet",
    "pairs",
    "lane_power",
    "mcs",
    "monitor",
    "verdict",
    "lane_power",
    "monitor",
    "serial_watch_cmd",
    "suite_health_cmd",
    "verdict",
    "lane_power",
    "mcs",
    "monitor",
    "suite_health_cmd",
    "verdict",
    "race",
    "race",
    "race",
    "race",
    "suite_health_cmd",
    "monitor",
    "online_fdr_cmd",
    "rankic",
    "research",
    "verdict",
    "verify_identities",
    "verify_all_cmd",
    "verify_receipt_cmd",
    "tape_pin_cmd",
    "tape_verify_cmd",
    "replay_cmd",
    "vol_bench",
]


@app.command("lattice")
def lattice_cmd(
    receipts_dir: Path = typer.Option(
        Path("receipts"), "--receipts-dir", help="Directory of receipts to lattice."
    ),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    float_rel_tol: float = typer.Option(
        1e-9, "--float-rel-tol", help="Relative tolerance for numeric-drift edges."
    ),
    head_sha: str | None = typer.Option(
        None, "--head-sha", help="Current HEAD sha for stale-code flags (default: auto)."
    ),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = receipt_lattice.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    known_inconsistent: Path | None = typer.Option(
        None,
        "--known-inconsistent",
        help="JSON map of receipt filename -> sha256 whose byte-exact "
        "inconsistent claim groups are acknowledged (demo artifacts).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero iff the verdict is 'inconsistent' (stale/drift/known-pinned pass).",
    ),
) -> None:
    """Cross-receipt consistency lattice over a receipts directory.

    Edges receipts that claim the same inputs/dataset fingerprints and
    compares their shared claim paths: consistent / numeric_drift /
    inconsistent. Provenance claims only — no P&L.
    """
    from quant_fund.research.receipt_lattice import receipt_lattice, write_lattice_receipt
    from quant_fund.utils.reproducibility import git_revision

    root = Path(receipts_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"does not exist: {root}")
    pins: dict[str, str] | None = None
    if known_inconsistent is not None:
        if not known_inconsistent.is_file():
            raise typer.BadParameter(f"known-inconsistent file {known_inconsistent} does not exist")
        try:
            raw_pins = json.loads(known_inconsistent.read_text())
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"known-inconsistent is not JSON: {exc}") from exc
        if not isinstance(raw_pins, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in raw_pins.items()
        ):
            raise typer.BadParameter(
                "known-inconsistent must be a JSON object mapping filename -> 64-hex sha256"
            )
        pins = dict(raw_pins)
    receipt = receipt_lattice(
        root,
        head_sha=head_sha or git_revision(),
        float_rel_tol=float_rel_tol,
        known_inconsistent=pins,
    )
    try:
        path = write_lattice_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"lattice receipts={receipt['n_receipts']} groups={receipt['n_claim_groups']} "
        f"verdict={receipt['verdict']}"
    )
    typer.echo(f"receipt={path}")
    if strict and receipt["verdict"] == "inconsistent":
        raise typer.Exit(code=1)


@app.command()
def corpus_epoch(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Evidence corpus to epoch-stamp / chain-check."
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help="Verify the committed epoch chain against the live corpus "
        "(single fork-free chain, monotone membership, head == live root).",
    ),
    allowed_removals: Path | None = typer.Option(
        None,
        "--allowed-removals",
        help="JSON map of receipt filename -> sha256 whose removal is acknowledged "
        "(e.g. moved to legacy-unsealed/).",
    ),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = corpus_epoch.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    glob: str | None = typer.Option(
        None,
        "--glob",
        help="Member file pattern — chains are per-(dir, glob). Default: the "
        "corpus's established chain pattern (fail-closed if it has several); "
        "'*.json' when the dir has no chain yet.",
    ),
    heads_pin: Path | None = typer.Option(
        None,
        "--heads-pin",
        help="Committed epoch-heads pin (e.g. quality/epoch_heads.json). --check "
        "enforces it (missing/mutated/rolled-back head is an error); write mode "
        "updates it so the pin and the new epoch land in the same commit.",
    ),
    require_stamped: bool = typer.Option(
        False,
        "--require-stamped",
        help="With --check: unstamped members are errors, not informational — "
        "for corpora where every member is security-critical (e.g. CI workflows).",
    ),
    allow_member_updates: bool = typer.Option(
        False,
        "--allow-member-updates",
        help="With --check: digest changes to stamped members between epochs are "
        "history attestation, not tamper errors — for mutable corpora (quality "
        "manifests, workflow definitions). Leave off for append-only corpora.",
    ),
    allow_new_pattern: bool = typer.Option(
        False,
        "--allow-new-pattern",
        help="Stamp mode: permit minting a chain under a --glob the corpus has "
        "never used. Without it, stamping a dir that already has epochs under "
        "a different pattern fails closed — a mismatched glob forges a parallel "
        "chain whose records read as unstamped members of the real one.",
    ),
) -> None:
    """Corpus epoch: hash-chained integrity root over the evidence store.

    Seals the corpus's full membership (name -> sha256 over file bytes) into a
    ``corpus_epoch.v1`` receipt linked to the previous epoch, so receipt
    deletion or rewrite becomes detectable without trusting git. ``--check``
    walks the committed chain and fails on forks, removed or mutated members,
    and drift between the head epoch and the live corpus. Provenance evidence
    only, never a market or P&L claim.
    """
    from quant_fund.research.corpus_epoch import (
        EpochStampLocked,
        _acquire_stamp_lock,
        check_epoch_chain,
        corpus_epoch,
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
        write_epoch_receipt,
    )

    root = Path(corpus_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"corpus dir {root} does not exist")
    if glob is None:
        from quant_fund.research.corpus_epoch import _epoch_receipts

        existing = {
            str((e.get("params") or {}).get("pattern", "*.json"))
            for _p, e in _epoch_receipts(root)[0]
        }
        if len(existing) > 1:
            raise typer.BadParameter(
                f"corpus {root} has epoch chains under several patterns "
                f"{sorted(existing)} — pass --glob to disambiguate"
            )
        glob = existing.pop() if existing else "*.json"
    allowed: dict[str, str] | None = None
    if allowed_removals is not None:
        if not allowed_removals.is_file():
            raise typer.BadParameter(f"allowed-removals file {allowed_removals} does not exist")
        try:
            raw_allowed = json.loads(allowed_removals.read_text())
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"allowed-removals is not JSON: {exc}") from exc
        if not isinstance(raw_allowed, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) == 64
            for k, v in raw_allowed.items()
        ):
            raise typer.BadParameter(
                "allowed-removals must be a JSON object mapping filename -> 64-hex sha256"
            )
        allowed = dict(raw_allowed)
    expected_head: dict[str, str] | None = None
    if heads_pin is not None:
        if not heads_pin.is_file():
            raise typer.BadParameter(f"heads-pin file {heads_pin} does not exist")
        try:
            expected_head = load_heads_pin(heads_pin).get(epoch_heads_key(root, glob))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"heads-pin is not a valid pin file: {exc}") from exc
        if check and expected_head is None:
            raise typer.BadParameter(
                f"heads-pin {heads_pin} has no entry for {epoch_heads_key(root, glob)}"
            )
    if check:
        result = check_epoch_chain(
            root,
            allowed_removals=allowed,
            pattern=glob,
            expected_head=expected_head,
            require_stamped=require_stamped,
            allow_member_updates=allow_member_updates,
        )
        typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
        if not require_stamped:
            for name in result["unstamped"]:
                typer.echo(f"epoch-chain info: unstamped member {name}")
        if result["errors"]:
            for err in result["errors"]:
                typer.echo(f"epoch-chain error: {err}")
            raise typer.Exit(code=1)
        typer.echo("epoch-chain intact")
        return
    head_sha: str | None = None
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            head_sha = proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        head_sha = None
    if heads_pin is not None and Path(out_dir) != root:
        raise typer.BadParameter(
            "--heads-pin requires --out-dir == --corpus-dir (the pinned head must "
            "be a corpus member)"
        )
    try:
        lock_fd = _acquire_stamp_lock(root)
    except EpochStampLocked as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        try:
            receipt = corpus_epoch(
                root,
                head_sha=head_sha,
                pattern=glob,
                allow_new_pattern=allow_new_pattern,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        try:
            path = write_epoch_receipt(receipt, out_dir, receipt_version=receipt_version)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if heads_pin is not None:
            update_heads_pin(heads_pin, root, glob, path)
            typer.echo(f"heads-pin={heads_pin}")
    finally:
        lock_fd.close()
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"epoch members={receipt['n_members']} "
        f"added={len(receipt['members_added'])} removed={len(receipt['members_removed'])} "
        f"verdict={receipt['verdict']} root={receipt['epoch_root_sha256'][:16]}"
    )
    typer.echo(f"receipt={path}")


@app.command("tombstone")
def tombstone_cmd(
    receipt: Path = typer.Argument(..., help="Receipt file to retract."),
    reason: str = typer.Option(..., "--reason", help="Why the receipt is retracted."),
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Corpus dir the tombstone joins."
    ),
    scope: str = typer.Option(
        "all",
        "--scope",
        help="'all' or a comma-separated list of claim paths to retract.",
    ),
) -> None:
    """Retract a receipt: append a sealed receipt_tombstone.v1 to the corpus.

    The corpus is append-only — a wrong or superseded receipt can't be
    deleted without breaking the epoch chain, so it is retracted instead:
    the lattice excludes its claims and the tombstone becomes chain
    evidence itself. Provenance evidence only, never a market or P&L claim.
    """
    from quant_fund.research.receipt_tombstone import write_tombstone

    target = Path(receipt)
    if not target.is_file():
        raise typer.BadParameter(f"receipt {target} does not exist")
    scope_val: str | list[str] = (
        "all" if scope == "all" else [s.strip() for s in scope.split(",") if s.strip()]
    )
    if scope_val != "all" and not scope_val:
        raise typer.BadParameter("--scope must be 'all' or non-empty claim paths")
    out = write_tombstone(target, corpus_dir=corpus_dir, reason=reason, scope=scope_val)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(f"tombstone={out} target={target.name}")


@app.command("corpus-proof")
def corpus_proof_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus the member lives in."
    ),
    member: str | None = typer.Option(
        None, "--member", help="Member name (corpus-relative, e.g. a receipt filename)."
    ),
    epoch: str | None = typer.Option(
        None, "--epoch", help="Epoch receipt name to bind to (default: chain head)."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed corpus_proof.v1 receipt here (default: inside the corpus).",
    ),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing corpus_proof receipt file instead of making one."
    ),
    pin: Path | None = typer.Option(
        None,
        "--pin",
        help="Offline mode: verify --check against this epoch_heads.json pin "
        "instead of the live corpus (third-party audit path).",
    ),
) -> None:
    """Merkle inclusion proof: this member sat in the corpus at epoch N.

    Emits an O(log n) ``corpus_proof.v1`` receipt — sibling path from the
    member's leaf to the epoch's Merkle root — so "was this file in the
    corpus then?" verifies offline without re-sending the epoch's whole
    member map. ``--check`` re-derives the root from the path *and* from
    the referenced epoch receipt and requires both to agree; ``--check``
    with ``--pin`` verifies against the signed heads pin alone — the
    third-party path, needing only the proof file plus the quorum-signed,
    OTS-anchored ``epoch_heads.json``.
    """
    import json as _json

    from quant_fund.research.epoch_merkle import (
        member_proof,
        verify_epoch_proof,
        verify_proof_pin,
    )

    if check is not None:
        payload = _json.loads(check.read_text(encoding="utf-8"))
        # Unwrap the receipt.v2 envelope — the proof body lives in "payload".
        body_payload = payload.get("payload", payload)
        if pin is not None:
            from quant_fund.research.corpus_epoch import load_heads_pin

            heads = load_heads_pin(pin)
            corpus_key = body_payload.get("corpus_key")
            if not isinstance(corpus_key, str):
                # Older proofs carry no corpus key — the pin is matched on
                # epoch_receipt name + tree_root instead.
                corpus_key = next(
                    (
                        k
                        for k, e in heads.items()
                        if e.get("receipt") == body_payload.get("epoch_receipt")
                    ),
                    "",
                )
            errors = verify_proof_pin(body_payload, heads.get(corpus_key, {}))
            for err in errors:
                typer.echo(f"corpus-proof error: {err}")
            if errors:
                raise typer.Exit(code=1)
            typer.echo(
                f"corpus-proof verified offline: {body_payload.get('member')} under "
                f"pin {pin.name} (key={corpus_key})"
            )
            return
        errors = verify_epoch_proof(body_payload, corpus_dir)
        for err in errors:
            typer.echo(f"corpus-proof error: {err}")
        if errors:
            raise typer.Exit(code=1)
        typer.echo(
            f"corpus-proof verified: {body_payload.get('member')} in "
            f"{body_payload.get('epoch_receipt')} (n={body_payload.get('n_members')})"
        )
        return

    if member is None:
        raise typer.BadParameter("--member is required unless --check is passed")
    body = member_proof(corpus_dir, member, epoch_receipt=epoch)
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    sealed = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(__file__).parent.parent / "research" / "epoch_merkle.py",),
            verdict="pass",
        )
    )
    dest = out or (corpus_dir / f"corpus_proof_{sealed['receipt_sha256'][:16]}.json")
    from quant_fund.utils.atomicio import atomic_write_text

    atomic_write_text(dest, _json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"corpus-proof member={member} epoch={body['epoch_receipt']} "
        f"depth={len(body['path'])} receipt={dest}"
    )


@app.command("corpus-absence")
def corpus_absence_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus to prove non-membership in."
    ),
    member: str | None = typer.Option(
        None, "--member", help="Name proven absent (corpus-relative)."
    ),
    epoch: str | None = typer.Option(
        None, "--epoch", help="Epoch receipt name to bind to (default: chain head)."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed corpus_absence.v1 receipt here (default: inside the corpus).",
    ),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing corpus_absence receipt instead of making one."
    ),
    pin: Path | None = typer.Option(
        None,
        "--pin",
        help="Offline mode: verify --check against this epoch_heads.json pin "
        "instead of the live corpus.",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        help="Prove absence at EVERY committed epoch (corpus_history_absence.v1) "
        "rather than at one bound epoch.",
    ),
) -> None:
    """Non-membership proof: this name was NOT in the corpus at epoch N.

    Emits a sealed ``corpus_absence.v1`` receipt — the two sorted-name
    neighbors bracketing the gap, each with a shape-bound inclusion path;
    ``hi == lo + 1`` proves nothing can sit between them. ``--check`` with
    ``--pin`` verifies against the quorum-signed, OTS-anchored heads pin —
    the third-party "was this receipt ever committed?" path.

    ``--history`` strengthens the claim to *every* epoch: the receipt binds
    the ordered genesis→head chain (names + file digests) and a verifier
    replays it, confirming the name never enters any member map. "Was
    secrets.env ever evidence?" answers in one sealed artifact.
    """
    import json as _json

    from quant_fund.research.epoch_merkle import (
        HISTORY_ABSENCE_SCHEMA,
        absence_receipt,
        verify_absence_pin,
        verify_epoch_absence,
        verify_history_absence,
        verify_history_absence_pin,
    )

    if check is not None:
        payload = _json.loads(check.read_text(encoding="utf-8"))
        body_payload = payload.get("payload", payload)
        if (
            body_payload.get("schema") == HISTORY_ABSENCE_SCHEMA
            or body_payload.get("kind") == HISTORY_ABSENCE_SCHEMA
        ):
            if pin is not None:
                from quant_fund.research.corpus_epoch import load_heads_pin

                heads = load_heads_pin(pin)
                corpus_key = body_payload.get("corpus_key")
                if not isinstance(corpus_key, str):
                    corpus_key = next(
                        (
                            k
                            for k, e in heads.items()
                            if e.get("receipt") == body_payload.get("head_receipt")
                        ),
                        "",
                    )
                errors = verify_history_absence_pin(body_payload, heads.get(corpus_key, {}))
                for err in errors:
                    typer.echo(f"history-absence error: {err}")
                if errors:
                    raise typer.Exit(code=1)
                typer.echo(
                    f"history-absence verified offline: {body_payload.get('name')} "
                    f"absent at all {body_payload.get('n_epochs')} epochs "
                    f"(pin {pin.name}, key={corpus_key})"
                )
                return
            errors = verify_history_absence(body_payload, corpus_dir)
            for err in errors:
                typer.echo(f"history-absence error: {err}")
            if errors:
                raise typer.Exit(code=1)
            typer.echo(
                f"history-absence verified: {body_payload.get('name')} absent at "
                f"all {body_payload.get('n_epochs')} committed epochs"
            )
            return
        if pin is not None:
            from quant_fund.research.corpus_epoch import load_heads_pin

            heads = load_heads_pin(pin)
            corpus_key = body_payload.get("corpus_key")
            if not isinstance(corpus_key, str):
                corpus_key = next(
                    (
                        k
                        for k, e in heads.items()
                        if e.get("receipt") == body_payload.get("epoch_receipt")
                    ),
                    "",
                )
            errors = verify_absence_pin(body_payload, heads.get(corpus_key, {}))
            for err in errors:
                typer.echo(f"corpus-absence error: {err}")
            if errors:
                raise typer.Exit(code=1)
            typer.echo(
                f"corpus-absence verified offline: {body_payload.get('name')} absent under "
                f"pin {pin.name} (key={corpus_key})"
            )
            return
        errors = verify_epoch_absence(body_payload, corpus_dir)
        for err in errors:
            typer.echo(f"corpus-absence error: {err}")
        if errors:
            raise typer.Exit(code=1)
        typer.echo(
            f"corpus-absence verified: {body_payload.get('name')} absent in "
            f"{body_payload.get('epoch_receipt')} (n={body_payload.get('n_members')})"
        )
        return

    if member is None:
        raise typer.BadParameter("--member is required unless --check is passed")
    from quant_fund.research.epoch_merkle import history_absence_receipt
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if history:
        # The member glob is the corpus's declared pattern (receipts/*.json,
        # configs/*, verifier/*.md ...) — derive it from the policy table, not
        # the filename convention, so chained non-JSON corpora emit too.
        from quant_fund.research.repo_integrity import CORPORA

        declared = {c[0]: c[1] for c in CORPORA}
        body = history_absence_receipt(
            corpus_dir,
            member,
            pattern=declared.get(corpus_dir.as_posix(), "*.json"),
        )
        sealed = seal_receipt(
            wrap_receipt_v2(
                body,
                code_files=(Path(__file__).parent.parent / "research" / "epoch_merkle.py",),
                verdict="pass",
            )
        )
        dest = out or corpus_dir
        if dest.suffix != ".json":
            dest = dest / f"corpus_history_absence_{sealed['receipt_sha256'][:16]}.json"
        from quant_fund.utils.atomicio import atomic_write_text

        atomic_write_text(dest, _json.dumps(sealed, indent=2, sort_keys=True) + "\n")
        typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
        typer.echo(
            f"history-absence name={member} epochs={body['n_epochs']} "
            f"head={body['head_receipt']} receipt={dest}"
        )
        return

    body = absence_receipt(corpus_dir, member, epoch_receipt=epoch)
    sealed = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(__file__).parent.parent / "research" / "epoch_merkle.py",),
            verdict="pass",
        )
    )
    dest = out or corpus_dir
    if dest.suffix != ".json":
        dest = dest / f"corpus_absence_{sealed['receipt_sha256'][:16]}.json"
    from quant_fund.utils.atomicio import atomic_write_text

    atomic_write_text(dest, _json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"corpus-absence name={member} epoch={body['epoch_receipt']} "
        f"bounds={len(body['bounds'])} receipt={dest}"
    )


@app.command("epoch-delta")
def epoch_delta_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus to diff within."
    ),
    prev_epoch: str | None = typer.Option(
        None, "--prev-epoch", help="Older epoch receipt filename (corpus_epoch_*.json)."
    ),
    next_epoch: str | None = typer.Option(
        None, "--next-epoch", help="Newer epoch receipt filename (corpus_epoch_*.json)."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed epoch_delta.v1 receipt here (default: inside the corpus).",
    ),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing epoch_delta receipt instead of making one."
    ),
) -> None:
    """Epoch delta: the sealed, completeness-verified change-set between epochs.

    Emits ``epoch_delta.v1`` — both epoch receipts bound by name+digest, both
    member-map digests, both Merkle roots, and the full transition table
    (added/removed/changed rows each carrying shape-bound inclusion paths
    against the root they belong to, plus the unchanged-count accounting
    pin). ``--check`` re-derives the member maps from the two epoch
    receipts and requires the declared table to equal the computed
    difference exactly — a dropped or invented row fails closed.
    """
    import json as _json

    from quant_fund.research.epoch_delta import (
        epoch_delta_receipt,
        verify_epoch_delta,
    )
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text

    if check is not None:
        payload = _json.loads(check.read_text(encoding="utf-8"))
        body_payload = payload.get("payload", payload)
        errors = verify_epoch_delta(body_payload, corpus_dir)
        for err in errors:
            typer.echo(f"epoch-delta error: {err}")
        if errors:
            raise typer.Exit(code=1)
        tr = body_payload["transitions"]
        typer.echo(
            f"epoch-delta verified: {body_payload['prev_epoch']['receipt']} -> "
            f"{body_payload['next_epoch']['receipt']} (+{len(tr['added'])} "
            f"-{len(tr['removed'])} ~{len(tr['changed'])} "
            f"={tr['unchanged_count']})"
        )
        return

    if prev_epoch is None or next_epoch is None:
        raise typer.BadParameter("--prev-epoch and --next-epoch are required")
    body = epoch_delta_receipt(corpus_dir, prev_epoch, next_epoch)
    sealed = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(__file__).parent.parent / "research" / "epoch_delta.py",),
            verdict="pass",
        )
    )
    dest = out or corpus_dir
    if dest.suffix != ".json":
        dest = dest / f"epoch_delta_{sealed['receipt_sha256'][:16]}.json"
    atomic_write_text(dest, _json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    tr = sealed["payload"]["transitions"] if "payload" in sealed else body["transitions"]
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"epoch-delta prev={prev_epoch} next={next_epoch} +{len(tr['added'])} "
        f"-{len(tr['removed'])} ~{len(tr['changed'])} ={tr['unchanged_count']} "
        f"receipt={dest}"
    )


@app.command("epoch-position")
def epoch_position_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus the epoch belongs to."
    ),
    receipt: str | None = typer.Option(
        None, "--receipt", help="corpus_epoch_*.json filename to prove a position for."
    ),
    epoch: str | None = typer.Option(
        None, "--epoch", help="Alias for --receipt (position of this epoch receipt)."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed epoch_position.v1 receipt here (default: inside the corpus).",
    ),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing epoch_position receipt instead of making one."
    ),
    pin: Path | None = typer.Option(
        None,
        "--pin",
        help="Offline mode: verify --check against this epoch_heads.json pin "
        "(its per-corpus chain_root + n_epochs anchor the proof in O(log n)).",
    ),
) -> None:
    """Epoch position proof: this receipt occupied position K in the chain.

    The heads pin commits a ``chain_root`` — an RFC 6962 tree over the
    *ordered* epoch chain (leaves bind (position, name, file digest), so
    indices can't be re-presented). The sealed ``epoch_position.v1``
    receipt proves "this exact corpus state was committed at position K"
    against the signed pin alone — no chain walk needed.
    """
    import json as _json

    from quant_fund.research.epoch_merkle import (
        epoch_position_receipt,
        verify_epoch_position,
        verify_epoch_position_pin,
    )

    if check is not None:
        payload = _json.loads(check.read_text(encoding="utf-8"))
        body_payload = payload.get("payload", payload)
        if pin is not None:
            from quant_fund.research.corpus_epoch import load_heads_pin

            heads = load_heads_pin(pin)
            corpus_key = body_payload.get("corpus_key")
            if not isinstance(corpus_key, str):
                corpus_key = next(
                    (
                        k
                        for k, e in heads.items()
                        if e.get("receipt") == body_payload.get("receipt")
                    ),
                    "",
                )
            errors = verify_epoch_position_pin(body_payload, heads.get(corpus_key, {}))
            for err in errors:
                typer.echo(f"epoch-position error: {err}")
            if errors:
                raise typer.Exit(code=1)
            typer.echo(
                f"epoch-position verified offline: {body_payload.get('receipt')} "
                f"at position {body_payload.get('position')}/"
                f"{body_payload.get('n_epochs')} (pin {pin.name})"
            )
            return
        errors = verify_epoch_position(body_payload, corpus_dir)
        for err in errors:
            typer.echo(f"epoch-position error: {err}")
        if errors:
            raise typer.Exit(code=1)
        typer.echo(
            f"epoch-position verified: {body_payload.get('receipt')} at position "
            f"{body_payload.get('position')} of {body_payload.get('n_epochs')}"
        )
        return

    target = receipt or epoch
    if target is None:
        raise typer.BadParameter("--receipt is required unless --check is passed")
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.atomicio import atomic_write_text

    body = epoch_position_receipt(corpus_dir, target)
    sealed = seal_receipt(
        wrap_receipt_v2(
            body,
            code_files=(Path(__file__).parent.parent / "research" / "epoch_merkle.py",),
            verdict="pass",
        )
    )
    dest = out or corpus_dir
    if dest.suffix != ".json":
        dest = dest / f"epoch_position_{sealed['receipt_sha256'][:16]}.json"
    atomic_write_text(dest, _json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"epoch-position receipt={target} position={body['position']}/{body['n_epochs']} out={dest}"
    )


@app.command("corpus-consistency")
def corpus_consistency_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus to prove extension over."
    ),
    from_receipt: str | None = typer.Option(
        None,
        "--from-epoch",
        help="Older epoch receipt name the chain must extend (default: genesis).",
    ),
    held_sha256: str | None = typer.Option(
        None,
        "--held",
        help="Digest of the from-head bytes a verifier already trusts — proves the "
        "live chain extends that exact state, not a rewritten one.",
    ),
    glob: str = typer.Option("*.json", "--glob", help="Member glob of the chain."),
    out: Path | None = typer.Option(
        None, "--out", help="Write the consistency proof JSON here (stdout otherwise)."
    ),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing proof file against the live corpus."
    ),
) -> None:
    """Chain-consistency proof: the current head *extends* a held older head.

    The RFC 6962 mirror of ``corpus-proof``: inclusion asks "was member m in
    epoch N?", consistency asks "does the live chain still contain the head
    I already verified?". A rewritten history can only satisfy the proof by
    keeping every real intermediate receipt verbatim — at which point it is
    the real history. Pass ``--held`` with the old head's digest (from an
    anchored pin or an earlier clone) to bind the proof to held state.
    Provenance evidence only.
    """
    import json as _json

    from quant_fund.research.epoch_consistency import (
        consistency_proof,
        verify_consistency,
    )

    if check is not None:
        proof = _json.loads(check.read_text(encoding="utf-8"))
        errors = verify_consistency(proof, corpus_dir, held_sha256=held_sha256)
        for err in errors:
            typer.echo(f"corpus-consistency error: {err}")
        if errors:
            raise typer.Exit(code=1)
        typer.echo(
            f"corpus-consistency verified: {proof.get('from_receipt', {}).get('name')} "
            f"-> {proof.get('to_receipt', {}).get('name')} "
            f"({proof.get('n_hops')} hops)"
        )
        return

    if from_receipt is None:
        # Genesis = the hop no epoch names as its successor's prev.
        from quant_fund.research.epoch_consistency import chain_index

        index = chain_index(corpus_dir, pattern=glob)
        if not index:
            raise typer.BadParameter(f"no epoch receipts in {corpus_dir} for {glob}")
        candidates = [
            name
            for name in index
            if index[name][1].get("prev_epoch_receipt") is None
            or index[name][1].get("prev_epoch_receipt") not in index
        ]
        if not candidates:
            raise typer.BadParameter("no genesis receipt reachable")
        from_receipt = sorted(candidates)[0]
    proof = consistency_proof(corpus_dir, from_receipt, pattern=glob)
    if held_sha256 is not None and proof["from_receipt"]["sha256"] != held_sha256:
        typer.echo("corpus-consistency error: held_head_digest_mismatch")
        raise typer.Exit(code=1)
    text = _json.dumps(proof, indent=2, sort_keys=True) + "\n"
    if out is not None:
        from quant_fund.utils.atomicio import atomic_write_text

        atomic_write_text(out, text)
    else:
        typer.echo(text.rstrip())
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))


@app.command("custody")
def custody_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Chained corpus the member lives in."
    ),
    member: str | None = typer.Option(
        None, "--member", help="Member name (corpus-relative, e.g. a receipt filename)."
    ),
    pattern: str = typer.Option("*.json", "--glob", help="Member glob of the chain."),
    root: Path = typer.Option(Path("."), "--root", help="Repo root."),
    out: Path | None = typer.Option(None, "--out", help="Write the custody bundle here."),
    check: Path | None = typer.Option(
        None, "--check", help="Verify an existing custody bundle instead of making one."
    ),
    member_file: Path | None = typer.Option(
        None,
        "--member-file",
        help="File whose bytes are the custody subject (required with --check).",
    ),
    timeline: bool = typer.Option(
        False,
        "--timeline",
        help="Print the member's byte lineage across the chain instead of a bundle.",
    ),
    digest: str | None = typer.Option(
        None,
        "--digest",
        help="Digest custody: prove a sha256 is referenced inside a stamped "
        "corpus member (scans all chains unless --corpus-dir narrows it).",
    ),
) -> None:
    """One-file provenance proof: member → epoch inclusion → chain head →
    signed pins → checkpoint → Rekor witness, composed into a single
    ``custody_proof.v1`` bundle that verifies with no repo access.

    The epoch bound is the *earliest* chained epoch pinning the member's
    current bytes — proof of first committed state. ``--check`` needs only
    the bundle plus the subject file. ``--timeline`` prints the member's
    byte lineage (committed digest per epoch) for mutable corpora.
    ``--digest`` instead proves a sha256 is *referenced* inside a stamped
    member — e.g. a data manifest pinning a dataset digest.
    """
    import json as _json

    from quant_fund.research.custody import (
        custody_proof,
        digest_custody,
        member_timeline,
        verify_custody_bundle,
    )

    if timeline:
        if member is None:
            raise typer.BadParameter("--timeline requires --member")
        for entry in member_timeline(member, corpus_dir, pattern=pattern):
            typer.echo(
                f"epoch_index={entry['epoch_index']} sha256={entry['sha256']} "
                f"epoch={entry['epoch']}"
            )
        return

    if check is not None:
        if member_file is None or not member_file.is_file():
            raise typer.BadParameter("--check requires --member-file pointing at the subject file")
        bundle = _json.loads(check.read_text(encoding="utf-8"))
        res = verify_custody_bundle(bundle, member_file.read_bytes())
        for err in res["errors"]:
            typer.echo(f"custody error: {err}")
        layer_str = " ".join(
            f"{k}:{'ok' if v.get('ok') else 'FAIL'}" for k, v in res.get("layers", {}).items()
        )
        typer.echo(f"custody layers: {layer_str}")
        if not res["ok"]:
            raise typer.Exit(code=1)
        typer.echo("custody verified: full provenance chain authentic")
        return

    if digest is not None:
        if member is not None:
            raise typer.BadParameter("--digest and --member are mutually exclusive")
        corpus_opt = None
        if corpus_dir != Path("receipts") or pattern != "*.json":
            corpus_opt = ((corpus_dir.as_posix(), pattern),)
        bundle = digest_custody(digest, root=root, corpora=corpus_opt)
        typer.echo(f"carrier={bundle['carrier']} first_epoch={bundle['first_epoch']}")
    else:
        if member is None:
            raise typer.BadParameter("--member or --digest is required unless --check is passed")
        bundle = custody_proof(member, corpus_dir, pattern=pattern, root=root)
    text = _json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    if out is not None:
        from quant_fund.utils.atomicio import atomic_write_text

        atomic_write_text(out, text)
    else:
        typer.echo(text.rstrip())
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"custody member={bundle['member']} first_epoch={bundle['first_epoch']} "
        f"head={bundle['chain_head']} hops={bundle['n_hops']}"
    )


@app.command("attest-release")
def attest_release_cmd(
    artifacts: list[Path] = typer.Option(
        ..., "--artifact", help="Shipped artifact file (repeatable)."
    ),
    out: Path | None = typer.Option(None, "--out", help="Attestation output path."),
    root: Path = typer.Option(Path("."), "--root", help="Repo root."),
    key: str | None = typer.Option(
        None,
        "--key",
        envvar="GATE_SIGNING_KEY",
        help="Ed25519 key material: '<priv_hex>:<pub_hex>', a key file path, "
        "or bare priv hex (env GATE_SIGNING_KEY).",
    ),
    witness: bool = typer.Option(
        False,
        "--witness",
        help="Anchor the signed attestation in Rekor and embed the proof "
        "(requires WITNESS_SIGNING_KEY env or --witness-key-file).",
    ),
    witness_key_file: Path | None = typer.Option(
        None, "--witness-key-file", help="ECDSA P-256 witness key PEM path."
    ),
) -> None:
    """Sign a ``release_attestation.v1`` binding shipped artifact bytes to
    this tree's pin state — the release-time counterpart of ``custody``.

    The signature uses the same key as ``make sign-pins``, so a wheel can be
    proven to have shipped from a gate-green tree without trusting the build
    host. ``--witness`` additionally anchors the attestation in the public
    Rekor transparency log and embeds the self-verifying proof — the
    attestation alone then convinces an auditor with no repo access.
    """
    import json as _json

    from quant_fund.research.release_attestation import release_attestation

    if not key or not key.strip():
        raise typer.BadParameter("GATE_SIGNING_KEY unset — cannot sign a release attestation")
    raw = Path(key).read_text().strip() if Path(key).is_file() else key.strip()
    fields = [s.strip() for s in raw.replace(":", "\n").splitlines() if s.strip()]
    priv = fields[0]
    pub_path = root / "quality/gate_signing.pub"
    if not pub_path.is_file():
        raise typer.BadParameter(f"no committed pubkey at {pub_path}")
    pub = pub_path.read_text().strip()
    if len(fields) == 2 and fields[1] != pub:
        raise typer.BadParameter(
            "supplied pubkey != committed quality/gate_signing.pub — refusing to sign"
        )
    blob = {p.name: p.read_bytes() for p in artifacts}
    att = release_attestation(blob, root=root, private_seed_hex=priv, pubkey_hex=pub)
    if witness:
        from quant_fund.research.release_attestation import witness_release

        pem: bytes | None = None
        if witness_key_file is not None:
            pem = witness_key_file.read_bytes()
        else:
            import os

            env_key = os.environ.get("WITNESS_SIGNING_KEY", "").strip()
            if env_key:
                pem = Path(env_key).read_bytes() if Path(env_key).is_file() else env_key.encode()
        if pem is None:
            raise typer.BadParameter(
                "--witness needs WITNESS_SIGNING_KEY env or --witness-key-file"
            )
        r_pub = root / "quality/rekor_pubkey.pem"
        att = witness_release(
            att,
            witness_key_pem=pem,
            rekor_pubkey_pem=r_pub.read_bytes() if r_pub.is_file() else None,
        )
        idx = att.get("witness", {}).get("rekor", {}).get("log_index")
        typer.echo(f"rekor_witness log_index={idx}")
    text = _json.dumps(att, indent=2, sort_keys=True) + "\n"
    if out is not None:
        from quant_fund.utils.atomicio import atomic_write_text

        atomic_write_text(out, text)
    else:
        typer.echo(text.rstrip())
    typer.echo(f"attestation signed over {len(blob)} artifact(s)")


@app.command("verify-release")
def verify_release_cmd(
    attestation: Path = typer.Argument(..., help="release_attestation.v1 JSON."),
    artifacts: list[Path] = typer.Option(
        ..., "--artifact", help="Shipped artifact file (repeatable)."
    ),
    root: Path = typer.Option(Path("."), "--root", help="Repo root (freshness check)."),
    pubkey: str | None = typer.Option(
        None, "--pubkey", help="Ed25519 pubkey hex; default quality/gate_signing.pub."
    ),
) -> None:
    """Verify a signed release attestation against artifact bytes."""
    import json as _json2

    from quant_fund.research.release_attestation import verify_release_attestation

    att = _json2.loads(attestation.read_text(encoding="utf-8"))

    blob = {p.name: p.read_bytes() for p in artifacts}
    res = verify_release_attestation(att, blob, root=root, pubkey_hex=pubkey)
    for err in res["errors"]:
        typer.echo(f"verify-release error: {err}")
    if not res["ok"]:
        raise typer.Exit(code=1)
    typer.echo("release attestation verified")


@app.command("crown-jewels")
def crown_jewels_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo root the jewels live under."),
    pin: Path = typer.Option(
        Path("quality/crown_jewels.json"),
        "--pin",
        help="Committed pin file (quality/crown_jewels.json).",
    ),
    check: bool = typer.Option(
        False, "--check", help="Verify every crown jewel against the pin; nonzero on any drift."
    ),
    write: bool = typer.Option(
        False, "--write", help="Rewrite the pin over the current files (run after a legit change)."
    ),
) -> None:
    """Byte-pin the gate-defining files: lint/type/test config, the dependency
    lock, hooks, the secret-scan allowlist, and AGENTS.md — the honesty
    contract. A silent edit to any of them weakens every check downstream.
    ``--check`` fails on missing/mutated/symlinked jewels and on a pin that
    drifted from the code-defined coverage set. Config integrity only —
    never a market or P&L claim.
    """
    from quant_fund.research.crown_jewels import (
        crown_jewel_digests,
        crown_jewels_errors,
        write_crown_jewels_pin,
    )

    if check == write:
        raise typer.BadParameter("pass exactly one of --check / --write")
    if write:
        path = write_crown_jewels_pin(root, pin)
        typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
        typer.echo(f"crown-jewels pin={path} jewels={len(crown_jewel_digests(root))}")
        return
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    errors = crown_jewels_errors(root, pin)
    for err in errors:
        typer.echo(f"crown-jewels error: {err}")
    if errors:
        raise typer.Exit(code=1)
    typer.echo("crown-jewels intact")


@app.command("verify-repo")
def verify_repo_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo root to verify."),
    heads_pin: Path = typer.Option(
        Path("quality/epoch_heads.json"),
        "--heads-pin",
        help="Committed epoch-heads pin file.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed repo_integrity.v1 attestation here (e.g. quality/).",
    ),
    evidence_only: bool = typer.Option(
        False,
        "--evidence-only",
        help="Verify an evidence bundle (only evidence dirs, no src/): the "
        "crown-jewels gate reports skipped and the attestation records "
        "mode=evidence_only — a partial verdict can't masquerade as full.",
    ),
) -> None:
    """Repo integrity capstone: compose every evidence-integrity gate —
    crown-jewels byte pins plus all corpus epoch chains under the committed
    heads pin — into one verdict, optionally sealed as a repo_integrity.v1
    receipt. Provenance evidence only; never a market or P&L claim.
    """
    from quant_fund.research.repo_integrity import verify_repo, write_repo_integrity_receipt

    # Verify first: --out writes into a chain-covered corpus dir, so a
    # post-write verify would flag the attestation itself as drift.
    result = verify_repo(root, heads_pin=heads_pin, evidence_only=evidence_only)
    if out is not None:
        path = write_repo_integrity_receipt(
            out, root, heads_pin=heads_pin, evidence_only=evidence_only
        )
        typer.echo(f"receipt={path}")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    for name, gate in result["gates"].items():
        state = "ok" if gate["ok"] else "FAIL " + ",".join(gate["errors"])
        if name == "pin_signatures" and gate["ok"] and not gate.get("signed"):
            state = "unsigned"
        if gate.get("skipped"):
            state = f"skipped({gate['skipped']})"
        typer.echo(f"verify-repo {name}: {state}")
    if not result["ok"]:
        raise typer.Exit(code=1)
    typer.echo("repo integrity: all gates intact")


@app.command("evidence-export")
def evidence_export_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo root to export from."),
    out: Path = typer.Option(Path("evidence_bundle"), "--out", help="Bundle directory to write."),
) -> None:
    """Export the evidence bundle an auditor verifies with zero repo access:
    every member of the evidence corpora plus the gate signature, at
    repo-relative paths. Pair with ``verify-repo --evidence-only``. The
    epoch chains + signed pins travel inside the bundle, so the export
    needs no trust in the exporter. Provenance evidence only.
    """
    from quant_fund.research.evidence_export import export_evidence_bundle

    try:
        manifest = export_evidence_bundle(root, out)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"evidence-export: {exc}")
        raise typer.Exit(code=1) from exc
    total = sum(c["members"] for c in manifest["corpora"].values())
    typer.echo(f"bundle={out} corpora={len(manifest['corpora'])} members={total}")
    typer.echo("verify: dipcatcher verify-repo --root <bundle> --evidence-only")


@app.command("sign-pins")
def sign_pins_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: Path | None = typer.Option(
        None,
        "--key-file",
        help="File containing the hex Ed25519 private seed + public key "
        "(two lines, or `PRIV\\nPUB`). Falls back to GATE_SIGNING_KEY env.",
    ),
) -> None:
    """Ed25519-sign the integrity pins (epoch_heads.json + crown_jewels.json).

    Writes ``gate_pins.sig`` (repo root, outside every chained corpus) and
    ``quality/gate_signing.pub``. The private seed is never persisted —
    supply it via --key-file or the GATE_SIGNING_KEY env var
    ("<priv_hex>:<pub_hex>" or a file path). Run after `make stamp-epochs`
    so the signature covers the fresh pin bytes.
    """
    import os

    from quant_fund.research.gate_signatures import (
        generate_keypair,
        key_id,
        sign_pins,
    )

    raw: str | None = None
    if key_file is not None:
        raw = key_file.read_text().strip()
    else:
        env = os.environ.get("GATE_SIGNING_KEY", "").strip()
        if env:
            raw = Path(env).read_text().strip() if Path(env).is_file() else env
    if raw is None:
        priv, pub = generate_keypair()
        typer.echo(
            "generated an ephemeral keypair (no key supplied) — "
            "the signature only attests to this key's holder"
        )
    else:
        fields = [line.strip() for line in raw.replace(":", "\n").splitlines() if line.strip()]
        if len(fields) != 2:
            typer.echo("sign-pins: key material must be '<priv_hex>:<pub_hex>' or two lines")
            raise typer.Exit(code=2)
        priv, pub = fields
    sig_path = sign_pins(root, priv, pub)
    typer.echo(f"signature={sig_path} key_id={key_id(pub)}")


def _load_signer_pairs(raw: str) -> list[tuple[str, str]]:
    """Parse 'PRIV:PUB' or two-line material; comma/newline separates signers."""
    fields = [
        f.strip()
        for line in raw.replace(",", "\n").splitlines()
        for f in line.replace(":", "\n").splitlines()
        if f.strip()
    ]
    if not fields or len(fields) % 2 != 0:
        raise ValueError("signer material must be '<priv_hex>:<pub_hex>' pairs")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields), 2)]


@app.command("quorum-init")
def quorum_init_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    pubkeys: str = typer.Option(
        ...,
        "--pubkeys",
        help="Comma-separated registered Ed25519 public keys (64-hex each).",
    ),
    threshold: int = typer.Option(..., "--threshold", min=1),
) -> None:
    """Write the M-of-N signer registry (``quality/gate_quorum.json``).

    Once committed, lone-key ``gate_signatures.v1`` files fail closed — the pin
    manifest needs ``threshold`` valid signatures from registered keys. The
    registry's own bytes are covered by the quality epoch chain + checkpoint.
    """
    from quant_fund.research.gate_signatures import init_quorum

    pubs = [p.strip() for p in pubkeys.split(",") if p.strip()]
    path = init_quorum(root, pubs, threshold=threshold)
    typer.echo(f"registry={path} threshold={threshold}/{len(pubs)}")


@app.command("quorum-rotate")
def quorum_rotate_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    registry: Path = typer.Option(
        ...,
        "--registry",
        help="JSON file holding the new gate_quorum.v1 body.",
    ),
    key_file: list[Path] = typer.Option(
        [],
        "--key-file",
        help="OUTGOING quorum member key material. Repeatable — must reach "
        "the *current* registry's threshold to authorize the rotation.",
    ),
    reason: str = typer.Option("quorum rotation", "--reason"),
) -> None:
    """Rotate the quorum registry under authorization of the outgoing quorum.

    Writes a ``quorum_rotation.v1`` record (``quality/quorum_rotations/``)
    binding prev→new registry digests, signed by ≥threshold distinct current
    members, then installs the new ``gate_quorum.json`` byte-exact. The
    checkpoint head goes stale on rotation — re-sign pins and re-checkpoint
    in the same ceremony.
    """
    import json as _json

    from quant_fund.research.quorum_rotation import rotate_quorum

    raws = [f.read_text().strip() for f in key_file]
    if not raws:
        typer.echo("quorum-rotate: supply --key-file(s) of current quorum members")
        raise typer.Exit(code=2)
    try:
        new_registry = _json.loads(registry.read_text())
        signers = [pair for raw in raws for pair in _load_signer_pairs(raw)]
        out = rotate_quorum(root, new_registry, signers, reason=reason)
    except ValueError as exc:
        typer.echo(f"quorum-rotate: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"rotation={out}")


@app.command("quorum-sign")
def quorum_sign_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: list[Path] = typer.Option(
        [],
        "--key-file",
        help="Signer key material ('<priv>:<pub>' or two lines). Repeatable.",
    ),
) -> None:
    """Sign the pin manifest under the quorum registry (M-of-N).

    Each ``--key-file`` contributes one registered signature; unattainable
    quorums refuse to write (a partial signature file would be a brick).
    ``GATE_QUORUM_KEYS`` supplies pairs comma-separated when no files given.
    """
    import os

    from quant_fund.research.gate_signatures import sign_pins_quorum

    raws = [f.read_text().strip() for f in key_file]
    if not raws:
        env = os.environ.get("GATE_QUORUM_KEYS", "").strip()
        if env:
            raws = [env] if not Path(env).is_file() else [Path(env).read_text().strip()]
    if not raws:
        typer.echo("quorum-sign: supply --key-file(s) or GATE_QUORUM_KEYS")
        raise typer.Exit(code=2)
    try:
        signers = [pair for raw in raws for pair in _load_signer_pairs(raw)]
        out = sign_pins_quorum(root, signers)
    except ValueError as exc:
        typer.echo(f"quorum-sign: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"signature={out} signers={len(signers)}")


@app.command("anchor-timestamp")
def anchor_timestamp_cmd(
    file: Path = typer.Option(
        Path("quality/epoch_heads.json"),
        "--file",
        help="Repo-relative file to anchor (default: the epoch-heads pin).",
    ),
    root: Path = typer.Option(Path("."), "--root"),
    tsr_url: str = typer.Option("https://freetsa.org/tsr", "--tsr-url"),
) -> None:
    """RFC 3161-anchor a file to wall-clock time via a public TSA.

    POSTs the file's sha256 (never its contents) to the timestamp authority
    and commits the returned token under ``quality/timestamps/`` with an
    ``anchors.json`` entry. Proves the pinned state existed by the TSA's
    signature time — a history rewriter cannot mint the chain retroactively.
    """
    from quant_fund.research.timestamp_anchor import stamp_timestamp

    token = stamp_timestamp(file, root=root, tsr_url=tsr_url)
    typer.echo(f"timestamp={token}")


@app.command("ots-stamp")
def ots_stamp_cmd(
    file: Path = typer.Option(
        Path("quality/epoch_heads.json"),
        "--file",
        help="Repo-relative file to anchor (default: the epoch-heads pin).",
    ),
    root: Path = typer.Option(Path("."), "--root"),
    calendar: list[str] = typer.Option(
        [], "--calendar", help="OTS calendar URL override (repeatable)."
    ),
) -> None:
    """Bitcoin-anchor a file via OpenTimestamps public calendars.

    POSTs the file's sha256 (never its contents) to the calendar digest
    endpoints and commits the merged ``.ots`` proof under
    ``quality/timestamps/ots/`` + ``ots_anchors.json``. Pending attestations
    prove calendar submission immediately; once the calendar confirms on
    Bitcoin the same file verifies against the committed block header
    (``<name>.<height>.hdr``) with a pure sha256d<nBits PoW check — no API
    trust at verify time. Distinct trust root from the RFC 3161 lane.
    """
    from quant_fund.research.ots_anchor import DEFAULT_CALENDARS, stamp_ots

    cals = tuple(calendar) if calendar else DEFAULT_CALENDARS
    try:
        token = stamp_ots(file, root=root, calendars=cals)
    except ValueError as exc:
        typer.echo(f"ots-stamp: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"ots={token}")


@app.command("ots-upgrade")
def ots_upgrade_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    explorer: str = typer.Option(
        "", "--explorer", help="Block explorer API base (default blockstream.info)."
    ),
    full: bool = typer.Option(
        False, "--full", help="Also commit the block's txid list + coinbase."
    ),
    bury: int = typer.Option(
        0, "--bury", min=0, max=100, help="With --full: commit K successor headers (SPV burial)."
    ),
) -> None:
    """Upgrade pending OTS anchors to Bitcoin-confirmed proofs.

    Polls each anchor's own calendars for the upgraded timestamp; on a
    ``bitcoin`` attestation the proof is rewritten and the claimed block
    header committed (``<name>.<height>.hdr``), after which ``verify-repo``
    reports ``pow_verified`` — self-checked sha256d<nBits, no API trust.
    With ``--full`` the block's txid list + raw coinbase are committed too,
    and verify reaches ``fully_verified``: the OP_RETURN commitment is
    proven inside the block's own merkle root.
    """
    from quant_fund.research.ots_anchor import DEFAULT_EXPLORER, upgrade_ots

    res = upgrade_ots(root=root, explorer=explorer or DEFAULT_EXPLORER, full=full, bury=bury)
    for label, states in res.get("anchors", {}).items():
        typer.echo(f"{label}: {','.join(states)}")
    if not res["ok"]:
        for e in res.get("errors", []):
            typer.echo(f"error: {e}")
        raise typer.Exit(code=2)


@app.command("witness-scan")
def witness_scan_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    rekor_url: str = typer.Option("", "--rekor-url", help="Rekor API base."),
) -> None:
    """Sweep Rekor for every entry our witness key ever signed (online).

    The transparency log as a key-misuse oracle: a stolen WITNESS_SIGNING_KEY
    minting a divergent checkpoint tree lands entries under our key — any
    attested digest absent from the committed spine reports ``foreign``.
    """
    from quant_fund.research.witness_scan import DEFAULT_REKOR_URL, scan_witness_log

    res = scan_witness_log(root, rekor_url=rekor_url or DEFAULT_REKOR_URL)
    typer.echo(
        f"scanned={res['scanned']} foreign={len(res['foreign'])} "
        f"unrecognized={res['unrecognized']} online={res.get('online', True)}"
    )
    for f in res.get("foreign", []):
        typer.echo(f"FOREIGN: {f}")
    for f in res.get("explained", []):
        typer.echo(f"orphaned (registered): {f}")
    for e in res.get("errors", []):
        typer.echo(f"note: {e}")
    if not res["ok"]:
        raise typer.Exit(code=2)


@app.command("checkpoint")
def checkpoint_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: list[Path] = typer.Option(
        [],
        "--key-file",
        help="Signer key material ('<priv>:<pub>' or two lines). Repeatable "
        "for multisig checkpoints under a committed quorum registry. "
        "Falls back to GATE_SIGNING_KEY env.",
    ),
    anchor: bool = typer.Option(
        False,
        "--anchor",
        help="Also RFC 3161-anchor the checkpoint via FreeTSA (network).",
    ),
) -> None:
    """Sign the pin state into ``quality/checkpoint.json`` — a portable
    signed-tree-head an auditor can verify with just the pubkey.

    Covers the sha256 of both pin files and gate_pins.sig plus every
    corpus's pinned head receipt. When a ``gate_quorum.v1`` registry is
    committed the checkpoint is emitted as an M-of-N multisig envelope and
    refuses to write below quorum. Run LAST, after `make sign-pins`, so the
    checkpoint binds the current signature. ``--anchor`` time-binds the
    checkpoint itself.
    """
    import os

    from quant_fund.research.integrity_checkpoint import (
        anchor_checkpoint,
        write_checkpoint,
    )

    raws = [f.read_text().strip() for f in key_file]
    if not raws:
        env = os.environ.get("GATE_SIGNING_KEY", "").strip()
        if env:
            raws = [env] if not Path(env).is_file() else [Path(env).read_text().strip()]
    if not raws:
        typer.echo("checkpoint: no key material — supply --key-file or GATE_SIGNING_KEY")
        raise typer.Exit(code=2)
    try:
        signers = [pair for raw in raws for pair in _load_signer_pairs(raw)]
        path = write_checkpoint(root, signers)
    except ValueError as exc:
        typer.echo(f"checkpoint: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"checkpoint={path}")
    if anchor:
        typer.echo(f"timestamp={anchor_checkpoint(root)}")


@app.command("verify-checkpoint")
def verify_checkpoint_cmd(
    root: Path = typer.Option(Path("."), "--root"),
) -> None:
    """Verify the committed integrity checkpoint: signature under the
    committed pubkey, pinned digests vs the live tree (``current``), and
    the TSA anchor. Missing/unsigned checkpoint is neutral, not a failure.
    """
    from quant_fund.research.integrity_checkpoint import verify_checkpoint

    res = verify_checkpoint(root)
    if not res["signed"]:
        typer.echo("checkpoint: unsigned (no checkpoint committed)")
        return
    typer.echo(
        f"checkpoint: {'ok' if res['ok'] else 'FAIL'} "
        f"anchored={res['anchored']} current={res['current']}"
    )
    for err in res["errors"]:
        typer.echo(f"  {err}")
    if not res["ok"]:
        raise typer.Exit(code=1)


@app.command("witness-checkpoint")
def witness_checkpoint_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    key_file: Path | None = typer.Option(
        None,
        "--key-file",
        help="PEM file holding an ECDSA P-256 private key. "
        "Falls back to WITNESS_SIGNING_KEY env (PEM text or path).",
    ),
    target: Path = typer.Option(
        Path("quality/checkpoint.json"), "--target", help="Repo-relative file to witness."
    ),
    rekor_url: str = typer.Option("https://rekor.sigstore.dev", "--rekor-url"),
) -> None:
    """Witness ``quality/checkpoint.json`` into the public Rekor transparency
    log and commit the self-verifying proof under ``quality/witness/``.

    Only the artifact's sha256 and our ECDSA signature leave the machine;
    the committed proof verifies fully offline (RFC 6962 inclusion walk +
    Rekor's signed entry timestamp and checkpoint note under the pinned
    Rekor pubkey). Requires network on submit; none on verify.
    """
    import os

    from quant_fund.research.integrity_witness import submit_witness

    pem: bytes | None = None
    if key_file is not None:
        pem = key_file.read_bytes()
    else:
        env = os.environ.get("WITNESS_SIGNING_KEY", "").strip()
        if env:
            pem = Path(env).read_bytes() if Path(env).is_file() else env.encode()
    if pem is None:
        typer.echo("witness-checkpoint: no key material — supply --key-file or WITNESS_SIGNING_KEY")
        raise typer.Exit(code=2)
    out = submit_witness(root, pem, target=target, rekor_url=rekor_url)
    typer.echo(f"witness={out}")


@app.command("verify-witness")
def verify_witness_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    online: bool = typer.Option(
        False,
        "--online",
        help="Also check the live log: committed tree must be a prefix of "
        "Rekor's current signed tree head (network).",
    ),
) -> None:
    """Verify every committed ``quality/witness/*.json`` proof offline:
    digest + our witness signature + RFC 6962 inclusion + Rekor SET and
    checkpoint-note signatures. ``--online`` additionally proves the log
    still contains our tree (RFC 6962 consistency to the current STH).
    """
    if online:
        from quant_fund.research.integrity_witness import verify_witness_online

        res = verify_witness_online(root)
        if not res["witnessed"]:
            typer.echo("witness: none committed")
            return
        typer.echo(f"witness: {'ok' if res['ok'] else 'FAIL'} log_size={res.get('log_size')}")
        for err in res["errors"]:
            typer.echo(f"  {err}")
        if not res["ok"]:
            raise typer.Exit(code=1)
        return

    from quant_fund.research.integrity_witness import verify_witnesses

    res = verify_witnesses(root)
    if not res["witnessed"]:
        typer.echo("witness: none committed")
        return
    typer.echo(f"witness: {'ok' if res['ok'] else 'FAIL'}")
    for err in res["errors"]:
        typer.echo(f"  {err}")
    if not res["ok"]:
        raise typer.Exit(code=1)


@app.command("witness-bundle")
def witness_bundle_cmd(
    root: Path = typer.Option(Path("."), "--root"),
    out: Path = typer.Option(Path("quality/auditor_bundle.json"), "--out"),
) -> None:
    """Emit the zero-trust auditor bundle: checkpoint + pin files + pubkeys
    + freshest Rekor witness proof in one JSON document. Refuses to bundle
    a tree that doesn't verify. Hand the file to anyone — they need no repo
    access, only this CLI and trust in the public Rekor log.
    """
    from quant_fund.research.auditor_bundle import build_bundle

    try:
        path = build_bundle(root, out)
    except ValueError as exc:
        typer.echo(f"bundle refused: {exc}")
        raise typer.Exit(code=2) from exc
    typer.echo(f"bundle={path}")


@app.command("verify-bundle")
def verify_bundle_cmd(
    bundle: Path = typer.Argument(..., help="auditor_bundle.v1 JSON document"),
    rekor_pubkey: Path | None = typer.Option(
        None, "--rekor-pubkey", help="caller-pinned Rekor public key PEM (strongest)"
    ),
    no_fetch: bool = typer.Option(
        False, "--no-fetch", help="use the bundle's pinned Rekor key, never fetch live"
    ),
) -> None:
    """Verify an auditor bundle with zero trusted repo input.

    The witness key is authenticated by the transparency log itself: the
    Rekor entry body records which public key signed the witnessed digest,
    and the bundled ``witness_signing.pub`` must equal it byte-for-byte.
    """
    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(
        bundle,
        rekor_pubkey_pem=rekor_pubkey.read_bytes() if rekor_pubkey else None,
        rekor_url=None if no_fetch else "https://rekor.sigstore.dev",
    )
    typer.echo(f"bundle: {'ok' if res['ok'] else 'FAIL'} log_index={res.get('log_index')}")
    for err in res["errors"]:
        typer.echo(f"  {err}")
    if not res["ok"]:
        raise typer.Exit(code=1)


@app.command("tamper-drill")
def tamper_drill_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree to clone + attack."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed tamper_drill.v1 receipt here (e.g. quality/).",
    ),
) -> None:
    """Mutation-drill the integrity substrate: clone the state, attack every
    layer (pins, jewels, chains, checkpoint, witness proofs, corpus members),
    and prove ``verify-repo`` catches each one. A probe that escapes is the
    finding — the receipt only seals ``fail_closed`` when all were caught."""
    from quant_fund.research.tamper_drill import tamper_drill, write_drill_receipt

    result = tamper_drill(root)
    typer.echo(format_data_label(synthetic=True, data_source="CORPUS"))
    typer.echo(
        f"tamper-drill: {result['n_caught']}/{result['n_probes']} probes caught "
        f"— verdict={result['verdict']}"
    )
    for p in result["probes"]:
        if not p.get("caught"):
            typer.echo(f"  ESCAPED: {p['probe']} {p.get('errors', p.get('error'))}")
    if out is not None:
        path = write_drill_receipt(result, out)
        typer.echo(f"receipt={path}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("checkpoint-chain")
def checkpoint_chain_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree to verify."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed checkpoint_chain.v1 receipt here.",
    ),
) -> None:
    """Verify the checkpoint spine end-to-end: every archived predecessor
    resolves, every link's signature verifies, no forks, no orphans, and
    Rekor log indexes follow the chain order."""
    from quant_fund.research.checkpoint_chain import checkpoint_spine, write_chain_receipt

    result = checkpoint_spine(root)
    typer.echo(
        f"checkpoint-chain: spine={result['spine_length']}/{result['n_records']} "
        f"forks={result['n_forks']} orphans={result['n_orphans']} "
        f"verdict={result['verdict']}"
    )
    for e in result["errors"]:
        typer.echo(f"  {e}")
    if out is not None:
        path = write_chain_receipt(result, out)
        typer.echo(f"receipt={path}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("verify-rotations")
def verify_rotations_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree to verify."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed key_rotation_audit.v1 receipt here.",
    ),
) -> None:
    """Verify the gate-key rotation chain: each link dual-signed by the
    outgoing and incoming keys, genesis anchored to a spine-observed key,
    and the live gate_signing.pub equal to the chain terminus."""
    from quant_fund.research.key_rotation import verify_rotations

    result = verify_rotations(root)
    typer.echo(
        f"verify-rotations: n={result['n_rotations']} verdict={result['verdict']} "
        f"current={result.get('current_key_id', '?')}"
    )
    for e in result["errors"]:
        typer.echo(f"  {e}")
    for n in result["notes"]:
        typer.echo(f"  note: {n}")
    if out is not None:
        from quant_fund.research.receipt_v2 import seal_receipt
        from quant_fund.utils.atomicio import atomic_write_text

        payload = dict(result)
        payload["schema"] = "key_rotation_audit.v1"
        sealed = seal_receipt(payload)
        atomic_write_text(out, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
        typer.echo(f"receipt={out}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("rotate-key")
def rotate_key_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree."),
    old_key_env: str = typer.Option(
        "GATE_SIGNING_KEY", "--old-key-env", help="Env var with the outgoing Ed25519 seed hex."
    ),
    new_key_env: str = typer.Option(
        "GATE_SIGNING_KEY_NEW",
        "--new-key-env",
        help="Env var with the incoming Ed25519 seed hex.",
    ),
    reason: str = typer.Option("scheduled rotation", "--reason"),
) -> None:
    """Record an authorized gate-key rotation: a dual-signed receipt under
    quality/rotation_<id>.json proving the outgoing key authorized the
    incoming one. After this, re-sign pins with the new key, update
    gate_signing.pub to the new pubkey, then checkpoint + witness."""
    import os

    old_seed = os.environ.get(old_key_env)
    new_seed = os.environ.get(new_key_env)
    if not old_seed or not new_seed:
        typer.echo(f"rotate-key: need both ${old_key_env} and ${new_key_env}")
        raise typer.Exit(code=2)
    from quant_fund.research.key_rotation import rotate_key

    try:
        out = rotate_key(root, old_seed, new_seed, reason=reason)
    except ValueError as exc:
        typer.echo(f"rotate-key: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"rotation={out}")
    typer.echo("next: update quality/gate_signing.pub, make sign-pins, checkpoint, witness")


@app.command("fuzz-drill")
def fuzz_drill_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree to attack."),
    seed: int = typer.Option(1, "--seed", help="Mutation-sequence seed (replays identically)."),
    rounds: int | None = typer.Option(
        None, "--rounds", help="Cap the sampled mutations (default: all)."
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed fuzz_drill.v1 receipt here.",
    ),
) -> None:
    """Metamorphic fuzz of the integrity verifier: seeded random mutations
    classified must-fail vs must-pass — a verifier that rejects *everything*
    is just as broken as one that misses a forgery."""
    from quant_fund.research.fuzz_drill import fuzz_drill, write_fuzz_receipt

    result = fuzz_drill(root, seed=seed, rounds=rounds)
    typer.echo(format_data_label(synthetic=True, data_source="CORPUS"))
    typer.echo(
        f"fuzz-drill: seed={seed} mutations={result['n_mutations']} "
        f"escaped={result['n_escaped']} false_positive={result['n_false_positive']} "
        f"verdict={result['verdict']}"
    )
    for m in result["mutations"]:
        if m.get("outcome") in ("escaped", "false_positive"):
            typer.echo(f"  {m['outcome'].upper()}: {m['mutation']} {m.get('errors', '')}")
    if out is not None:
        path = write_fuzz_receipt(result, Path(out).parent if out.suffix else out)
        typer.echo(f"receipt={path}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command("fuzz-receipts")
def fuzz_receipts_cmd(
    root: Path = typer.Option(Path("."), "--root", help="Repo tree whose receipts/ to forge."),
    seed: int = typer.Option(1, "--seed", help="Mutation-selection seed."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write the sealed receipt_fuzz.v1 receipt here.",
    ),
) -> None:
    """Forge-and-reseal drill: mutate one claim inside each committed
    receipt, then re-seal it *honestly* — sha256 seals are integrity, not
    authenticity, so minting a self-consistent forgery is free. Only the
    verifier's semantic contract re-derivation can catch it."""
    from quant_fund.research.fuzz_drill import receipt_fuzz, write_fuzz_receipt

    result = receipt_fuzz(root, seed=seed)
    typer.echo(format_data_label(synthetic=True, data_source="CORPUS"))
    typer.echo(
        f"fuzz-receipts: seed={seed} mutations={result['n_mutations']} "
        f"escaped={result['n_escaped']} verdict={result['verdict']}"
    )
    for m in result["mutations"]:
        if m.get("outcome") == "escaped":
            typer.echo(f"  ESCAPED: {m['receipt']} {m['mutation']}")
    if out is not None:
        path = write_fuzz_receipt(result, Path(out).parent if out.suffix else out)
        typer.echo(f"receipt={path}")
    if not result["ok"]:
        raise typer.Exit(code=1)


@app.command()
def admit(
    receipt: Path = typer.Argument(..., help="Candidate receipt JSON to gate."),
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Committed evidence corpus the candidate joins."
    ),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level for the corpus delta check."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = receipt_admission.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    known_inconsistent: Path | None = typer.Option(
        None,
        "--known-inconsistent",
        help="JSON map of receipt filename -> sha256 whose byte-exact "
        "inconsistent claim groups are acknowledged (demo artifacts).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero unless the verdict is 'admit'.",
    ),
) -> None:
    """Receipt admission gate: may this receipt join the evidence corpus?

    verify-receipt answers validity; ``admit`` answers admission — seal +
    honesty stamps + lattice delta (new contradictions against the corpus)
    + corpus-FDR delta. Verdicts: admit | quarantine | reject. Provenance
    evidence only, never a market or P&L claim.
    """
    from quant_fund.research.admission import admission_check, write_admission_receipt

    candidate = Path(receipt)
    root = Path(corpus_dir)
    if not candidate.is_file():
        raise typer.BadParameter(f"candidate receipt {candidate} does not exist")
    if not root.is_dir():
        raise typer.BadParameter(f"corpus dir {root} does not exist")
    pins: dict[str, str] | None = None
    if known_inconsistent is not None:
        if not known_inconsistent.is_file():
            raise typer.BadParameter(f"known-inconsistent file {known_inconsistent} does not exist")
        try:
            raw_pins = json.loads(known_inconsistent.read_text())
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"known-inconsistent is not JSON: {exc}") from exc
        if not isinstance(raw_pins, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in raw_pins.items()
        ):
            raise typer.BadParameter(
                "known-inconsistent must be a JSON object mapping filename -> 64-hex sha256"
            )
        pins = dict(raw_pins)
    result = admission_check(candidate, root, q=q, known_inconsistent=pins)
    try:
        path = write_admission_receipt(result, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    failed = [c["name"] for c in result["checks"] if not c["ok"]]
    typer.echo(
        f"admission candidate={result['candidate']} corpus={result['n_corpus_receipts']} "
        f"verdict={result['verdict']}" + (f" failed_checks={failed}" if failed else "")
    )
    typer.echo(f"receipt={path}")
    if strict and result["verdict"] != "admit":
        raise typer.Exit(code=1)


@app.command("admit-batch")
def admit_batch_cmd(
    receipts: list[Path] = typer.Argument(
        ..., help="Incoming receipt JSONs to gate as one diff (e.g. PR-changed files)."
    ),
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Committed evidence corpus the batch joins."
    ),
    q: float = typer.Option(0.05, "--q", help="BH-FDR level for the corpus delta check."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = receipt_admission.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    known_inconsistent: Path | None = typer.Option(
        None,
        "--known-inconsistent",
        help="JSON map of receipt filename -> sha256 whose byte-exact "
        "inconsistent claim groups are acknowledged (demo artifacts).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero unless every candidate's verdict is 'admit'.",
    ),
) -> None:
    """Batch admission gate for a diff of receipts.

    Each file is gated against ``corpus minus the whole batch plus the
    already-processed files`` — the state a merge actually creates — so a
    committed receipt's lattice delta is never vacuous and intra-diff
    contradictions are attributed to the file that introduces them. Writes
    one admission receipt per candidate. ``--strict`` fails on any verdict
    other than ``admit``.
    """
    from quant_fund.research.admission import admit_batch, write_admission_receipt

    root = Path(corpus_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"corpus dir {root} does not exist")
    pins: dict[str, str] | None = None
    if known_inconsistent is not None:
        if not known_inconsistent.is_file():
            raise typer.BadParameter(f"known-inconsistent file {known_inconsistent} does not exist")
        try:
            raw_pins = json.loads(known_inconsistent.read_text())
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"known-inconsistent is not JSON: {exc}") from exc
        if not isinstance(raw_pins, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in raw_pins.items()
        ):
            raise typer.BadParameter(
                "known-inconsistent must be a JSON object mapping filename -> 64-hex sha256"
            )
        pins = dict(raw_pins)
    try:
        batch = admit_batch(receipts, root, q=q, known_inconsistent=pins)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    for result in batch["results"]:
        path = write_admission_receipt(result, out_dir, receipt_version=receipt_version)
        failed = [c["name"] for c in result["checks"] if not c["ok"]]
        typer.echo(
            f"admission candidate={result['candidate']} corpus={result['n_corpus_receipts']} "
            f"verdict={result['verdict']}" + (f" failed_checks={failed}" if failed else "")
        )
        typer.echo(f"receipt={path}")
    typer.echo(f"admit-batch candidates={batch['n_candidates']} verdict={batch['verdict']}")
    if strict and batch["verdict"] != "admit":
        raise typer.Exit(code=1)


@app.command("graph")
def graph_cmd(
    corpus_dir: Path = typer.Option(
        Path("receipts"), "--corpus-dir", help="Receipt corpus directory to audit."
    ),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = receipt_graph.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero unless the citation graph verdict is 'clean'.",
    ),
) -> None:
    """Provenance citation-graph audit over a receipt corpus.

    Resolves digest and filename references between corpus members and
    reports resolved edges, dangling references, filename cycles,
    unresolvable receipt names, and orphans. Structural audit only —
    no P&L.
    """
    from quant_fund.research.receipt_graph import receipt_graph, write_graph_receipt

    root = Path(corpus_dir)
    if not root.is_dir():
        raise typer.BadParameter(f"corpus dir {root} does not exist")
    receipt = receipt_graph(root)
    try:
        path = write_graph_receipt(receipt, out_dir, receipt_version=receipt_version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"graph members={receipt['n_members']} edges={receipt['n_edges']} "
        f"dangling={receipt['n_dangling']} cycles={receipt['n_cycles']} "
        f"verdict={receipt['verdict']}"
    )
    typer.echo(f"receipt={path}")
    if strict and receipt["verdict"] != "clean":
        raise typer.Exit(code=1)


@app.command("basis-carry")
def basis_carry_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    spot_path: Path | None = typer.Option(
        None,
        "--spot",
        help="Spot daily frame (parquet/CSV with event_time|date + close).",
    ),
    spot_source: str | None = typer.Option(
        None,
        "--spot-source",
        help="Collect the spot frame inline via a registered source (e.g. kraken_spot).",
    ),
    pair: str = typer.Option(
        "XBTUSD", "--pair", help="Spot pair passed to --spot-source (kraken_spot default XBTUSD)."
    ),
    contract: list[str] = typer.Option(
        [],
        "--contract",
        help="Dated contract symbol, collected via kraken_futures_mark (repeatable).",
    ),
    contracts_file: Path | None = typer.Option(
        None,
        "--contracts-file",
        help="Text file with one contract symbol per line (# comments allowed).",
    ),
    contract_path: list[str] = typer.Option(
        [],
        "--contract-path",
        help="SYM=PATH future mark frame inputs, parquet/CSV (repeatable).",
    ),
    delivery: list[str] = typer.Option(
        [],
        "--delivery",
        help="SYM=ISO8601 delivery-instant override (repeatable); default derives "
        "the Kraken FI_/FF_ symbol tail (FI 16:00Z, FF 08:00Z on the dated day).",
    ),
    data_label: str | None = typer.Option(
        None,
        "--data-label",
        help="Provenance label sealed into the receipt; every declared input shares "
        "it (default 'kraken' when inputs are collected from Kraken sources, "
        "required for file inputs — e.g. SYNTHETIC for fixtures).",
    ),
    tolerance: float = typer.Option(
        0.005,
        "--tolerance",
        help="|convergence_residual| share-of-spot tolerance for the settlement anchor.",
    ),
    min_overlap: int = typer.Option(5, "--min-overlap", help="Minimum shared dates per contract."),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = basis_carry.v1 (default), 2 = unified receipt.v2 envelope.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero when any contract fails to produce a measured row.",
    ),
) -> None:
    """Settlement-anchored cash-and-carry bench (P5.3).

    Inner-joins spot and dated-future mark closes on calendar date, reports the
    annualized basis curve by days-to-delivery, the basis at fixed dte buckets,
    and the convergence residual at the last observed date — for delivered
    contracts that is the true terminal settlement anchor. Descriptive
    statistics only, sealed as a ``basis_carry.v1`` receipt; never P&L.
    """
    import polars as pl

    from quant_fund.data.collector import collect_source
    from quant_fund.research.basis_carry import (
        CarryContractInput,
        kraken_delivery_from_symbol,
        run_basis_carry,
        write_basis_carry_receipt,
    )

    cfg = _cfg(config)
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")

    def load_frame(path: Path) -> pl.DataFrame:
        if not path.is_file():
            raise typer.BadParameter(f"frame path {path} does not exist")
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path)

    # --delivery SYM=ISO8601 overrides.
    delivery_overrides: dict[str, str] = {}
    for item in delivery:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise typer.BadParameter(f"--delivery must be SYM=ISO8601, got {item!r}")
        delivery_overrides[key.strip().upper()] = value.strip()

    if spot_path is not None and spot_source is not None:
        raise typer.BadParameter("pass either --spot or --spot-source, not both")
    used_files = spot_path is not None or bool(contract_path)
    if data_label is None and used_files:
        raise typer.BadParameter("file inputs need --data-label (e.g. SYNTHETIC for fixtures)")
    resolved_label = data_label or "kraken"
    if spot_source is not None:
        spot_result = collect_source(
            spot_source, cfg.data.root, fetch_kwargs={"pair": pair, "interval": 1440}
        )
        spot_frame = spot_result.frame
        typer.echo(f"spot source={spot_source} rows={spot_frame.height} data={spot_result.data}")
    elif spot_path is not None:
        spot_frame = load_frame(spot_path)
    else:
        raise typer.BadParameter("a spot input is required: --spot PATH or --spot-source NAME")

    contract_symbols = [symbol.strip() for symbol in contract if symbol.strip()]
    if contracts_file is not None:
        if not contracts_file.is_file():
            raise typer.BadParameter(f"contracts file {contracts_file} does not exist")
        contract_symbols.extend(
            line.strip().split("#")[0].strip()
            for line in contracts_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    inputs: list[CarryContractInput] = []
    seen: set[str] = set()
    for spec in contract_path:
        key, sep, value = spec.partition("=")
        if not sep or not key.strip():
            raise typer.BadParameter(f"--contract-path must be SYM=PATH, got {spec!r}")
        symbol = key.strip().upper()
        if symbol in seen:
            raise typer.BadParameter(f"duplicate contract {symbol!r}")
        seen.add(symbol)
        if symbol in delivery_overrides:
            delivery_instant: str = delivery_overrides[symbol]
        else:
            try:
                delivery_instant = kraken_delivery_from_symbol(symbol).isoformat()
            except ValueError:
                delivery_instant = "UNRESOLVED"
        inputs.append(
            CarryContractInput(
                symbol=symbol,
                frame=load_frame(Path(value)),
                delivery=delivery_instant,
                data_label=resolved_label,
            )
        )
    for symbol_raw in contract_symbols:
        symbol = symbol_raw.upper()
        if symbol in seen:
            raise typer.BadParameter(f"duplicate contract {symbol!r}")
        seen.add(symbol)
        mark_result = collect_source(
            "kraken_futures_mark", cfg.data.root, fetch_kwargs={"symbol": symbol}
        )
        typer.echo(f"mark {symbol} rows={mark_result.frame.height} data={mark_result.data}")
        if symbol in delivery_overrides:
            delivery_instant = delivery_overrides[symbol]
        else:
            try:
                delivery_instant = kraken_delivery_from_symbol(symbol).isoformat()
            except ValueError:
                delivery_instant = "UNRESOLVED"
        inputs.append(
            CarryContractInput(
                symbol=symbol,
                frame=mark_result.frame,
                delivery=delivery_instant,
                data_label=resolved_label,
            )
        )
    if not inputs:
        raise typer.BadParameter(
            "no contract inputs: pass --contract, --contracts-file, or --contract-path"
        )

    try:
        frame, receipt = run_basis_carry(
            spot=spot_frame,
            spot_label=resolved_label,
            contracts=inputs,
            tolerance=tolerance,
            min_overlap=min_overlap,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_basis_carry_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(frame)
    typer.echo(f"verdict={receipt['verdict']}")
    typer.echo(f"receipt={path}")
    if strict and receipt["n_error_rows"]:
        raise typer.Exit(code=1)


#: ``source:ARG`` legs resolve through this table — the symbol kwarg name and
#: the daily-resolution defaults differ per venue adapter (Kraken takes
#: ``pair``/``symbol``, OKX takes ``inst_id``). File legs bypass it entirely.
_XVENUE_SOURCE_KWARGS: dict[str, tuple[str, dict[str, Any]]] = {
    "kraken_spot": ("pair", {"interval": 1440}),
    "kraken_futures_mark": ("symbol", {"tick_type": "mark", "resolution": "1d"}),
    "kraken_funding": ("symbol", {}),
    "okx_spot": ("inst_id", {"bar": "1Dutc"}),
    "okx_mark": ("inst_id", {"bar": "1Dutc"}),
    "okx_funding": ("inst_id", {}),
}

#: Canonical BTC legs for ``--preset kraken-okx``: Kraken spot + perpetual
#: mark/funding against OKX spot + linear-swap mark/funding.
_XVENUE_PRESETS: dict[str, list[dict[str, str]]] = {
    "kraken-okx": [
        {
            "venue": "kraken",
            "spot": "kraken_spot:XBTUSD",
            "mark": "kraken_futures_mark:PF_XBTUSD",
            "funding": "kraken_funding:PF_XBTUSD",
        },
        {
            "venue": "okx",
            "spot": "okx_spot:BTC-USDT",
            "mark": "okx_mark:BTC-USDT-SWAP",
            "funding": "okx_funding:BTC-USDT-SWAP",
        },
    ],
}

_XVENUE_LEG_FIELDS = ("venue", "spot", "mark", "funding")


def _xvenue_parse_leg_spec(spec: str) -> dict[str, str]:
    """Parse one ``--leg`` spec: comma-separated venue/spot/mark/funding keys."""
    out: dict[str, str] = {}
    for chunk in spec.split(","):
        key, sep, value = chunk.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not sep or not key or not value:
            raise typer.BadParameter(
                f"--leg entries must be key=value pairs, got {chunk!r} in {spec!r}"
            )
        if key not in _XVENUE_LEG_FIELDS:
            raise typer.BadParameter(
                f"unknown --leg key {key!r}; expected one of {sorted(_XVENUE_LEG_FIELDS)}"
            )
        if key in out:
            raise typer.BadParameter(f"duplicate --leg key {key!r}")
        out[key] = value
    if "venue" not in out or "spot" not in out or "mark" not in out:
        raise typer.BadParameter(
            "--leg needs at least venue=, spot=, and mark= (funding= is optional)"
        )
    return out


def _xvenue_frame(spec_value: str, *, cfg_root: Path, field: str, venue: str) -> Any:
    """Resolve one leg field to a frame: ``source:ARG`` collect or file path."""
    import polars as pl

    from quant_fund.data.collector import collect_source

    head, sep, arg = spec_value.partition(":")
    if sep and head in _XVENUE_SOURCE_KWARGS:
        symbol_kwarg, defaults = _XVENUE_SOURCE_KWARGS[head]
        fetch_kwargs = {symbol_kwarg: arg, **defaults}
        result = collect_source(head, cfg_root, fetch_kwargs=fetch_kwargs)
        typer.echo(f"{venue}.{field} source={head} arg={arg} rows={result.frame.height}")
        return result.frame
    path = Path(spec_value)
    if not path.is_file():
        raise typer.BadParameter(
            f"{venue}.{field}: {spec_value!r} is neither a known source:ARG "
            f"({sorted(_XVENUE_SOURCE_KWARGS)}) nor an existing file"
        )
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


@app.command("xvenue-basis")
def xvenue_basis_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    leg: list[str] = typer.Option(
        [],
        "--leg",
        help="One venue leg, comma-separated key=value: "
        "'venue=kraken,spot=kraken_spot:XBTUSD,mark=kraken_futures_mark:PF_XBTUSD,"
        "funding=kraken_funding:PF_XBTUSD'. Each field is source:ARG or a "
        "parquet/CSV path. Repeat >= 2 times.",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Expand a canned pair of legs instead of passing --leg "
        f"(choices: {sorted(_XVENUE_PRESETS)}).",
    ),
    asset: str = typer.Option(
        "BTC", "--asset", help="Underlying asset label sealed into the receipt."
    ),
    data_label: str | None = typer.Option(
        None,
        "--data-label",
        help="Provenance label for legs that load frames from files (e.g. "
        "SYNTHETIC for fixtures). Source-collected legs are labeled by venue.",
    ),
    min_overlap: int = typer.Option(
        5, "--min-overlap", help="Minimum shared dates per leg and per venue pair."
    ),
    out_dir: Path = typer.Option(Path("receipts"), "--out-dir", help="Receipt output directory."),
    receipt_version: int = typer.Option(
        1,
        "--receipt-version",
        help="Receipt schema version: 1 = crossvenue_basis.v1 (default), "
        "2 = unified receipt.v2 envelope.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit nonzero when any leg or venue pair fails to produce a measured row.",
    ),
) -> None:
    """Cross-venue funding/basis bench (P5.5).

    Inner-joins each venue's spot and mark closes on calendar date, then
    diffs the basis (and the daily-summed realized funding rates, where both
    venues have history) across every venue pair — the cross-venue carry
    differential. Descriptive statistics only, sealed as a
    ``crossvenue_basis.v1`` receipt; never P&L.
    """
    from quant_fund.research.crossvenue_basis import (
        VenueLeg,
        run_crossvenue_basis,
        write_crossvenue_basis_receipt,
    )

    cfg = _cfg(config)
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    if preset is not None and leg:
        raise typer.BadParameter("pass either --preset or --leg, not both")
    if preset is not None:
        if preset not in _XVENUE_PRESETS:
            raise typer.BadParameter(
                f"unknown --preset {preset!r}; expected one of {sorted(_XVENUE_PRESETS)}"
            )
        leg_specs = _XVENUE_PRESETS[preset]
    else:
        leg_specs = [_xvenue_parse_leg_spec(spec) for spec in leg]
    if len(leg_specs) < 2:
        raise typer.BadParameter("crossvenue basis needs >= 2 legs (--leg or --preset)")

    uses_files = False
    for spec in leg_specs:
        for field in ("spot", "mark", "funding"):
            value = spec.get(field)
            if value is not None and value.split(":", 1)[0] not in _XVENUE_SOURCE_KWARGS:
                uses_files = True
    if uses_files and data_label is None:
        raise typer.BadParameter("file inputs need --data-label (e.g. SYNTHETIC for fixtures)")

    legs: list[VenueLeg] = []
    for spec in leg_specs:
        venue = spec["venue"].lower()
        leg_uses_files = any(
            spec[field].split(":", 1)[0] not in _XVENUE_SOURCE_KWARGS for field in ("spot", "mark")
        ) or (
            spec.get("funding") is not None
            and spec["funding"].split(":", 1)[0] not in _XVENUE_SOURCE_KWARGS
        )
        label = data_label if leg_uses_files else venue
        if not label:
            raise typer.BadParameter(f"leg {venue!r} needs --data-label for its file inputs")
        legs.append(
            VenueLeg(
                venue=venue,
                spot=_xvenue_frame(spec["spot"], cfg_root=cfg.data.root, field="spot", venue=venue),
                mark=_xvenue_frame(spec["mark"], cfg_root=cfg.data.root, field="mark", venue=venue),
                funding=(
                    _xvenue_frame(
                        spec["funding"], cfg_root=cfg.data.root, field="funding", venue=venue
                    )
                    if spec.get("funding")
                    else None
                ),
                data_label=str(label),
            )
        )

    try:
        frame, receipt = run_crossvenue_basis(legs=legs, asset=asset, min_overlap=min_overlap)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_crossvenue_basis_receipt(receipt, out_dir, receipt_version=receipt_version)
    typer.echo(
        format_data_label(
            synthetic=receipt["data_label"] == "SYNTHETIC",
            data_source=str(receipt["data_label"]),
        )
    )
    if receipt["data_label"] == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(frame)
    typer.echo(f"verdict={receipt['verdict']}")
    typer.echo(f"receipt={path}")
    if strict and receipt["n_error_rows"]:
        raise typer.Exit(code=1)


@app.command()
def coherence(
    panels: str | None = typer.Option(
        None, help="Comma-separated panel names (default: every registered generator)."
    ),
    methods: str | None = typer.Option(
        None,
        help="Comma-separated reconciliation methods (default: direct, naive_sum, "
        "independent_mc, copula_mc).",
    ),
    n_train: int = typer.Option(384, help="Leading rows used to fit margins and correlation."),
    n_eval: int = typer.Option(128, help="Trailing scored origins."),
    n_mc: int = typer.Option(512, help="Monte-Carlo paths for the copula aggregators (>=16)."),
    seed: int = typer.Option(0, help="Synthetic panel seed."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Score quantile-reconciliation methods on SYNTHETIC panels and seal a receipt.

    Compares direct / naive-sum / independent-MC / copula-MC aggregation on
    seeded synthetic factor, independent, heavy-tail and regime-copula panels.
    Reported as CRPS, PIT-KS and central-interval coverage — proper scores only,
    never market evidence and never a P&L or Sharpe claim.
    """
    from quant_fund.research.coherence import (
        METHODS,
        PANEL_GENERATORS,
        format_coherence_table,
        run_coherence,
        write_coherence_receipt,
    )

    panel_names = None if panels is None else [p.strip() for p in panels.split(",") if p.strip()]
    if panel_names is not None:
        unknown = sorted(set(panel_names) - set(PANEL_GENERATORS))
        if unknown:
            raise typer.BadParameter(
                f"unknown panel(s) {', '.join(unknown)}; "
                f"registered: {', '.join(sorted(PANEL_GENERATORS))}"
            )
    method_names = None if methods is None else [m.strip() for m in methods.split(",") if m.strip()]
    if method_names is not None:
        unknown = sorted(set(method_names) - set(METHODS))
        if unknown:
            raise typer.BadParameter(
                f"unknown method(s) {', '.join(unknown)}; registered: {', '.join(METHODS)}"
            )
    try:
        frame, receipt = run_coherence(
            panel_names,
            n_train,
            n_eval,
            seed=seed,
            n_mc=n_mc,
            methods=method_names if method_names is not None else METHODS,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_coherence_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_coherence_table(frame))
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command()
def concordance(
    config: Path = typer.Option(Path("configs/research.yaml")),
    heads: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    alpha: float = typer.Option(0.10, help="DM elimination significance level."),
    n_boot: int = typer.Option(500, help="Stationary-bootstrap replicates."),
    block: float | None = typer.Option(
        None, help="Mean bootstrap block length (default: lane choice)."
    ),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Measure selection concordance across the SYNTHETIC fleet and seal a receipt.

    Does the ranking a lane picks agree with the ranking an independent
    elimination procedure produces? Reports Kendall tau, Jaccard overlap and the
    DM-eliminated set — selection diagnostics only, never a live-trading claim.
    """
    from quant_fund.research.concordance import (
        format_concordance_table,
        run_concordance_eval,
        write_concordance_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    head_names = None if heads is None else [h.strip() for h in heads.split(",") if h.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        # The lane resolves the fleet and shards itself and raises ValueError on
        # an unknown name, so validating here would build every head twice.
        frame, receipt = run_concordance_eval(
            seed=base_seed,
            n_train=n_train,
            n_eval=n_eval,
            alpha=alpha,
            n_boot=n_boot,
            block=block,
            head_names=head_names,
            shard_names=shard_names,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_concordance_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_concordance_table(frame))
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command("multih-fleet")
def multih_fleet_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    models: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    horizons: str = typer.Option("1,5,20", help="Comma-separated h-step horizons (>=1)."),
    n_train: int = typer.Option(200, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(40, help="Trailing scored origins per horizon."),
    n: int = typer.Option(400, help="Synthetic series length per shard."),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Race h-step quantile constructions on the SYNTHETIC fleet and seal a receipt.

    For each head and horizon, scores the native multi-horizon fit against the
    iid-sqrt and empirical-ratio h-step extensions. Pinball, central-interval
    coverage and Kupiec p — proper scores only, never market evidence.
    """
    from quant_fund.research.fleet_eval import resolve_shard_generators
    from quant_fund.research.multih_fleet import (
        multih_factories,
        run_multih_fleet_eval,
        write_multih_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    taus = list(cfg.quantiles.levels)
    model_names = None if models is None else [m.strip() for m in models.split(",") if m.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        horizon_list = sorted({int(h) for h in horizons.split(",") if h.strip()})
    except ValueError as exc:
        raise typer.BadParameter(f"--horizons must be comma-separated integers: {exc}") from exc
    if not horizon_list or min(horizon_list) < 1:
        raise typer.BadParameter("--horizons must contain at least one horizon >= 1")
    try:
        factories = multih_factories(taus, base_seed, names=model_names)
        generators = resolve_shard_generators(shard_names)
        rows, receipt = run_multih_fleet_eval(
            factories,
            generators,
            taus=taus,
            horizons=horizon_list,
            n_train=n_train,
            n_eval=n_eval,
            n=n,
            seed=base_seed,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_multih_receipt(receipt, out_dir)
    ok = sum(1 for r in rows if r.status == "ok")
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(
        f"rows={len(rows)} ok={ok} errors={len(rows) - ok} "
        f"heads={len(factories)} shards={len(generators)} horizons={','.join(map(str, horizon_list))}"
    )
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command("expert-mixture")
def expert_mixture_cmd(
    config: Path = typer.Option(Path("configs/research.yaml")),
    heads: str | None = typer.Option(
        None, help="Comma-separated head names (default: full fleet registry)."
    ),
    shards: str | None = typer.Option(
        None, help="Comma-separated shard names (default: all synthetic shards)."
    ),
    n_train: int = typer.Option(512, help="Leading fit rows per shard."),
    n_eval: int = typer.Option(256, help="Trailing scored rows per shard."),
    alpha: float = typer.Option(0.05, help="Fixed-share switching rate."),
    seed: int | None = typer.Option(None, help="Base seed (default: train.random_seed)."),
    dev: bool = typer.Option(False, "--dev", help="Acknowledge dev-only use; required to run."),
    out_dir: Path = typer.Option(Path("receipts"), help="Receipt output directory."),
) -> None:
    """Run prediction-with-expert-advice over the SYNTHETIC fleet and seal a receipt.

    Mixes the fleet under uniform, exponentially-weighted-average and
    fixed-share weighting. Scored on regret against the best fixed expert in
    hindsight — a mixture is never claimed dominant, and nothing here is market
    evidence.
    """
    if not dev:
        raise typer.BadParameter(
            "expert-mixture is dev-only evidence tooling; pass --dev to acknowledge."
        )
    from quant_fund.research.expert_mixture import (
        MIXERS,
        format_expert_mixture_table,
        run_expert_mixture_eval,
        write_expert_mixture_receipt,
    )

    cfg = _cfg(config)
    base_seed = cfg.train.random_seed if seed is None else seed
    head_names = None if heads is None else [h.strip() for h in heads.split(",") if h.strip()]
    shard_names = None if shards is None else [s.strip() for s in shards.split(",") if s.strip()]
    try:
        # As in `concordance`: the lane resolves and validates the fleet itself.
        frame, receipt = run_expert_mixture_eval(
            seed=base_seed,
            n_train=n_train,
            n_eval=n_eval,
            alpha=alpha,
            head_names=head_names,
            shard_names=shard_names,
            taus=cfg.quantiles.levels,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    path = write_expert_mixture_receipt(receipt, out_dir)
    typer.echo(format_data_label(synthetic=True, data_source="SYNTHETIC"))
    typer.echo(format_expert_mixture_table(frame))
    typer.echo(f"mixers={','.join(MIXERS)}")
    typer.echo(f"verdict={receipt.get('verdict')}")
    typer.echo(f"receipt={path}")


@app.command()
def compare(
    run_a: Path = typer.Argument(..., exists=True, help="Receipt JSON or result dir for run A."),
    run_b: Path = typer.Argument(..., exists=True, help="Receipt JSON or result dir for run B."),
    higher_is_better: bool = typer.Option(
        False, help="Score direction: positive delta favors A (default: loss convention)."
    ),
    alpha: float = typer.Option(0.05, help="Paired bootstrap significance level."),
    n_boot: int = typer.Option(2000, help="Paired bootstrap replicates."),
    seed: int = typer.Option(7, help="Bootstrap seed."),
    min_paired: int = typer.Option(
        10, help="Below this paired n the verdict is 'insufficient paired observations'."
    ),
    series: str | None = typer.Option(
        None, help="Comma-separated series keys to restrict the comparison to."
    ),
    fmt: str = typer.Option("markdown", "--format", help="markdown or json."),
    out: Path | None = typer.Option(None, help="Write the report here instead of stdout."),
    receipt_out: Path | None = typer.Option(None, help="Also write a run_compare receipt here."),
    receipt_version: int = typer.Option(
        1, "--receipt-version", help="1 = run_compare.v1 blob (default), 2 = sealed receipt.v2."
    ),
) -> None:
    """Paired comparison of two stored research runs.

    Same lane as ``python -m quant_fund.research.compare`` (one shared
    implementation, so the two surfaces cannot drift). Diagnostic only: proper
    scores on stored receipts, never a live-trading or P&L claim. The envelope
    is labeled ``DATA_LABEL=UNKNOWN`` because a comparison of two stored runs
    cannot know whether either was SYNTHETIC.
    """
    from quant_fund.research.compare import (
        COMPARE_DATA_LABEL,
        build_compare_receipt,
        compare_runs,
    )

    if fmt not in ("markdown", "json"):
        raise typer.BadParameter("--format must be markdown or json")
    if receipt_version not in (1, 2):
        raise typer.BadParameter("--receipt-version must be 1 or 2")
    series_keys = None if series is None else [s.strip() for s in series.split(",") if s.strip()]
    try:
        comparison = compare_runs(
            run_a,
            run_b,
            higher_is_better=higher_is_better,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            min_paired=min_paired,
            series_keys=series_keys,
        )
    except (ValueError, TypeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    text = (
        json.dumps(comparison.to_dict(), indent=2, allow_nan=False)
        if fmt == "json"
        else comparison.to_markdown()
    )
    if out is not None:
        out.write_text(text + "\n")
        typer.echo(f"report={out}")
    else:
        typer.echo(text)
    if receipt_out is not None:
        document = build_compare_receipt(comparison, version=receipt_version)
        receipt_out.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        typer.echo(f"receipt={receipt_out}")
    typer.echo(format_data_label(synthetic=False, data_source=COMPARE_DATA_LABEL))
    counts = comparison.verdict_counts
    typer.echo(
        f"series={len(comparison.series)} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
