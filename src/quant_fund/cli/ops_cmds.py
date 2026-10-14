"""API, paper, monitor, and sim-live commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .app import app
from .support import _cfg


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000) -> None:
    import os

    import uvicorn

    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    if host not in loopback_hosts and not os.environ.get("QUANT_API_KEY"):
        raise typer.BadParameter(
            "non-loopback API binding requires QUANT_API_KEY; refusing unauthenticated exposure"
        )
    uvicorn.run("quant_fund.api.app:app", host=host, port=port, reload=False)


def _reject_halt_flags(halt: bool, clear_halt: bool) -> None:
    """Reject the mutually exclusive kill-switch overrides."""
    if halt and clear_halt:
        raise typer.BadParameter("pass only one of --halt / --clear-halt")


@app.command()
def paper(
    config: Path = typer.Option(Path("configs/paper.yaml")),
    max_steps: int | None = typer.Option(None, help="Cap replay steps (paper accelerated)."),
    shadow: bool = typer.Option(True, help="Enable shadow challenger slot (no capital)."),
    wall_clock: bool = typer.Option(False, help="Use wall clock instead of bar replay."),
    halt: bool = typer.Option(False, help="Force kill switch HALT_NEW_ORDERS for this run."),
    resume: bool = typer.Option(False, help="Resume from broker_state.json for --run-id / latest."),
    run_id: str | None = typer.Option(None, help="Paper run_id (new or resume target)."),
    clear_halt: bool = typer.Option(
        False, help="Force kill switch ENABLED (overrides config / pairs with --resume)."
    ),
    from_start: bool = typer.Option(
        False,
        help="Use earliest decision dates (multi-day grind); default prefers latest window.",
    ),
    forward_stage: str | None = typer.Option(
        None, help="Paired paper: commitment, freeze, decide, execute, interrupt, or verify."
    ),
    forward_run: Path | None = typer.Option(None, help="Forward paper run directory."),
    forward_packet: Path | None = typer.Option(
        None, help="Externally timestamped close/open packet."
    ),
    forward_attestation: Path | None = typer.Option(None, help="External protocol freeze record."),
    forward_spec: Path = typer.Option(Path("configs/net_tournament.json")),
    forward_benchmark: Path = typer.Option(Path("data/metadata/real_benchmark/us_wide_20260925")),
    forward_tournament: Path = typer.Option(Path("data/metadata/net_tournament/us_wide_20260925")),
    forward_reason: str | None = typer.Option(
        None, help="Interruption reason: no_feed/downtime/missing_name/bad_timestamp/other."
    ),
) -> None:
    """Phase 17 paper / shadow loop with simulated broker (no live fills)."""
    import json

    import polars as pl

    if forward_stage is not None:
        from quant_fund.paper import forward_shadow

        if forward_stage not in {
            "commitment",
            "freeze",
            "decide",
            "execute",
            "interrupt",
            "verify",
        }:
            raise typer.BadParameter(
                "--forward-stage must be commitment, freeze, decide, execute, interrupt or verify"
            )
        if forward_stage != "commitment" and forward_run is None:
            raise typer.BadParameter("--forward-run is required")
        if forward_stage == "freeze" and forward_attestation is None:
            raise typer.BadParameter("--forward-attestation is required to freeze")
        if forward_stage in {"decide", "execute"} and forward_packet is None:
            raise typer.BadParameter("--forward-packet is required")
        if forward_stage == "interrupt" and forward_reason is None:
            raise typer.BadParameter("--forward-reason is required for an interruption")
        try:
            if forward_stage in {"commitment", "freeze"}:
                forward_result = forward_shadow.prepare(
                    forward_spec,
                    forward_benchmark,
                    forward_tournament,
                    forward_attestation if forward_stage == "freeze" else None,
                    forward_run if forward_stage == "freeze" else None,
                )
            elif forward_stage == "decide":
                forward_result = forward_shadow.decide(forward_run, forward_packet)  # type: ignore[arg-type]
            elif forward_stage == "execute":
                forward_result = forward_shadow.execute(forward_run, forward_packet)  # type: ignore[arg-type]
            elif forward_stage == "interrupt":
                forward_result = forward_shadow.interrupt(forward_run, forward_reason)  # type: ignore[arg-type]
            else:
                forward_result = forward_shadow.verify(forward_run)  # type: ignore[arg-type]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            phase_only_error = str(exc).startswith(
                (
                    "next-open execution must reconcile",
                    "a prior close decision must be recorded",
                    "forward paper run is already interrupted",
                )
            )
            if (
                forward_stage in {"decide", "execute"}
                and forward_run is not None
                and not phase_only_error
            ):
                reason = (
                    "no_feed"
                    if isinstance(exc, FileNotFoundError)
                    else "missing_name"
                    if "missing/extra" in str(exc)
                    else "bad_timestamp"
                    if "timestamp" in str(exc) or "cutoff" in str(exc)
                    else "other"
                )
                try:
                    stopped = forward_shadow.interrupt(
                        forward_run,
                        reason,
                        attempted_stage=forward_stage,
                        attempted_packet=forward_packet,
                        error=str(exc),
                    )
                    typer.echo(f"interruption_receipt={stopped['receipt_sha256']}")
                except (OSError, ValueError, KeyError, TypeError) as stop_error:
                    typer.echo(f"interruption_not_recorded={stop_error}")
            raise typer.BadParameter(str(exc)) from exc
        typer.echo("DATA_LABEL=PROSPECTIVE_PACKET_UNVERIFIED")
        typer.echo(
            "LOCAL_LEDGER_VERIFY_ONLY; external feed/calendar and strategy replay unverified"
        )
        displayed = {
            key: value
            for key, value in forward_result.items()
            if key
            in {
                "protocol_commitment_sha256",
                "git_revision",
                "git_worktree_sha256",
                "exchange_schedule_sha256",
                "last_historical_warmup_session",
                "receipt_sha256",
                "stage",
                "seq",
                "session",
                "kind",
                "valid",
                "state",
                "paired_sessions",
                "minimum",
                "interruption_reason",
                "external_attestation_verified",
                "independent_strategy_replay",
                "forward_evidence_accepted",
                "research_only",
                "live_pnl_claim",
                "errors",
            }
        }
        typer.echo(json.dumps(displayed, indent=2, allow_nan=False))
        if forward_result.get("valid") is False:
            raise typer.Exit(code=1)
        return

    from quant_fund.features.engine import build_features
    from quant_fund.paper.ledger import latest_run_id
    from quant_fund.paper.loop import build_scaled_challenger_weights, run_paper_loop
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel, decision_dates

    cfg = _cfg(config)
    _reject_halt_flags(halt, clear_halt)
    if halt:
        cfg.kill_switch.state = "HALT_NEW_ORDERS"
    if clear_halt:
        cfg.kill_switch.state = "ENABLED"
    bars = ensure_silver(cfg)
    feat = build_features(bars, cfg)
    # Same gold-panel overlap contract as execution-sensitivity/backtest:
    # warmup and label-tail dates have no panel rows, and optimize_asof
    # fails closed on them. Intersect before any --max-steps tail slice.
    dates = decision_dates(cfg, feat["event_time"].unique().sort().to_list())
    if not dates:
        raise typer.BadParameter("no decision dates on the causal gold panel")
    steps = max_steps if max_steps is not None else cfg.paper.max_steps
    resume_id = run_id or (
        latest_run_id(cfg.data.root, cfg.paper.ledger_subdir) if resume else None
    )
    if resume and not resume_id:
        raise typer.BadParameter("--resume needs --run-id or a prior latest_run.json")
    if steps is not None and not resume and not from_start:
        # Use the *latest* window so warmup/cov history exists (not day-0 cash).
        n = max(int(steps) + 2, 3)
        dates = dates[-n:]
    elif steps is not None and (resume or from_start):
        # Sequential: keep full history for cov; loop itself caps steps.
        pass
    weights = build_causal_weight_panel(cfg, dates)
    shadow_w = None
    enable_shadow = shadow and cfg.paper.enable_shadow
    if enable_shadow:
        shadow_w = weights.with_columns(
            (pl.col("target_weight") * float(cfg.paper.shadow_scale) * 0.85).alias("target_weight")
        )
    # Extra named scaled challengers from paper.challenger_scales (research-only).
    # Primary shadow above stays for ledger / promotion_dry_run compatibility.
    built = build_scaled_challenger_weights(weights, list(cfg.paper.challenger_scales or []))
    challenger_w = built or None
    result = run_paper_loop(
        feat,
        cfg,
        champion_weights=weights,
        shadow_weights=shadow_w,
        challenger_weights=challenger_w,
        initial_nav=float(cfg.paper.initial_nav),
        max_steps=steps,
        use_wall_clock=wall_clock or cfg.paper.use_wall_clock,
        run_id=run_id,
        resume=resume,
        resume_run_id=resume_id if resume else None,
        prefer_latest=not (resume or from_start),
    )
    typer.echo(f"DATA_LABEL={result.source_note}")
    if result.source_note == "SYNTHETIC":
        typer.echo("SYNTHETIC — paper ledger is simulated research/infrastructure only.")
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"divergence={result.divergence}")
    m = result.metrics
    typer.echo(
        json.dumps(
            {
                "n_steps": m.get("n_steps"),
                "n_steps_this_run": m.get("n_steps_this_run"),
                "n_fills": m.get("n_fills"),
                "risk_gate_rejects": m.get("risk_gate_rejects"),
                "kill_switch_halts": m.get("kill_switch_halts"),
                "resumed": m.get("resumed"),
                "execution": m.get("execution"),
                "var_es": m.get("var_es"),
                "exposure": m.get("exposure"),
                "stress": m.get("stress"),
                "portfolio_conformal": m.get("portfolio_conformal"),
                "promotion_dry_run": m.get("promotion_dry_run"),
                "research_only": True,
                "live_pnl_claim": False,
                "paths": result.paths,
            },
            indent=2,
            default=str,
        )
    )


@app.command()
def monitor(
    config: Path = typer.Option(Path("configs/paper.yaml")),
    run_id: str | None = typer.Option(None, help="Paper run_id (default: latest_run.json)."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of markdown."),
    out: Path | None = typer.Option(None, "--out", help="Write output to this path."),
) -> None:
    """Ops snapshot over the latest paper run: limits, staleness, kill state."""
    import json

    import polars as pl

    from quant_fund.monitoring.dashboard import ops_snapshot, render_markdown
    from quant_fund.paper.ledger import latest_run_id, load_broker_state, paper_root

    cfg = _cfg(config)
    rid = run_id or latest_run_id(cfg.data.root, cfg.paper.ledger_subdir)
    if not rid:
        raise typer.BadParameter("no paper run found — pass --run-id or run `dipcatcher paper`")
    state = load_broker_state(cfg.data.root, rid, cfg.paper.ledger_subdir)
    if not state:
        raise typer.BadParameter(f"no broker_state.json for run_id={rid}")
    champion = state.get("champion") or {}

    equity_path = paper_root(cfg.data.root, cfg.paper.ledger_subdir) / rid / "equity.parquet"
    nav = peak = None
    asof = None
    if equity_path.is_file():
        eq = pl.read_parquet(equity_path).sort("asof")
        if eq.height:
            nav = float(eq["nav"][-1])
            peak = float(eq["nav"].to_numpy().max())
            asof = eq["asof"][-1]
    marks = champion.get("last_marks") or {}
    shares = champion.get("shares") or {}
    broker_nav = float(champion.get("cash", 0.0)) + sum(
        float(q) * float(marks.get(s, 0.0)) for s, q in shares.items()
    )
    if nav is None:
        # Fall back to broker cash + marks for a resumed-but-unflushed run.
        nav = broker_nav
    # Reconcile broker state vs the last ledger equity row.
    recon_mismatches = None
    if peak is not None and nav > 0:
        recon_mismatches = 0 if abs(broker_nav - nav) / nav <= 1e-6 else 1

    snap = ops_snapshot(
        nav=nav,
        cash=float(champion.get("cash", 0.0)),
        positions={str(k): float(v) for k, v in shares.items()},
        marks={str(k): float(v) for k, v in marks.items()},
        config=cfg,
        asof=asof,
        mark_age_bars=state.get("mark_ages"),
        n_open_orders=len(champion.get("open_orders") or []),
        kill_switch_state=champion.get("kill_state"),
        peak_nav=peak,
        recon_mismatches=recon_mismatches,
    )
    from quant_fund.utils.atomicio import atomic_write_text

    snap["run_id"] = rid
    text = json.dumps(snap, indent=2, default=str) if json_out else render_markdown(snap)
    if out is not None:
        atomic_write_text(out, text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)
    # Nagios-style exit codes so schedulers/alerting can consume status.
    status = snap["overall_status"]
    if status == "breach":
        raise typer.Exit(2)
    if status == "warn":
        raise typer.Exit(1)


@app.command("sim-live")
def sim_live(
    config: Path = typer.Option(Path("configs/sim_live.yaml")),
    interval: str = typer.Option("4h", help="Bar interval suffix: 4h | 1d"),
    symbols: str = typer.Option(
        "BNBUSDT,BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT", help="Comma-separated symbols"
    ),
    spec: str = typer.Option(
        "fhs", help="Champion forecaster: fhs|evt|garch_t|egarch_l|empirical|ewma_emp"
    ),
    mode: str = typer.Option("long_flat", help="Champion policy: long_flat|symmetric"),
    challenger: list[str] = typer.Option(
        [],
        "--challenger",
        help="Book 'name:spec:mode[:entry_bps[:kappa]]' (repeatable; shadows in full runs)",
    ),
    gross: float = typer.Option(1.0, help="Book gross cap"),
    name_cap: float = typer.Option(0.25, help="Per-name |weight| cap"),
    kappa: float = typer.Option(0.30, help="Size per unit predicted Sharpe"),
    entry_bps: float = typer.Option(
        20.0, help="|mu| entry gate in per-bar bps (z units if --gate-on edge)"
    ),
    gate_on: str = typer.Option("mu", help="Gate metric: 'mu' (bps) | 'edge' (mu/disp z-score)"),
    deadband: float = typer.Option(0.01, help="Min |Δtarget| before re-emit"),
    window: int = typer.Option(750, help="Trailing returns window per origin"),
    tail_bars: int | None = typer.Option(None, help="Use only last N bars per asset"),
    eval_tail: int | None = typer.Option(
        None,
        "--eval-tail",
        help="OOS eval: panels on full history, loop/bench on last N shared dates",
    ),
    sizing: str = typer.Option("edge", help="Sizing law: 'edge' (k*edge) | 'risk' (k*edge/disp)"),
    book_vol_target: float | None = typer.Option(
        None, help="Champion book vol target (per-bar); scales Σ|w·disp| toward it"
    ),
    tail_gate: float | None = typer.Option(
        None, help="Tail conviction gate (return units): longs need q_lo > -tail_gate"
    ),
    persist: int = typer.Option(1, help="Consecutive gate-passing bars before entry"),
    exit_persist: int = typer.Option(
        1, help="Consecutive gate-FAILING bars before exit (1 = instant)"
    ),
    mkt_disp_cut: float | None = typer.Option(
        None, help="Market vol breaker: flat book when median cross-asset disp exceeds this"
    ),
    top_k: int | None = typer.Option(None, help="Keep only the k largest |target| names per date"),
    gate_out: float | None = typer.Option(
        None, help="Exit threshold hysteresis (held names exit below this)"
    ),
    meta_min: float | None = typer.Option(
        None, help="Entry gate: rolling signal-Sharpe of the name >= this"
    ),
    rebal_every: int = typer.Option(
        1, help="Emit targets only every k-th decision date (book carries otherwise)"
    ),
    breadth_gross: bool = typer.Option(
        False, help="Scale gross cap by fraction of names with edge > 0"
    ),
    leader_sid: str | None = typer.Option(
        None, help="Leadership gate: alts need this sid's edge > leader_edge_min"
    ),
    leader_edge_min: float = typer.Option(0.0, help="Leader edge threshold for alts"),
    max_steps: int | None = typer.Option(None, help="Cap decision steps"),
    run_id: str | None = typer.Option(None, help="Paper run_id"),
    resume: bool = typer.Option(
        False, help="Resume book from broker_state.json (new appended bars only)"
    ),
    n_jobs: int = typer.Option(-1, help="joblib parallelism for quantile panels"),
    bench: bool = typer.Option(
        True, "--bench/--no-bench", help="Bench every slot's book via run_backtest"
    ),
    bench_only: bool = typer.Option(
        False, "--bench-only", help="Skip paper loop; leaderboard of books only (fast)"
    ),
    shared_calendar: bool = typer.Option(
        True,
        "--shared-calendar/--no-shared-calendar",
        help="Clip all assets to the common span (off: each trades its own history)",
    ),
    out: Path = typer.Option(Path(".dsh-24x7/lane-simlive"), "--out"),
) -> None:
    """Simulated-live PnL: proven quantile forecasters trade the paper loop on real bars."""
    import json
    import subprocess

    from quant_fund.paper.quantile_signals import QuantilePolicy
    from quant_fund.paper.sim_live import StrategySlot, run_sim_live

    cfg = _cfg(config)
    try:
        git_sha = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001
        git_sha = None
    champion_policy = QuantilePolicy(
        mode=mode,
        kappa=kappa,
        gross_target=gross,
        name_cap=name_cap,
        cost_gate=entry_bps / 1e4 if gate_on == "mu" else entry_bps,
        deadband=deadband,
        gate_on=gate_on,
        sizing=sizing,
        book_vol_target=book_vol_target,
        tail_gate=tail_gate,
        persist_bars=persist,
        exit_persist=exit_persist,
        mkt_disp_cut=mkt_disp_cut,
        top_k=top_k,
        gate_out=gate_out,
        meta_min=meta_min,
        rebal_every=rebal_every,
        breadth_gross=breadth_gross,
        leader_sid=leader_sid,
        leader_edge_min=leader_edge_min,
    )
    slots = [StrategySlot(name=f"{spec}_{mode}", spec=spec, policy=champion_policy)]
    challengers: list[StrategySlot] = []
    for raw in challenger:
        parts = raw.split(":")
        if len(parts) < 3:
            raise typer.BadParameter("--challenger must be name:spec:mode[:entry_bps[:kappa]]")
        cname, cspec, cmode = parts[0], parts[1], parts[2]
        c_bps = float(parts[3]) if len(parts) > 3 else entry_bps
        c_kappa = float(parts[4]) if len(parts) > 4 else kappa
        c_gate_on = parts[5] if len(parts) > 5 else gate_on
        c_sizing = parts[6] if len(parts) > 6 else sizing
        # Trailing key=value overrides: bvt=<book_vol_target> tg=<tail_gate>
        # pb= xp= tk= rb= rvlb= (ints) · gross= nc= db= go= meta= ac= cut= le=
        # rvol= epow= (floats) · lead=<SID> (string)
        kv: dict[str, float] = {}
        kv_str: dict[str, str] = {}
        for extra in parts[7:]:
            if "=" in extra:
                k, v = extra.split("=", 1)
                if k.strip() == "lead":
                    kv_str["lead"] = v.strip().upper()
                else:
                    kv[k.strip()] = float(v)
        challengers.append(
            StrategySlot(
                name=cname,
                spec=cspec,
                policy=QuantilePolicy(
                    mode=cmode,
                    kappa=c_kappa,
                    gross_target=kv.get("gross", gross),
                    name_cap=kv.get("nc", name_cap),
                    cost_gate=c_bps / 1e4 if c_gate_on == "mu" else c_bps,
                    deadband=kv.get("db", deadband),
                    gate_on=c_gate_on,
                    sizing=c_sizing,
                    book_vol_target=kv.get("bvt"),
                    tail_gate=kv.get("tg"),
                    persist_bars=int(kv.get("pb", persist)),
                    exit_persist=int(kv.get("xp", exit_persist)),
                    mkt_disp_cut=kv.get("cut"),
                    top_k=int(kv["tk"]) if "tk" in kv else top_k,
                    gate_out=kv.get("go", gate_out),
                    meta_min=kv.get("meta", meta_min),
                    rebal_every=int(kv.get("rb", rebal_every)),
                    accel_min=kv.get("ac"),
                    leader_sid=kv_str.get("lead"),
                    leader_edge_min=kv.get("le", 0.0),
                    w_alpha=kv.get("wa", 1.0),
                    breadth_gross=bool(kv.get("bg", 0.0)),
                    edge_pow=kv.get("epow", 1.0),
                    rvol_target=kv.get("rvol"),
                    rvol_lookback=int(kv.get("rvlb", 20)),
                ),
            )
        )
    result = run_sim_live(
        bars_root=Path("data/raw/sources"),
        symbols=[s.strip().upper() for s in symbols.split(",") if s.strip()],
        interval=interval,
        config=cfg,
        champion=slots[0],
        challengers=challengers,
        window=window,
        tail_bars=tail_bars,
        eval_tail_bars=eval_tail,
        out_dir=out,
        run_id=run_id,
        max_steps=max_steps,
        git_sha=git_sha,
        n_jobs=n_jobs,
        resume=resume,
        resume_run_id=run_id if resume else None,
        bench=bench,
        bench_only=bench_only,
        shared_calendar=shared_calendar,
    )
    typer.echo(f"DATA_LABEL={result.receipt['data_label']}")
    typer.echo(f"run_id={result.run_id}")
    typer.echo(json.dumps(result.receipt["champion_equity_stats"], indent=2, default=str))
    typer.echo(json.dumps(result.receipt["loop_metrics"], indent=2, default=str))
    if result.receipt.get("book_stats"):
        typer.echo("book leaderboard (same bars/costs/gates, simulated):")
        rows = sorted(
            result.receipt["book_stats"].items(),
            key=lambda kv: kv[1].get("total_return", float("-inf")),
            reverse=True,
        )
        for name, st in rows:
            if st.get("status") == "ok":
                typer.echo(
                    f"  {name:<18} ret={st['total_return']:+.4%} "
                    f"sharpe={st['sharpe_simulated']:+.3f} maxDD={st['max_drawdown']:+.3%} "
                    f"vol={st['ann_vol']:.3%} fills={st.get('n_fills')}"
                )
            else:
                typer.echo(f"  {name:<18} {st.get('status', 'failed')}")
    typer.echo(f"receipt: {result.receipt_path}")
    typer.echo("SIMULATED — no live-PnL claim.")


@app.command()
def trade(
    op: str = typer.Argument(..., help="readiness or session"),
    state: Path | None = typer.Option(None, help="Paper session state path."),
    commands: Path | None = typer.Option(None, help="JSONL paper-session commands."),
    starting_cash: float = typer.Option(
        1_000_000.0, help="Simulated starting cash for a new session."
    ),
    resume: bool = typer.Option(False, help="Resume an existing paper session."),
    max_order_notional: float = typer.Option(1_000_000.0),
    max_position_notional: float = typer.Option(5_000_000.0),
    max_gross_notional: float = typer.Option(5_000_000.0),
    max_order_quantity: float = typer.Option(1_000_000.0),
) -> None:
    """Paper trading session. Live orders are refused.

    ``readiness`` prints why live capital is blocked. ``session`` applies a
    JSONL script of simulated commands (mark, submit, fill, cancel, kill,
    clear_kill, checkpoint). Prices in the script are caller-supplied.
    """
    import json

    from quant_fund.execution.trading_session import (
        SessionCommandError,
        SessionLimits,
        format_readiness,
        live_readiness_report,
        load_commands,
        run_session,
    )

    if op == "readiness":
        report = live_readiness_report()
        typer.echo("DATA_LABEL=SIMULATED")
        typer.echo(format_readiness(report))
        typer.echo(json.dumps(report, indent=2, allow_nan=False))
        return
    if op != "session":
        raise typer.BadParameter("op must be readiness or session")
    if state is None or commands is None:
        raise typer.BadParameter("session requires --state and --commands")
    try:
        limits = SessionLimits(
            max_order_quantity=max_order_quantity,
            max_order_notional=max_order_notional,
            max_position_notional=max_position_notional,
            max_gross_notional=max_gross_notional,
        )
        script = load_commands(commands)
        result = run_session(
            state_path=state,
            commands=script,
            resume=resume,
            starting_cash=starting_cash,
            limits=limits,
        )
    except SessionCommandError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("DATA_LABEL=SIMULATED")
    typer.echo("PAPER_SESSION_ONLY; live orders are refused")
    typer.echo(json.dumps(result, indent=2, allow_nan=False))
    if not result["consistent"]:
        raise typer.Exit(code=1)


__all__ = [
    "api",
    "monitor",
    "paper",
    "sim_live",
    "trade",
]
