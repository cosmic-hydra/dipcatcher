"""Smoke entry point: ``python -m quant_fund.parity smoke``.

Runs a tiny SYNTHETIC tape through the shared strategy on both clocks and
refuses to exit 0 unless the report is a zero-divergence simulated replay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_fund.parity.session import MarketSession


def _smoke_session() -> MarketSession:
    from datetime import UTC, datetime, timedelta

    from quant_fund.parity.session import Bar, MarketSession

    bars: list[Bar] = []
    for day in range(4):
        stamp = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day)
        for sid, px in (("A", 100.0 + day), ("B", 50.0 + 0.25 * day)):
            bars.append(
                Bar(
                    security_id=sid,
                    event_time=stamp,
                    open=px,
                    high=px,
                    low=px,
                    close=px,
                    volume=1_000_000.0,
                    source="synthetic",
                    revision_id="r0",
                    available_time=stamp,
                    adv=100_000_000.0,
                    vol_20=0.02,
                )
            )
    return MarketSession(bars)


def smoke(out: Path) -> None:
    """Write a zero-divergence SYNTHETIC report or raise."""
    from quant_fund.config.loader import load_config
    from quant_fund.parity.reference import run_backtest_session
    from quant_fund.parity.replay import ReplayOptions
    from quant_fund.parity.report import assert_clean_report, build_report, write_report
    from quant_fund.parity.shadow import run_shadow_session
    from quant_fund.parity.strategy import FixedWeightStrategy

    config = load_config("configs/paper.yaml")
    session = _smoke_session()
    strategy = FixedWeightStrategy({"A": 0.05, "B": -0.02})
    options = ReplayOptions(pacing="accelerated")
    backtest = run_backtest_session(session, strategy, config, options=options)
    shadow = run_shadow_session(session, strategy, config, options=options)
    report = build_report(
        backtest,
        shadow,
        backtest_strategy=strategy,
        shadow_strategy=strategy,
    )
    assert_clean_report(report)
    paths = write_report(out, report)
    print(
        "parity smoke: zero divergences, research_only, live_pnl_claim=false, "
        f"report={paths['markdown']}"
    )


def main(argv: list[str] | None = None) -> None:
    from quant_fund import __version__

    parser = argparse.ArgumentParser(prog="python -m quant_fund.parity")
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"parity {__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke", help="SYNTHETIC zero-divergence smoke")
    smoke_parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/metadata/parity-smoke"),
        help="Directory for the smoke report",
    )
    args = parser.parse_args(argv)
    if args.command == "smoke":
        try:
            smoke(args.out)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"parity smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
