"""Command line for the agent-market simulator.

``python -m quant_fund.market_sim report`` writes the measurement JSON used
by ``docs/MARKET_SIM.md``. Output is a simulation diagnostic. ``--help`` and
``--version`` do not import the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    from quant_fund import __version__

    parser = argparse.ArgumentParser(prog="python -m quant_fund.market_sim")
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"market-sim {__version__}",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench = sub.add_parser("bench", help="time the matching core")
    bench.add_argument("--events", type=int, default=1_000_000)
    bench.add_argument("--seed", type=int, default=1)

    stylized = sub.add_parser("stylized", help="run the pre-registered fact tests")
    stylized.add_argument("--skip-impact", action="store_true")

    sub.add_parser("scenarios", help="run the five scenarios at the default length")
    sub.add_parser("stress", help="stress momentum and mean-reversion weights")

    report = sub.add_parser("report", help="bench, stylized facts, scenarios, and stress")
    report.add_argument("--events", type=int, default=1_000_000)
    report.add_argument("--out", type=Path, default=None)
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _dispatch(args: argparse.Namespace) -> None:
    from quant_fund.market_sim.harness import (
        lightspeed_momentum_weight,
        mean_reversion_weight,
        stress_strategy,
    )
    from quant_fund.market_sim.native import matching_benchmark
    from quant_fund.market_sim.scenarios import SCENARIOS, run_scenario, scenario_spread
    from quant_fund.market_sim.stylized import run_stylized_validation
    from quant_fund.utils.atomicio import atomic_write_text

    def scenario_blob(name: str) -> dict[str, object]:
        result = run_scenario(name)
        blob: dict[str, object] = dict(result.summary())
        blob["hook"] = result.hook
        return blob

    if args.cmd == "bench":
        _print(matching_benchmark(args.events, args.seed))
        return
    if args.cmd == "stylized":
        _print(run_stylized_validation(measure=not args.skip_impact))
        return
    if args.cmd == "scenarios":
        payload: dict[str, object] = {name: scenario_blob(name) for name in SCENARIOS}
        payload["spread"] = scenario_spread()
        _print(payload)
        return
    if args.cmd == "stress":
        _print(
            {
                "momentum": stress_strategy(lightspeed_momentum_weight, name="lightspeed_momentum"),
                "mean_reversion": stress_strategy(mean_reversion_weight, name="mean_reversion"),
            }
        )
        return
    if args.cmd == "report":
        body = {
            "benchmark": matching_benchmark(args.events, 1),
            "stylized": run_stylized_validation(),
            "scenarios": {name: scenario_blob(name) for name in SCENARIOS},
            "spread": scenario_spread(),
            "stress": {
                "momentum": stress_strategy(lightspeed_momentum_weight, name="lightspeed_momentum"),
                "mean_reversion": stress_strategy(mean_reversion_weight, name="mean_reversion"),
            },
        }
        text = json.dumps(body, indent=2, sort_keys=True, default=str)
        if args.out is not None:
            atomic_write_text(args.out, text)
        print(text)
        return
    raise SystemExit(f"unknown command {args.cmd}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        _dispatch(args)
    except OSError as exc:
        sys.stderr.write(f"error: {exc}\n")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
