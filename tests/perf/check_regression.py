"""Fail when a benchmark's calibration-normalized time regresses.

Absolute milliseconds are not comparable across GitHub-hosted runners and a
developer machine. Each benchmark is scored as::

    ratio = median_seconds(benchmark) / median_seconds(test_calibration_matmul)

and compared with the same ratio in the committed baseline. A benchmark fails
when that ratio exceeds the baseline ratio by more than ``threshold`` and the
slowdown is not explained by the matmul running ahead of the other benches.
See ``_is_regression``.

The baseline JSON is whatever ``pytest-benchmark --benchmark-json`` wrote,
including ``machine_info``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _benchmarks(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    machine = payload.get("machine_info", {})
    medians: dict[str, float] = {}
    for row in payload.get("benchmarks", []):
        name = str(row.get("fullname") or row.get("name"))
        stats = row.get("stats") or {}
        median = stats.get("median")
        if median is None:
            raise SystemExit(f"{path}: {name} has no stats.median")
        medians[name] = float(median)
    return machine, medians


# Unmarked tests/perf cases. The CI job passes --ci so dropping one fails the gate.
_CI_SUFFIXES = (
    "test_calibration_matmul",
    "test_cross_section[24sym-252d]",
    "test_build_features[24sym-252d]",
    "test_resample[1sym-1y-5m]",
    "test_resample[1sym-1y-1d]",
    "test_security_master[8sym-20d]",
    "test_read_ohlcv_1m[8sym-20d]",
    "test_backtest[8sym-126d]",
    "test_dip_detect[40sym-1000d]",
    "test_dip_bench[20sym-800d]",
)


def _calibration_key(medians: dict[str, float]) -> str:
    matches = [name for name in medians if name.endswith("test_calibration_matmul")]
    if len(matches) != 1:
        raise SystemExit(f"expected one calibration benchmark, found {matches}")
    return matches[0]


def _is_regression(
    bench_factor: float, pack_factor: float, cal_factor: float, threshold: float
) -> bool:
    """True when a bench regressed beyond both the matmul and the bench pack.

    ``bench_factor`` and ``pack_factor`` are absolute slowdowns
    (current median / baseline median). ``cal_factor`` is the same ratio for
    ``test_calibration_matmul``. The historical gate is
    ``bench_factor / cal_factor > 1 + threshold``.

    That denominator is a 512×512 BLAS gemm. On a 64-core EPYC it measured
    6.03 ms and the gate passed; on a Xeon Platinum the same tree measured
    3.77 ms against a 4.00 ms baseline while the polars and Python benches
    barely moved, and three benches failed at 1.51×. The gemm does not track
    those benches across CPUs.

    A bench still fails when its matmul-normalized ratio exceeds the
    threshold and either the pack of benches does too (a uniform regression
    the gemm does not explain) or this bench is that far ahead of the pack
    (an isolated regression). A gemm that merely runs faster than the pack
    no longer fails every bench that moved together.
    """
    if cal_factor <= 0.0 or pack_factor <= 0.0 or bench_factor <= 0.0:
        return True
    limit = 1.0 + threshold
    if bench_factor / cal_factor <= limit:
        return False
    if pack_factor / cal_factor > limit:
        return True
    return bench_factor > pack_factor * limit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="require the fast-subset benchmark ids (see _CI_SUFFIXES)",
    )
    args = parser.parse_args(argv)
    if args.threshold < 0:
        raise SystemExit("threshold must be >= 0")
    current_machine, current = _benchmarks(args.current)
    baseline_machine, baseline = _benchmarks(args.baseline)
    current_key = _calibration_key(current)
    baseline_key = _calibration_key(baseline)
    current_cal = current[current_key]
    baseline_cal = baseline[baseline_key]
    if current_cal <= 0 or baseline_cal <= 0:
        raise SystemExit("calibration median must be positive")

    print(f"threshold: {args.threshold:.0%} above the calibration-normalized baseline")
    print(f"current machine: {_machine_line(current_machine)}")
    print(f"baseline machine: {_machine_line(baseline_machine)}")
    print(
        f"calibration median: current {current_cal * 1e3:.3f} ms, baseline {baseline_cal * 1e3:.3f} ms"
    )
    print()
    header = f"{'benchmark':<72} {'cur_ms':>10} {'base_ms':>10} {'ratio':>8} {'limit':>8}"
    print(header)
    failures: list[str] = []
    if args.ci:
        node_names = {name.split("::")[-1] for name in current}
        for suffix in _CI_SUFFIXES:
            if suffix not in node_names:
                failures.append(f"CI benchmark missing from this run: {suffix}")
    shared = sorted(
        name for name in current if name in baseline and name not in {current_key, baseline_key}
    )
    for name in sorted(current):
        if name in baseline or name.endswith("test_calibration_matmul"):
            continue
        failures.append(f"no baseline entry for {name}")
    cal_factor = current_cal / baseline_cal
    factors = {
        name: current[name] / baseline[name]
        for name in shared
        if not name.endswith("test_calibration_matmul")
    }
    pack_factor = statistics.median(factors.values()) if factors else cal_factor
    for name in shared:
        if name.endswith("test_calibration_matmul"):
            continue
        cur_ratio = current[name] / current_cal
        base_ratio = baseline[name] / baseline_cal
        limit = base_ratio * (1.0 + args.threshold)
        flag = ""
        if _is_regression(factors[name], pack_factor, cal_factor, args.threshold):
            flag = " FAIL"
            failures.append(
                f"{name}: ratio {cur_ratio:.3f} > limit {limit:.3f} "
                f"(current {current[name] * 1e3:.3f} ms, baseline {baseline[name] * 1e3:.3f} ms)"
            )
        short = name.split("::")[-1]
        relative = (cur_ratio / base_ratio) if base_ratio else float("inf")
        print(
            f"{short:<72} {current[name] * 1e3:10.3f} {baseline[name] * 1e3:10.3f} "
            f"{relative:8.3f} {1.0 + args.threshold:8.3f}{flag}"
        )
    print()
    if failures:
        print("regressions:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("no calibration-normalized regressions")
    return 0


def _machine_line(info: dict[str, Any]) -> str:
    cpu = info.get("cpu") or {}
    brand = cpu.get("brand_raw") or cpu.get("arch") or "unknown cpu"
    python = info.get("python_version") or "unknown python"
    system = info.get("system") or info.get("platform") or ""
    return f"{brand}; python {python}; {system}"


if __name__ == "__main__":
    sys.exit(main())
