"""Minimal CLI: ``python -m quant_fund.data.quality.cli bars.parquet [--out r.json]``.

Read-only research tooling; exits 1 when the dataset fails any check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from quant_fund.data.quality import report_json, score_bars


def main(argv: list[str] | None = None) -> int:
    from quant_fund import __version__

    parser = argparse.ArgumentParser(prog="data-quality")
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"data-quality {__version__}",
    )
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.parquet.is_file():
        sys.stderr.write(f"error: file not found: {args.parquet}\n")
        return 2
    try:
        frame = pl.read_parquet(args.parquet)
    except (OSError, pl.exceptions.PolarsError) as exc:
        sys.stderr.write(f"error: cannot read parquet {args.parquet}: {exc}\n")
        return 2
    report = score_bars(frame, dataset=str(args.parquet))
    text = report_json(report)
    if args.out is not None:
        try:
            args.out.write_text(text, encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2
    else:
        sys.stdout.write(text)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
