"""``python -m quant_fund.robustness`` runs the certification smoke check.

``--help`` and ``--version`` exit before the smoke imports. With no arguments
the smoke still runs, which is what CI calls.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from quant_fund import __version__

    parser = argparse.ArgumentParser(
        prog="python -m quant_fund.robustness",
        description=(
            "Robustness certification smoke. Research only. "
            "Does not submit orders and does not claim live profit."
        ),
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"robustness {__version__}",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("smoke",),
        default="smoke",
        help="Certification smoke (the default).",
    )
    args = parser.parse_args(argv)
    if args.command != "smoke":
        sys.stderr.write(f"error: unknown command {args.command}\n")
        return 2
    from quant_fund.robustness.smoke import main as smoke_main

    try:
        return smoke_main()
    except OSError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
