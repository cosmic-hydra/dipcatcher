"""Dipcatcher lab CLI.

``app`` is loaded on first access so ``python -m quant_fund.cli.main`` does not
warn about the module already being imported by this package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from quant_fund.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
