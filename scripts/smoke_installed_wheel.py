#!/usr/bin/env python3
"""Check the wheel's console scripts from outside the source checkout.

Run with the Python interpreter into which the wheel was installed. A source
checkout or editable install must not satisfy this release smoke test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import distribution
from pathlib import Path

_SCRIPTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("fx1", ("--help",), "Usage"),
    ("quant", ("--help",), "research, never live"),
    ("dipcatcher", ("--help",), "research, never live"),
    ("fxi", ("--help",), "Usage"),
    ("verify-ledger", ("--help",), "Verify a hash-chained audit ledger"),
    ("mc-engine", ("--help",), "Usage"),
)
_PACKAGE_FILES = {
    "fx1/__init__.py",
    "quant_fund/__init__.py",
    "dipcatcher_cli/__init__.py",
    "quant_fund/py.typed",
}


def _installed_distribution() -> None:
    dist = distribution("fx-1")
    files = {str(path).replace("\\", "/") for path in dist.files or ()}
    missing = _PACKAGE_FILES - files
    if missing:
        raise RuntimeError(
            "fx-1 must be installed from its wheel; missing installed files: "
            + ", ".join(sorted(missing))
        )
    expected = {name for name, _, _ in _SCRIPTS}
    actual = {entry.name for entry in dist.entry_points if entry.group == "console_scripts"}
    if actual != expected:
        raise RuntimeError(
            f"console scripts differ: expected {sorted(expected)}, got {sorted(actual)}"
        )


def main() -> int:
    _installed_distribution()
    bin_dir = Path(sysconfig.get_path("scripts"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update({"NO_COLOR": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8"})
    with tempfile.TemporaryDirectory(prefix="dipcatcher-wheel-smoke-") as scratch:
        cwd = Path(scratch)
        env["FX1_CONFIG_DIR"] = str(cwd / "config")
        for name, args, expected in _SCRIPTS:
            script = bin_dir / (name + (".exe" if os.name == "nt" else ""))
            if not script.is_file():
                raise RuntimeError(f"installed console script missing: {script}")
            result = subprocess.run(
                [str(script), *args],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
            )
            if result.returncode != 0 or expected not in result.stdout:
                raise RuntimeError(
                    f"{name} {' '.join(args)} failed (exit {result.returncode}): "
                    f"{(result.stderr or result.stdout)[-1000:]}"
                )
        if (cwd / "config").exists():
            raise RuntimeError("help/capabilities smoke wrote configuration state")
    print("installed wheel console scripts: ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"installed wheel smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
