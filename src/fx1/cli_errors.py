"""User-facing fx-1 CLI failures, plus --version on every command.

Mirrors ``quant_fund.utils.cli_errors`` without importing the harness. The
architecture boundary forbids that direction.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import click
from typer.core import TyperGroup


def user_error_message(exc: BaseException) -> str | None:
    """Return a single-line message, or None when ``exc`` is not a user error."""
    if isinstance(exc, FileNotFoundError):
        return f"file not found: {_target(exc)}"
    if isinstance(exc, PermissionError):
        return f"permission denied: {_target(exc)}"
    if isinstance(exc, IsADirectoryError):
        return f"expected a file: {_target(exc)}"
    if isinstance(exc, NotADirectoryError):
        return f"expected a directory: {_target(exc)}"
    if isinstance(exc, UnicodeDecodeError):
        return "file is not valid UTF-8"
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON: {exc.msg} (line {exc.lineno})"
    yaml_message = _yaml_message(exc)
    if yaml_message is not None:
        return yaml_message
    return _validation_message(exc)


def version_requested(args: Sequence[str] | None) -> bool:
    """True when ``--version`` or ``-V`` appears before a ``--`` end-of-options marker."""
    argv = list(sys.argv[1:] if args is None else args)
    for token in argv:
        if token == "--":
            return False
        if token in {"--version", "-V"}:
            return True
    return False


class GuardedTyperGroup(TyperGroup):
    """Typer group that reports user errors without a traceback."""

    program = "fx1"

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        if version_requested(args):
            from fx1 import __version__

            click.echo(f"{self.program} {__version__}")
            raise SystemExit(0)
        try:
            return super().main(
                args=args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        except SystemExit:
            raise
        except BrokenPipeError:
            raise SystemExit(1) from None
        except Exception as exc:
            message = user_error_message(exc)
            if message is None:
                raise
            click.echo(f"error: {message}", err=True)
            raise SystemExit(2) from None


class Fx1TyperGroup(GuardedTyperGroup):
    program = "fx1"


class FxiTyperGroup(GuardedTyperGroup):
    program = "fxi"


def _target(exc: OSError) -> object:
    if exc.filename:
        return exc.filename
    return exc.strerror or exc


def _yaml_message(exc: BaseException) -> str | None:
    import yaml

    if not isinstance(exc, yaml.YAMLError):
        return None
    text = str(exc).splitlines()
    detail = text[0] if text else type(exc).__name__
    return f"invalid YAML: {detail}"


def _validation_message(exc: BaseException) -> str | None:
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return None
    parts: list[str] = []
    errors = exc.errors()
    for err in errors[:3]:
        loc = ".".join(str(item) for item in err.get("loc", ())) or "value"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    extra = len(errors) - len(parts)
    suffix = f" (+{extra} more)" if extra else ""
    return "invalid value: " + "; ".join(parts) + suffix
