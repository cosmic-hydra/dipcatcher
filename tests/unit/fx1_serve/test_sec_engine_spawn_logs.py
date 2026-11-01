"""SYNTHETIC engine spawn error redaction regression."""

from __future__ import annotations

import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

from fx1.serve.backends import LocalFx1Backend


def test_spawn_error_does_not_expose_command_or_os_error() -> None:
    secret = "SYNTHETIC-credential-123"
    backend = object.__new__(LocalFx1Backend)
    backend._serve_cmd = f"engine --api-key {secret}"
    backend._root = Path("SYNTHETIC-checkpoint")
    with (
        patch("fx1.serve.backends.subprocess.Popen", side_effect=OSError(secret)),
        pytest.raises(RuntimeError, match="failed to spawn local fx-1 engine") as caught,
    ):
        backend._spawn_engine()

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in rendered
    assert secret not in str(caught.value)
