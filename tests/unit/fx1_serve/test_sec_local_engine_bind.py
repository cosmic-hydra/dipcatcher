"""SYNTHETIC listener exposure checks for the unauthenticated local engine."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fx1.serve.local_engine import serve


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "::", "localhost", "example.com"])
def test_unauthenticated_engine_refuses_non_literal_loopback(host: str) -> None:
    with (
        patch("fx1.serve.local_engine._EngineHTTPServer") as server,
        pytest.raises(ValueError, match="literal loopback"),
    ):
        serve(None, host, 0, watch_orphan=False)  # type: ignore[arg-type]
    server.assert_not_called()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_engine_can_bind_literal_loopback(host: str) -> None:
    with patch("fx1.serve.local_engine._EngineHTTPServer") as server:
        assert serve(None, host, 0, watch_orphan=False) == 0  # type: ignore[arg-type]
    server.assert_called_once()
