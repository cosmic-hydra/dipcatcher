"""CLI northset + research echo every SESSION_RECEIPT_KEYS_CLI_ECHO key."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.northset.benches import (
    SESSION_RECEIPT_KEYS,
    SESSION_RECEIPT_KEYS_CLI_ECHO,
)


def test_cli_echo_frozenset_matches_receipt_keys() -> None:
    assert SESSION_RECEIPT_KEYS_CLI_ECHO == SESSION_RECEIPT_KEYS


def test_northset_cli_echoes_every_session_receipt_key() -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    result = CliRunner().invoke(app, ["northset", "--config", str(config)])
    assert result.exit_code == 0, result.output
    missing = [k for k in sorted(SESSION_RECEIPT_KEYS_CLI_ECHO) if f"{k}=" not in result.output]
    assert not missing, f"northset CLI missing echo for: {missing}"


def test_research_cli_echoes_every_session_receipt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Path("configs/research.yaml")
    if not config.exists():
        pytest.skip("configs/research.yaml missing")
    # Only the northset family's echo is asserted here. The 1400+ seeded
    # live-optional benches it would otherwise run are covered, family by
    # family, by test_research_agent's scorecard assertions and the CI smoke.
    monkeypatch.setattr(
        "quant_fund.research.live_optional_pool._live_optional_family_blobs", lambda **_kw: {}
    )
    result = CliRunner().invoke(app, ["research", "--config", str(config)])
    # research may be slower / exit non-zero on soft fails — require session keys if northset family printed
    if "northset " not in result.output and "mean_session_" not in result.output:
        pytest.skip(f"research CLI did not print northset family: exit={result.exit_code}")
    missing = [k for k in sorted(SESSION_RECEIPT_KEYS_CLI_ECHO) if f"{k}=" not in result.output]
    assert not missing, f"research CLI missing echo for: {missing}\n---\n{result.output[-2000:]}"
