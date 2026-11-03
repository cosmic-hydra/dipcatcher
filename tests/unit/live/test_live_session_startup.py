"""Startup credential checks for live sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant_fund.live.brokers import AlpacaBroker, LoopbackBroker
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
FAKE_KEY = "synthetic-key-id"
FAKE_SECRET = "synthetic-secret"
ALPACA_ENV = (
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def _clear_alpaca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ALPACA_ENV:
        monkeypatch.delenv(name, raising=False)


def _config(state_dir: Path, venue: str) -> LiveConfig:
    return LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": venue,
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(state_dir),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )


def _no_network(*args: Any) -> tuple[int, bytes]:
    raise AssertionError(f"unexpected network call during startup: {args!r}")


@pytest.mark.parametrize("venue", ["alpaca-paper", "alpaca-live"])
def test_missing_alpaca_credentials_fail_before_state_creation(
    tmp_path: Path, venue: str
) -> None:
    state_dir = tmp_path / "state"

    with pytest.raises(LiveGateError, match="broker credentials are not set"):
        open_session(
            _config(state_dir, venue),
            PRETRADE,
            state_dir=state_dir,
            now_ns=lambda: 0,
            transport=_no_network,
        )

    assert not state_dir.exists()
    assert not (state_dir / "hmac.key").exists()


@pytest.mark.parametrize("venue", ["alpaca-paper", "alpaca-live"])
@pytest.mark.parametrize(
    ("name", "value"),
    [("ALPACA_API_KEY_ID", FAKE_KEY), ("ALPACA_API_SECRET_KEY", FAKE_SECRET)],
)
def test_incomplete_alpaca_credentials_fail_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, venue: str, name: str, value: str
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv(name, value)

    with pytest.raises(LiveGateError, match="credentials") as exc_info:
        open_session(
            _config(state_dir, venue),
            PRETRADE,
            state_dir=state_dir,
            now_ns=lambda: 0,
            transport=_no_network,
        )

    assert value not in str(exc_info.value)
    assert not state_dir.exists()
    assert not (state_dir / "hmac.key").exists()


@pytest.mark.parametrize("venue", ["alpaca-paper", "alpaca-live"])
def test_valid_synthetic_alpaca_credentials_construct_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, venue: str
) -> None:
    state_dir = tmp_path / "state"
    monkeypatch.setenv("ALPACA_API_KEY_ID", FAKE_KEY)
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", FAKE_SECRET)

    session = open_session(
        _config(state_dir, venue),
        PRETRADE,
        state_dir=state_dir,
        now_ns=lambda: 0,
        transport=_no_network,
    )

    assert isinstance(session.broker, AlpacaBroker)
    assert session.broker.venue == venue
    assert session.broker.credentials_present is True
    assert (state_dir / "hmac.key").is_file()


@pytest.mark.parametrize("venue", ["alpaca-paper", "alpaca-live"])
def test_injected_alpaca_broker_needs_no_environment_credentials(
    tmp_path: Path, venue: str
) -> None:
    state_dir = tmp_path / "state"
    broker = AlpacaBroker(venue, (FAKE_KEY, FAKE_SECRET), transport=_no_network)

    session = open_session(
        _config(state_dir, venue),
        PRETRADE,
        state_dir=state_dir,
        now_ns=lambda: 0,
        broker=broker,
    )

    assert session.broker is broker


def test_loopback_needs_no_broker_environment(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"

    session = open_session(
        _config(state_dir, "loopback"),
        PRETRADE,
        state_dir=state_dir,
        now_ns=lambda: 0,
        transport=_no_network,
    )

    assert isinstance(session.broker, LoopbackBroker)
    assert (state_dir / "hmac.key").is_file()
