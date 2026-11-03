"""Startup hardening for live gateway configuration."""

from __future__ import annotations

import traceback
from pathlib import Path

import pytest
import yaml

from quant_fund.live.config import (
    LiveConfig,
    load_gateway_pretrade,
    load_live_config,
    resolve_pretrade_path,
)
from quant_fund.pretrade.config import load_pretrade_config

ROOT = Path(__file__).resolve().parents[3]
HMAC_KEY = b"0123456789abcdef0123456789abcdef"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "venue": "loopback",
        "starting_cash": 100_000.0,
        "calendar": "us_rth",
        "pretrade_config": "pretrade.yaml",
        "state_dir": "state/loopback",
    }
    payload.update(overrides)
    return payload


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _assert_safe_error(caught: pytest.ExceptionInfo[ValueError], *markers: str) -> None:
    message = str(caught.value)
    rendered = "".join(traceback.format_exception(caught.value))
    assert len(message) <= 512
    for marker in markers:
        assert marker not in message
        assert marker not in rendered


def test_checked_in_config_loads_and_preserves_calendar_behavior() -> None:
    config_path = ROOT / "configs" / "live.yaml"
    config = load_live_config(config_path)

    assert config.state_dir == "data/live/default"
    pretrade_path = resolve_pretrade_path(config, config_path)
    assert pretrade_path == ROOT / "configs" / "pretrade_risk.yaml"

    expected, _, _ = load_pretrade_config(pretrade_path, hmac_key=HMAC_KEY)
    us_rth, _, _ = load_gateway_pretrade(config, config_path, hmac_key=HMAC_KEY)
    assert us_rth == expected

    continuous_config = config.model_copy(update={"calendar": "continuous"})
    continuous, _, _ = load_gateway_pretrade(
        continuous_config,
        config_path,
        hmac_key=HMAC_KEY,
    )
    assert continuous.session.timezone == "UTC"
    assert continuous.session.open_minute == 0
    assert continuous.session.close_minute == 24 * 60
    assert continuous.session.weekends_open is True


@pytest.mark.parametrize("state_dir", [None, "", " \t "])
def test_state_dir_is_explicit_and_nonblank(tmp_path: Path, state_dir: str | None) -> None:
    payload = _payload()
    if state_dir is None:
        del payload["state_dir"]
    else:
        payload["state_dir"] = state_dir
    path = tmp_path / "live.yaml"
    _write_config(path, payload)

    with pytest.raises(ValueError) as caught:
        load_live_config(path)

    assert "state_dir" in str(caught.value)


def test_malformed_yaml_does_not_echo_synthetic_secret(tmp_path: Path) -> None:
    marker = "SYNTHETIC-MALFORMED-YAML-SECRET-" + "x" * 600
    path = tmp_path / "live.yaml"
    path.write_text(f'schema_version: 1\nvenue: "{marker}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML") as caught:
        load_live_config(path)

    _assert_safe_error(caught, marker)


def test_extra_key_and_value_do_not_echo_synthetic_secrets(tmp_path: Path) -> None:
    marker = "SYNTHETIC_EXTRA_KEY_SECRET_" + "k" * 600
    value_marker = "SYNTHETIC-EXTRA-VALUE-SECRET-" + "v" * 600
    path = tmp_path / "live.yaml"
    _write_config(path, _payload(**{marker: value_marker}))

    with pytest.raises(ValueError, match="extra keys") as caught:
        load_live_config(path)

    _assert_safe_error(caught, marker, value_marker)


@pytest.mark.parametrize("field", ["schema_version", "venue", "starting_cash", "cash_tolerance"])
def test_invalid_value_does_not_echo_synthetic_secret(tmp_path: Path, field: str) -> None:
    marker = f"SYNTHETIC-{field}-INVALID-SECRET-" + "x" * 600
    path = tmp_path / "live.yaml"
    _write_config(path, _payload(**{field: marker}))

    with pytest.raises(ValueError) as caught:
        load_live_config(path)

    assert field in str(caught.value)
    _assert_safe_error(caught, marker)


def test_relative_pretrade_path_ignores_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = tmp_path / "deploy"
    working = tmp_path / "working"
    deploy.mkdir()
    working.mkdir()
    name = "pretrade.yaml"
    (deploy / name).write_text("beside live config", encoding="utf-8")
    (working / name).write_text("working directory", encoding="utf-8")
    config_path = deploy / "live.yaml"
    _write_config(config_path, _payload())
    config = load_live_config(config_path)
    monkeypatch.chdir(working)

    resolved = resolve_pretrade_path(config, config_path)

    assert resolved == deploy / name
    assert resolved.read_text(encoding="utf-8") == "beside live config"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_pretrade_path_must_be_a_file_without_echoing_input(
    tmp_path: Path,
    kind: str,
) -> None:
    marker = f"SYNTHETIC-{kind}-PRETRADE-SECRET"
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    if kind == "directory":
        (deploy / marker).mkdir()
    config = LiveConfig.model_validate(_payload(pretrade_config=marker))

    with pytest.raises(ValueError, match="existing file") as caught:
        resolve_pretrade_path(config, deploy / "live.yaml")

    _assert_safe_error(caught, marker)
