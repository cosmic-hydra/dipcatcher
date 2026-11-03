"""Live-gateway configuration. The venue in this file is the only venue that can arm."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from quant_fund.pretrade.config import PretradeConfig, SessionConfig, load_pretrade_config

Venue = Literal["loopback", "alpaca-paper", "alpaca-live"]
CalendarName = Literal["us_rth", "continuous"]


class LiveConfig(BaseModel):
    """Schema version 1. Extra keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    venue: Venue
    starting_cash: float = Field(gt=0.0, allow_inf_nan=False)
    calendar: CalendarName = "us_rth"
    pretrade_config: str
    state_dir: str = Field(min_length=1)
    cash_tolerance: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)
    position_tolerance: float = Field(default=1e-6, ge=0.0, allow_inf_nan=False)

    @field_validator("state_dir")
    @classmethod
    def _state_dir_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state_dir must not be blank")
        return value


def _safe_validation_error(exc: ValidationError) -> ValueError:
    details: list[str] = []
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    for error in errors[:8]:
        location = error["loc"]
        if error["type"] == "extra_forbidden":
            detail = "extra keys are not permitted"
        elif (
            len(location) == 1
            and isinstance(location[0], str)
            and location[0] in LiveConfig.model_fields
        ):
            detail = f"{location[0]}: {error['msg']}"
        else:
            continue
        if detail not in details:
            details.append(detail)
    message = "live config validation failed"
    if details:
        message = f"{message}: {'; '.join(details)}"
    return ValueError(message[:512])


def load_live_config(path: str | Path) -> LiveConfig:
    """Load YAML without reflecting configuration content in errors."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ValueError("live config contains invalid YAML") from None
    if not isinstance(raw, dict):
        raise ValueError("live config must be a YAML mapping")
    try:
        return LiveConfig.model_validate(raw)
    except ValidationError as exc:
        raise _safe_validation_error(exc) from None


def resolve_pretrade_path(config: LiveConfig, config_path: str | Path) -> Path:
    """Resolve and validate the pre-trade file relative to the live config file."""
    raw = Path(config.pretrade_config)
    path = raw if raw.is_absolute() else Path(config_path).resolve().parent / raw
    if not path.is_file():
        raise ValueError("pretrade config path must be an existing file")
    return path


def load_gateway_pretrade(
    config: LiveConfig,
    config_path: str | Path,
    *,
    hmac_key: bytes,
) -> tuple[PretradeConfig, str, str]:
    """Load pre-trade limits and apply the gateway calendar.

    ``us_rth`` keeps the pre-trade file's session. ``continuous`` is a UTC
    day that stays open on weekends, for venues the operator marked that way.
    The returned digest is the signed payload the arm file must match.
    """
    path = resolve_pretrade_path(config, config_path)
    loaded, _, _ = load_pretrade_config(path, hmac_key=hmac_key)
    if config.calendar == "continuous":
        loaded = loaded.model_copy(
            update={
                "session": SessionConfig(
                    timezone="UTC",
                    open_minute=0,
                    close_minute=24 * 60,
                    weekends_open=True,
                )
            }
        )
    from quant_fund.pretrade.config import sign_config

    digest, signature = sign_config(loaded, hmac_key)
    return loaded, digest, signature
