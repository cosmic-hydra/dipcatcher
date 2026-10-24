"""Fail-closed edges for book risk overlay and causal gate helpers.

Real assertions on validation, ruin, remaining-drawdown halt, and ES/Kelly
branches — not coverage padding. SYNTHETIC returns only.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.gates import (
    GateSpec,
    apply_gate_stack,
    crash_leverage,
    crc_leverage,
    dd_remaining_leverage,
    es_leverage,
    kelly_leverage,
    vol_target,
    vol_target_leverage,
)
from quant_fund.risk.overlay import BookRiskOverlay


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"vol_target": 0.0}, "vol_target"),
        ({"dd_limit": 1.5}, "dd_limit"),
        ({"es_limit": -0.1}, "es_limit"),
        ({"kelly_fraction": float("nan")}, "kelly_fraction"),
        ({"crc_alpha": 1.0}, "crc_alpha"),
    ],
)
def test_overlay_constructor_rejects_bad_knobs(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BookRiskOverlay(**kwargs)


def test_overlay_ruin_and_drawdown_halt_are_sticky() -> None:
    overlay = BookRiskOverlay(
        lookback=8,
        kelly_lookback=8,
        crc_lookback=8,
        kelly_fraction=0.0,
        crc_alpha=0.0,
    )
    overlay.observe(1_000_000.0)
    assert overlay.preview_scale() > 0.0
    overlay.observe(float("nan"))
    assert overlay.ruined is True
    assert overlay.preview_scale() == 0.0
    snap = overlay.snapshot()
    assert snap["ruined"] is True
    assert snap["research_only"] is True
    assert snap["execution_claim"] == "paper_backtest"

    healthy = BookRiskOverlay(
        lookback=8,
        kelly_lookback=100,
        crc_lookback=100,
        kelly_fraction=0.0,
        crc_alpha=0.0,
        dd_limit=0.05,
    )
    healthy.observe(100.0)
    healthy.observe(100.0)
    for nav in np.linspace(100.0, 94.0, 10):
        healthy.observe(float(nav))
        healthy.preview_scale()
    # Peak-to-trough near the 5% budget with cushion → halt.
    assert healthy.preview_scale() == 0.0


def test_overlay_kelly_clips_negative_mean_and_crc_scales() -> None:
    overlay = BookRiskOverlay(
        lookback=8,
        kelly_lookback=8,
        crc_lookback=8,
        kelly_fraction=0.25,
        crc_alpha=0.05,
        vol_target=0.25,
        es_limit=1.0,
    )
    overlay.observe(100.0)
    for _ in range(12):
        overlay.observe(overlay.navs[-1] * 0.999)
        scale = overlay.preview_scale()
    assert scale == 0.0  # negative μ → Kelly clips to cash

    # Disable Kelly/CRC so a mild up-path is sized by vol/ES only.
    winner = BookRiskOverlay(
        lookback=8,
        kelly_lookback=1000,
        crc_lookback=1000,
        kelly_fraction=0.0,
        crc_alpha=0.0,
        vol_target=0.50,
        es_limit=1.0,
    )
    winner.observe(100.0)
    for _ in range(20):
        winner.observe(winner.navs[-1] * 1.002)
        scale = winner.preview_scale()
    assert 0.0 < scale <= 1.0
    assert winner.snapshot()["n_observe"] == 21


def test_gate_helpers_fail_closed_and_remaining_drawdown() -> None:
    with pytest.raises(ValueError, match="vol target"):
        vol_target(np.zeros(10), target=0.0)
    with pytest.raises(ValueError, match="kelly fraction"):
        kelly_leverage(np.zeros(20), fraction=-1.0)

    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=40)
    lev = vol_target_leverage(r, target=0.025, lookback=8)
    assert lev.shape == r.shape
    assert np.all(lev >= 0.0)

    crash = np.concatenate([np.full(5, 0.0), np.full(5, -0.05)])
    remaining = dd_remaining_leverage(crash, limit=0.05, cushion=0.005)
    assert float(remaining[-1]) == 0.0

    flat = np.zeros(30)
    es = es_leverage(flat, es_limit=0.006, lookback=8)
    assert np.all(es[7:] == 1.0)  # zero ES leaves size alone

    noisy = rng.normal(-0.002, 0.02, size=40)
    crc = crc_leverage(noisy, alpha=0.05, lookback=10, step=3)
    assert np.all((crc >= 0.0) & (crc <= 1.0))

    ruined = np.full(15, -0.5)
    crash_lev = crash_leverage(ruined, lookback=3, crash_return=-0.2)
    assert float(np.min(crash_lev[3:])) == 0.0


def test_apply_gate_stack_remaining_dd_mode_and_stepm_off() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.0002, 0.01, size=100)
    result = apply_gate_stack(
        r,
        GateSpec(
            vol_target=0.05,
            dd_limit=0.05,
            dd_mode="remaining",
            es_limit=0.05,
            kelly_fraction=0.1,
            kelly_lookback=20,
            crc_alpha=0.1,
            crc_lookback=20,
            crash_lookback=5,
            stepm_enable=False,
        ),
    )
    assert result.returns.shape == r.shape
    snap = result.snapshot()
    assert snap["research_only"] is True
    assert snap["live_pnl_claim"] is False
    assert "dd=remaining-budget" in result.notes
