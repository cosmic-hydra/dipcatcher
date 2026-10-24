"""Tests for research/benches_w20.py — SOTA canon wave 20 scorecard families.

Module-scoped fixtures (values are seeded deterministic); the determinism
test re-runs each numpy bench once and compares bit-for-bit.

``arl_mm`` (torch-gated) may return ``{}`` without the ``nn`` extra — the
clean-blob contract holds on whatever it emits (wave-12/16 precedent).
Science assertions are
DIRECTIONAL at the shrunk fixtures; the lane suites in
tests/unit/{execution,models,metrics} pin the tight windows.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w20 import (
    bench_arl_mm,
    bench_gaussian_normalized_coords,
    bench_hidden_markov_equilibrium,
    bench_liquidity_tail_lob,
    bench_varswap_stopping,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


_HAS_TORCH = _torch_present()

_FAMILIES_LANDED = (
    "varswap_stopping",
    "hidden_markov_equilibrium",
    "gaussian_normalized_coords",
    "liquidity_tail_lob",
)
_FAMILIES_IN_FLIGHT: tuple[str, ...] = ()
_LANDED_NUMPY_BLOBS = _FAMILIES_LANDED
# May return {} without the torch extra (arl_mm is torch-gated).
_SOFT_BLOBS = ("arl_mm",)


@pytest.fixture(scope="module")
def varswap_stopping() -> dict[str, float]:
    return bench_varswap_stopping()


@pytest.fixture(scope="module")
def hidden_markov_equilibrium() -> dict[str, float]:
    return bench_hidden_markov_equilibrium()


@pytest.fixture(scope="module")
def gaussian_normalized_coords() -> dict[str, float]:
    return bench_gaussian_normalized_coords()


@pytest.fixture(scope="module")
def arl_mm() -> dict[str, float]:
    return bench_arl_mm()


@pytest.fixture(scope="module")
def liquidity_tail_lob() -> dict[str, float]:
    return bench_liquidity_tail_lob()


def test_landed_families_registered_as_optional() -> None:
    for fam in _FAMILIES_LANDED:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_LANDED_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


@pytest.mark.parametrize("blob_name", list(_SOFT_BLOBS))
def test_soft_blobs_are_clean(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert family_blob_forbidden_metrics_absent(blob)
    if not blob:
        return
    assert family_blob_has_finite_observation(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_varswap_stopping_threshold_structure(
    varswap_stopping: dict[str, float],
) -> None:
    # Lorig-Lozano-Gomez: the eigenfunction boundary value solves its
    # equation exactly, the smooth-pasting system is exact, the MC tail
    # mass at the computed trigger matches the boundary condition, and
    # the short window sits strictly inside (0, d).
    blob = varswap_stopping
    assert blob["synthetic_eigen_eq_max_resid"] < 1e-8
    assert blob["synthetic_smooth_pasting_max_resid"] < 1e-8
    assert blob["synthetic_mc_entry_tail_abs_err"] < 0.01
    assert 0.0 < blob["synthetic_exit_threshold_short"] < blob["synthetic_entry_threshold_short"]
    assert 0.0 < blob["synthetic_window_lower_edge"] < blob["synthetic_window_upper_edge"]
    assert blob["synthetic_forcing_sign_flip_geometry"] == 1.0


def test_hidden_markov_equilibrium_calibration(
    hidden_markov_equilibrium: dict[str, float],
) -> None:
    # §5.5 calibration: the pricing ODE and boundary residuals are at
    # machine/band tolerance, the vol curve peaks at the printed p*,
    # the affine risk-free range matches the paper's band, and the
    # closed-form vs MC pricing gaps are within MC error.
    blob = hidden_markov_equilibrium
    # The lane bench prefixes every diagnostic with synthetic_.
    assert blob["synthetic_ode_residual_max"] < 1e-6
    assert blob["synthetic_bc_residual_max"] < 1e-6
    assert blob["synthetic_phi_min"] > 0.0 and blob["synthetic_phi_min"] < blob["synthetic_phi_max"]
    assert 0.0 < blob["synthetic_p_star"] < 1.0
    assert blob["synthetic_sigmaS_max"] > blob["synthetic_sigmaS_endpoint"]
    assert 0.0 < blob["synthetic_rf_min"] < blob["synthetic_rf_max"] < 0.25
    assert blob["synthetic_skew_rel_err"] < 0.10
    assert blob["synthetic_option_rel_err"] < 0.10
    assert 0.0 <= blob["synthetic_filter_state_corr"] <= 1.0
    assert 0.35 < blob["synthetic_pit_mean"] < 0.65


def test_gaussian_normalized_coords_arb_cycle(
    gaussian_normalized_coords: dict[str, float],
) -> None:
    # Sun 2026: a clean convex smile passes the m>=0/Mills checks, the
    # injected concavity is flagged in both Bachelier and Black
    # coordinates, the q-inequality detects the bump with high precision,
    # the isotonic repair restores admissibility, and the h->eta->h
    # round trip is exact.
    blob = gaussian_normalized_coords
    assert blob["synthetic_clean_arb_free"] == 1.0
    assert blob["synthetic_viol_flagged"] == 1.0
    assert blob["synthetic_m_min_clean"] > 0.0
    assert blob["synthetic_m_min_viol"] < 0.0
    assert blob["synthetic_viol_m_neg_count"] > 0.0
    assert blob["synthetic_q_precision"] >= 0.8
    assert blob["synthetic_q_recall"] >= 0.5
    assert blob["synthetic_repair_m_min"] > -0.02
    assert blob["synthetic_black_clean_free"] == 1.0
    assert blob["synthetic_black_viol_flagged"] == 1.0
    assert blob["synthetic_roundtrip_eta_max_err"] < 1e-10


def test_liquidity_tail_lob_tail_shift(
    liquidity_tail_lob: dict[str, float],
) -> None:
    # Cetin-Lin-Livieri: under a t(3) demand prior the crossover to
    # informed dominance sits deeper than Gaussian, deep-book impact is
    # flatter, the fixed point converged, and the empirical tail index
    # tracks the paper's rho = -2/3 law.
    blob = liquidity_tail_lob
    assert blob["synthetic_crossover_ratio_t_over_gauss"] > 1.0
    assert blob["synthetic_h_deep_t"] < blob["synthetic_h_deep_gauss"]
    assert blob["synthetic_informed_share_deep_t"] < blob["synthetic_informed_share_deep_gauss"]
    assert blob["synthetic_fixedpoint_max_resid_t"] < 1e-6
    assert blob["synthetic_fixedpoint_max_resid_gauss"] < 1e-6
    assert abs(blob["synthetic_tail_rho_hat_t"] - blob["synthetic_tail_rho_theory_t1"]) < 0.25
    assert blob["synthetic_posterior_consistency_error"] < 1e-6


@pytest.mark.skipif(not _HAS_TORCH, reason="torch extra absent")
def test_arl_mm_left_tail_improvement(arl_mm: dict[str, float]) -> None:
    # Yang & Xu 2026: the ARL maker improves the left tail vs the
    # non-adversarial baseline on the pooled adversarial grid, and the
    # Hawkes flow fixture stays clustered (Fano > Poisson control).
    blob = arl_mm
    if not blob:
        pytest.skip("arl_mm bench emitted {} (torch unavailable at runtime)")
    assert blob["synthetic_arl_left_tail_q05_improvement"] > 0.0
    assert blob["synthetic_arl_left_tail_env_improvement_frac"] > 0.5
    assert blob["synthetic_hawkes_fano_factor"] > blob["synthetic_hawkes_fano_factor_poisson"]


def test_numpy_benches_are_deterministic(
    varswap_stopping: dict[str, float],
    hidden_markov_equilibrium: dict[str, float],
    gaussian_normalized_coords: dict[str, float],
    liquidity_tail_lob: dict[str, float],
) -> None:
    assert bench_varswap_stopping() == varswap_stopping
    assert bench_hidden_markov_equilibrium() == hidden_markov_equilibrium
    assert bench_gaussian_normalized_coords() == gaussian_normalized_coords
    assert bench_liquidity_tail_lob() == liquidity_tail_lob
