"""Tests for research/benches_w17.py — SOTA canon wave 17 scorecard families.

Each bench is executed once per module run (module-scoped fixtures): the values
are deterministic (seeded from module constants; the torch trainers are
single-threaded CPU and fully seeded), so re-running a bench per test buys
nothing. The determinism tests at the bottom re-run each bench once and compare
against its fixture bit-for-bit. The ``greek_neutral`` and
``diffusion_forecaster`` families are TORCH-GATED — their science and
determinism legs skip when the optional ``nn`` extra is absent, in which case
the benches legitimately return ``{}`` (the wave-12 ``deep_hedging`` / wave-16
``rl_market_maker`` precedent).

Tolerance policy: the wave-17 benches run SHRUNK Monte-Carlo / sampling budgets
relative to the lane suites in tests/unit/core and tests/unit/models (OCE
trials = 100 instead of 200; e-PS n_seeds = 6 / budget = 2500 instead of 8 /
3000; DiffPTS n_samples = 80 instead of 200 and torf_epochs = 20 instead of
200; the greek-neutral sweep runs the lane's tiny seeded config AS-IS). The
science assertions below are therefore DIRECTIONAL with wider documented slack
than the lane tests — they pin the qualitative claims of the cited papers (the
Eq. 20 high-probability OCE guarantee with a non-vacuous uncertified contrast,
the sqrt(n'/n) Hoeffding-radius law and the margin's over-certification
ablation; e-PS discovering all planted nonnulls with fewer samples than
round-robin and fixed-design e-BH at controlled FDP and full TPR; monotone
delta-exposure reduction with an interior optimum and bias -> neutrality tilt
movement; the DiffPTS full-ELBO CRPS win over the NGBoost Gaussian baseline at
near-nominal coverage and calibrated PIT) without asserting the lane suites'
tight windows.

Documented deviations (mirroring the bench module docstring):
- ``eps_speedup_vs_*`` asserts are > 1.0 (the lane suite asserts > 1.1 at the
  full n_seeds = 8 / budget = 3000); documented wider slack for the shrunk
  seeds/budget.
- ``gnp_net_delta_exposure_best`` is read from the sweep's net-exposure array at
  ``best_index`` (the module's metrics dict exposes only the baseline); the
  Sharpe-like objective stays under ``sim_internal_*`` keys OUT of the blob, so
  the interior-optimum science is asserted through the module's float
  ``gnp_interior_optimum`` / ``gnp_best_alpha`` flags plus exposure directions.
- DiffPTS keeps the lane reference design (train 700 / test 400 / epochs 200 /
  NGBoost 60 rounds); n_test = 400 is part of the fixture (on the n_test = 300
  subset the seeded CRPS gain flips sign), and the PIT KS p-value assert is
  weak (valid p-value, > 0.01) since the shrunk n_samples adds MC noise.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from quant_fund.research.benches_w17 import (
    bench_adaptive_eps,
    bench_conformal_oce,
    bench_diffusion_forecaster,
    bench_extra_conformal,
    bench_greek_neutral,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)


def _torch_present() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


_HAS_TORCH = _torch_present()
requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="greek-neutral / DiffPTS training requires the nn extra (torch)"
)

_FAMILIES = (
    "conformal_oce",
    "adaptive_eps",
    "greek_neutral",
    "diffusion_forecaster",
)
# The two numpy/scipy families (greek_neutral / diffusion_forecaster are
# torch-gated, handled apart).
_NUMPY_BLOBS = (
    "conformal_oce",
    "adaptive_eps",
)


@pytest.fixture(scope="module")
def conformal_oce() -> dict[str, float]:
    return bench_conformal_oce()


@pytest.fixture(scope="module")
def adaptive_eps() -> dict[str, float]:
    return bench_adaptive_eps()


@pytest.fixture(scope="module")
def greek_neutral() -> dict[str, float]:
    return bench_greek_neutral()


@pytest.fixture(scope="module")
def diffusion_forecaster() -> dict[str, float]:
    return bench_diffusion_forecaster()


def test_families_registered_as_optional() -> None:
    for fam in _FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


def test_w16_families_registered_as_optional() -> None:
    # The wave-16 registry block landed on main with the w16 benches; these
    # assertions pin it here since the w17 test file owns the w16 wiring check.
    for fam in ("vol_loss_decomposition", "hierarchical_conformal", "multisource_conformal"):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


@pytest.mark.parametrize("blob_name", list(_NUMPY_BLOBS))
def test_numpy_blobs_are_clean_and_finite(blob_name: str, request: pytest.FixtureRequest) -> None:
    blob = request.getfixturevalue(blob_name)
    assert isinstance(blob, dict) and blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())


def test_greek_neutral_blob_is_clean(greek_neutral: dict[str, float]) -> None:
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(greek_neutral)
    if not _HAS_TORCH:
        assert greek_neutral == {}
        return
    assert isinstance(greek_neutral, dict) and greek_neutral
    assert family_blob_has_finite_observation(greek_neutral)
    assert all(np.isfinite(v) for v in greek_neutral.values())
    # The Sharpe-like objective never leaks into the blob (honesty contract):
    # sim_internal_* keys stay out; only gnp_* exposure / flag keys are emitted.
    assert all(key.startswith("gnp_") for key in greek_neutral)


def test_diffusion_forecaster_blob_is_clean(diffusion_forecaster: dict[str, float]) -> None:
    # Forbidden-key scan holds whether or not torch is present.
    assert family_blob_forbidden_metrics_absent(diffusion_forecaster)
    if not _HAS_TORCH:
        assert diffusion_forecaster == {}
        return
    assert isinstance(diffusion_forecaster, dict) and diffusion_forecaster
    assert family_blob_has_finite_observation(diffusion_forecaster)
    assert all(np.isfinite(v) for v in diffusion_forecaster.values())


def test_conformal_oce_guarantee_and_margin_ablation(conformal_oce: dict[str, float]) -> None:
    assert conformal_oce["oce_delta"] == pytest.approx(0.1)
    assert conformal_oce["oce_alpha"] == pytest.approx(0.3)
    # Farzaneh & Simeone 2026 Eq. 20: the CERTIFIED policy's exact population
    # CVaR exceeds epsilon in at most delta of trials (SHRUNK trials = 100;
    # documented budget delta + 0.02).
    assert conformal_oce["oce_violation_rate_certified"] <= conformal_oce["oce_delta"] + 0.02
    # The uncertified risk-neutral model-greedy baseline violates visibly more
    # — the guarantee is not vacuous (cf. the paper's VaR baseline at 53.4%).
    assert (
        conformal_oce["oce_baseline_violation_rate"] > conformal_oce["oce_violation_rate_certified"]
    )
    # Certificates are issued at a healthy rate (fail-closed otherwise).
    assert 0.0 < conformal_oce["oce_cert_rate"] <= 1.0
    # The Hoeffding radius follows the sqrt(n'/n) concentration law:
    # sqrt(3000 / 750) = 2.0 (trials-independent; ~0.1 documented slack).
    assert conformal_oce["oce_radius_ratio"] == pytest.approx(2.0, abs=0.1)
    # Dropping the Hoeffding margin OVER-certifies (the plug-in UCB is looser):
    # the margin's role in the guarantee.
    assert conformal_oce["oce_plugin_cert_rate"] > conformal_oce["oce_cert_rate"]


def test_adaptive_eps_sample_efficiency_and_fdr(adaptive_eps: dict[str, float]) -> None:
    assert adaptive_eps["eps_alpha"] == pytest.approx(0.1)
    # Lin, Ma, Ren & Wei 2026: on the seeded planted world every seed reaches
    # full discovery inside the shrunk budget (no censoring)...
    assert adaptive_eps["eps_censored"] == 0.0
    assert adaptive_eps["eps_discovery_samples_mean"] > 0.0
    # ...with FEWER total samples than uniform round-robin + e-BH and than the
    # fixed-design e-BH baseline (paper Sections 1 and 6; budget > 1.0 — the
    # lane asserts > 1.1 at the full budget, documented wider slack here).
    assert adaptive_eps["eps_speedup_vs_round_robin"] > 1.0
    assert adaptive_eps["eps_speedup_vs_fixed_design"] > 1.0
    # FDR controlled at the data-dependent discovery stop (FDP <= alpha/2)...
    assert adaptive_eps["eps_fdp_at_discovery_mean"] <= 0.05
    # ...with full power: every planted nonnull is rejected at discovery.
    assert adaptive_eps["eps_tpr_mean"] == 1.0


@requires_torch
def test_greek_neutral_exposure_fall_and_interior_optimum(greek_neutral: dict[str, float]) -> None:
    # Tan, Roberts & Zohren 2026 §6.3 (DP-L1, tiny seeded config): realized
    # gross delta exposure FALLS from the unregularized baseline to the best
    # alpha (monotone exposure fall; their eqs. 14-15 diagnostics).
    assert greek_neutral["gnp_gross_delta_exposure_reduction"] > 0.0
    assert greek_neutral["gnp_gross_delta_exposure_baseline"] > 0.0
    # The OOS risk-adjusted objective has an INTERIOR optimum: the best alpha
    # is strictly inside the (0, 25) grid — calibrated regularization holds or
    # improves, extreme alpha degrades (the module computes the flag from the
    # sim_internal objective, which stays out of the blob).
    assert greek_neutral["gnp_interior_optimum"] == 1.0
    assert 0.0 < greek_neutral["gnp_best_alpha"] < 25.0
    # Bias -> neutrality direction: the baseline's persistent net directional
    # tilt moves TOWARD zero at the best alpha.
    assert abs(greek_neutral["gnp_net_delta_exposure_best"]) < abs(
        greek_neutral["gnp_net_delta_exposure_baseline"]
    )


@requires_torch
def test_diffusion_forecaster_crps_gain_and_calibration(
    diffusion_forecaster: dict[str, float],
) -> None:
    # Ye et al. 2026 (§4.2 / Table 2): the full-ELBO diffusion beats the
    # Gaussian density baseline on the CRPS proper score over the shared
    # seeded SYNTHETIC heteroskedastic stream.
    assert diffusion_forecaster["crps_gain_vs_ngboost"] > 0.0
    assert diffusion_forecaster["diffpts_crps"] > 0.0
    assert diffusion_forecaster["ngboost_crps"] > 0.0
    assert diffusion_forecaster["crps_gain_vs_ngboost"] == pytest.approx(
        diffusion_forecaster["ngboost_crps"] - diffusion_forecaster["diffpts_crps"], abs=1e-12
    )
    # The 90% central interval stays near nominal (lane window, documented).
    assert 0.75 <= diffusion_forecaster["diffpts_coverage_90"] <= 0.96
    # PIT uniformity is not strongly rejected by the KS test (weak documented
    # bound — the shrunk n_samples = 80 adds MC noise to the PIT estimate).
    assert 0.0 <= diffusion_forecaster["diffpts_pit_ks_pvalue"] <= 1.0
    assert diffusion_forecaster["diffpts_pit_ks_pvalue"] > 0.01


def test_extra_conformal_scorecard_blob_is_finite() -> None:
    """The model stamps ``synthetic_*`` keys; the scorecard blob must keep them.

    An empty blob is what ``verify-research`` reports as
    ``scorecard_invalid:extra_conformal``.
    """
    blob = bench_extra_conformal()
    assert blob
    assert family_blob_has_finite_observation(blob)
    assert family_blob_forbidden_metrics_absent(blob)
    assert all(np.isfinite(v) for v in blob.values())
    assert blob["xc_harm_coverage_gap"] == (
        blob["xc_harm_unweighted_coverage"] - blob["xc_harm_coverage"]
    )


def test_numpy_benches_are_deterministic(
    conformal_oce: dict[str, float],
    adaptive_eps: dict[str, float],
) -> None:
    # Seeded from module constants, so a fresh call must reproduce each fixture
    # bit-for-bit.
    assert bench_conformal_oce() == conformal_oce
    assert bench_adaptive_eps() == adaptive_eps


@requires_torch
def test_greek_neutral_is_deterministic(greek_neutral: dict[str, float]) -> None:
    # The full-batch Adam trainer is single-threaded and fully seeded (torch +
    # numpy), so a fresh sweep reproduces the fixture bit-for-bit.
    assert bench_greek_neutral() == greek_neutral


@requires_torch
def test_diffusion_forecaster_is_deterministic(diffusion_forecaster: dict[str, float]) -> None:
    # The DiffPTS / NGBoost / TORF fits and every MC sampling pass are seeded,
    # so a fresh shrunk run reproduces the fixture bit-for-bit.
    assert bench_diffusion_forecaster() == diffusion_forecaster
