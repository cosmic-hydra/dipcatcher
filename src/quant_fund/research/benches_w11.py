"""Benchmark batteries for SOTA canon wave 11 (rough-path signatures,
optimal transport, Wasserstein DRO, conformal PID control).

Seeded SYNTHETIC streams only — no panel or vendor data, no headline
performance ratios. Each bench returns a flat dict of proper diagnostic
statistics, or ``{}`` if its synthetic setup cannot be constructed.
"""

from __future__ import annotations

import numpy as np

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.wasserstein import (
    sinkhorn_divergence,
    wasserstein_1d,
    wasserstein_gaussian,
)
from quant_fund.models.conformal import AdaptiveConformal
from quant_fund.models.conformal_pid import ConformalPID
from quant_fund.models.path_signatures import (
    lead_lag_transform,
    logsignature,
    signature,
    signature_kernel,
)
from quant_fund.portfolio.wasserstein_dro import solve_wasserstein_dro

_SEED = 20260928


def bench_rough_paths() -> dict[str, float]:
    """Rough-path signature diagnostics (wave 11; Chen 1958, Lyons 1998,
    Kidger & Foster 2020): closed-loop level-1 residual must vanish, the
    lead-lag Levy area must match the circle's enclosed area, and the
    signature kernel must separate a path from its time-shuffle."""
    try:
        rng = np.random.default_rng(_SEED)
        order = 3
        # closed circle: level-1 signature = net displacement = 0
        theta = np.linspace(0.0, 2.0 * np.pi, 257)
        radius = 1.5
        circle = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        ll = lead_lag_transform(circle)
        sig = signature(ll, order)
        d = ll.shape[1]  # 2 * underlying dim; columns are (lead_1..lead_d/2, lag_1..lag_d/2)
        half = d // 2
        closure_residual = float(np.linalg.norm(sig[:d]))
        # Levy signed area from the antisymmetric level-2 lead/lag block:
        # level-2 starts after the d level-1 terms; word (i, j) sits at
        # i*d + j inside the block. Area = (1/2)(Sig[lead_1, lag_2] -
        # Sig[lead_2, lag_1]) per the lead_lag_transform docstring.
        w_lead1_lag2 = d + 0 * d + (half + 1)
        w_lead2_lag1 = d + 1 * d + half
        levy_area = 0.5 * float(sig[w_lead1_lag2] - sig[w_lead2_lag1])
        expected_area = float(np.pi * radius**2)
        # kernel separates the path from a time-shuffled copy
        perm = rng.permutation(circle.shape[0])
        shuffled = circle[perm]
        kernel_self = signature_kernel(circle, circle, order)
        kernel_shuffled = signature_kernel(circle, shuffled, order)
        logsig_dim = float(logsignature(circle, order).size)
        return {
            "order": float(order),
            "closure_residual": closure_residual,
            "levy_area": levy_area,
            "levy_area_expected": expected_area,
            "levy_area_abs_err": abs(levy_area - expected_area),
            "kernel_self": kernel_self,
            "kernel_shuffled": kernel_shuffled,
            "kernel_gap": kernel_self - kernel_shuffled,
            "logsig_dim": logsig_dim,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_optimal_transport() -> dict[str, float]:
    """Optimal-transport diagnostics (wave 11; Villani 2003, Cuturi 2013,
    Bhatia-Jain-Lim 2019): W1 of a unit shift must recover 1, the Bures-
    Wasserstein distance must vanish for identical Gaussians and equal the
    squared mean gap for shifted ones, and the Sinkhorn divergence must
    separate same-law from shifted samples."""
    try:
        rng = np.random.default_rng(_SEED)
        # W1 of a unit shift is exact at any n. Sinkhorn is O(n^2) and the
        # separation claim (shifted divergence >> same-law) holds at n=500;
        # n=2000 was ~12s inside every research sweep.
        n = 500
        a = rng.standard_normal(n)
        b = a + 1.0
        c = rng.standard_normal(n)
        w1_shift = wasserstein_1d(a, b, p=1.0)
        w1_same = wasserstein_1d(a, c, p=1.0)
        div_same = sinkhorn_divergence(a.reshape(-1, 1), c.reshape(-1, 1), reg=0.1)
        div_shift = sinkhorn_divergence(a.reshape(-1, 1), b.reshape(-1, 1), reg=0.1)
        mu0 = np.zeros(2)
        eye = np.eye(2)
        w2_identical = wasserstein_gaussian(mu0, eye, mu0.copy(), eye.copy())
        delta = np.array([1.0, -2.0])
        w2_shifted = wasserstein_gaussian(mu0, eye, delta, eye.copy())
        return {
            "n_samples": float(n),
            "w1_shifted": w1_shift,
            "w1_shifted_abs_err": abs(w1_shift - 1.0),
            "w1_same_law": w1_same,
            "sink_div_same_law": div_same,
            "sink_div_shifted": div_shift,
            "w2_gauss_identical": w2_identical,
            "w2_gauss_shifted": w2_shifted,
            "w2_shift_expected": float(np.dot(delta, delta)),
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_dro() -> dict[str, float]:
    """Wasserstein-DRO radius sweep (wave 11; Mohajerin Esfahani & Kuhn
    2018 Thm 4.2): the risky allocation norm must shrink monotonically in
    the ambiguity radius, the worst-case objective must dominate the
    nominal one, and the budget constraint must hold exactly."""
    try:
        rng = np.random.default_rng(_SEED)
        d = 5
        mu = np.abs(rng.standard_normal(d)) + 0.05
        g = rng.standard_normal((d, d))
        cov = g @ g.T / d + np.eye(d) * 0.1
        radii = (0.0, 0.05, 0.2)
        norms: list[float] = []
        wc_ge_nom = 1.0
        budget_err = 0.0
        worst0 = nominal0 = 0.0
        for i, r in enumerate(radii):
            res = solve_wasserstein_dro(mu, cov, radius=r, risk_aversion=1.0)
            norms.append(float(np.linalg.norm(res.weights)))
            budget_err = max(budget_err, abs(float(res.weights.sum()) - 1.0))
            if res.worst_case_objective + 1e-9 < res.nominal_objective:
                wc_ge_nom = 0.0
            if i == 0:
                nominal0 = float(res.nominal_objective)
                worst0 = float(res.worst_case_objective)
        monotone = 1.0 if norms[0] >= norms[1] >= norms[2] else 0.0
        return {
            "dim": float(d),
            "norm_radius_0": norms[0],
            "norm_radius_005": norms[1],
            "norm_radius_02": norms[2],
            "shrinkage_monotone": monotone,
            "worstcase_ge_nominal": wc_ge_nom,
            "radius0_worst_eq_nominal": 1.0 if abs(worst0 - nominal0) < 1e-9 else 0.0,
            "budget_abs_err": budget_err,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}


def bench_conformal_pid() -> dict[str, float]:
    """Conformal PID control vs scalar ACI (wave 11; Angelopoulos, Candes &
    Tibshirani 2023, arXiv:2307.16895): under an abrupt scale shift the PID
    controller's cumulative coverage deviation must beat ACI's, and its
    trailing coverage must sit near nominal. Both loops run the identical
    two-sided |y|-score band on the same seeded stream (mirrors
    tests/unit/models/test_conformal_pid.py)."""
    try:
        rng = np.random.default_rng(_SEED)
        n_steps, switch = 1200, 600
        alpha = 0.10
        nominal = 1.0 - alpha
        sigma = np.where(np.arange(n_steps) < switch, 1.0, 3.0)
        stream = sigma * rng.standard_normal(n_steps)

        # PID: half-width = conformal quantile of past |y| scores at 1 - alpha_t
        pid = ConformalPID(alpha=alpha)
        scores: list[float] = []
        cov_pid: list[float] = []
        for val in stream:
            q = conformal_quantile(np.asarray(scores, dtype=float), pid.alpha_t)
            cov = float(abs(val) <= q)
            cov_pid.append(cov)
            scores.append(abs(float(val)))
            pid.update(1.0 - cov)
        cov_pid_arr = np.asarray(cov_pid, dtype=float)

        # ACI baseline on the same band construction
        aci = AdaptiveConformal(alpha=alpha, gamma=0.05)
        path = aci.run(
            y=stream,
            lower=np.zeros(n_steps),
            upper=np.zeros(n_steps),
            dates=[f"{i:08d}" for i in range(n_steps)],
        )
        cov_aci_arr = np.nan_to_num(path.covered, nan=0.0)

        tail = slice(n_steps - 600, n_steps)
        pid_dev = float(np.abs(cov_pid_arr[tail] - nominal).sum())
        aci_dev = float(np.abs(cov_aci_arr[tail] - nominal).sum())
        trail = slice(n_steps - 300, n_steps)
        return {
            "nominal_coverage": nominal,
            "pid_trailing_coverage": float(cov_pid_arr[trail].mean()),
            "aci_trailing_coverage": float(cov_aci_arr[trail].mean()),
            "pid_abs_coverage_error": abs(float(cov_pid_arr[trail].mean()) - nominal),
            "pid_cum_deviation": pid_dev,
            "aci_cum_deviation": aci_dev,
            "pid_beats_aci": 1.0 if pid_dev < aci_dev else 0.0,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return {}
