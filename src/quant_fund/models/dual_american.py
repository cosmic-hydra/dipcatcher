"""Andersen-Broadie dual upper bound for the American put (SYNTHETIC).

Builds a martingale from the tree-implied value function: at each outer path
step the conditional expectation E[V_t | F_{t-1}] is estimated by one-step
inner draws evaluated on the tree value grid, so the increments are
approximately martingale differences and max_t(f_t - M_t) prices the upper
bound without nested full-policy simulation.
"""

import numpy as np

from quant_fund.models._amopt_synth import S0, SIG, K, Q, R, T, crr_price, european_put


def _tree_grid(n: int) -> tuple[np.ndarray, float, float, float]:
    dt = T / n
    u = np.exp(SIG * np.sqrt(dt))
    d = 1.0 / u
    p = float((np.exp((R - Q) * dt) - d) / (u - d))
    disc = np.exp(-R * dt)
    grid = np.zeros((n + 1, n + 1))
    j = np.arange(n + 1)
    S = S0 * u**j * d ** (n - j)
    grid[n, : n + 1] = np.maximum(K - S, 0.0)
    for t in range(n - 1, -1, -1):
        j = np.arange(t + 1)
        S = S0 * u**j * d ** (t - j)
        cont = disc * (p * grid[t + 1, 1 : t + 2] + (1 - p) * grid[t + 1, : t + 1])
        grid[t, : t + 1] = np.maximum(K - S, cont)
    return grid, float(np.log(u)), dt, p


def _vhat(grid: np.ndarray, t: int, s: float, logu: float) -> float:
    d = np.exp(-logu)
    jpos = np.log(s / (S0 * d**t)) / (2 * logu) if t > 0 else 0.0
    j0 = int(np.clip(np.floor(jpos), 0, max(t - 1, 0)))
    j1 = min(j0 + 1, t)
    w = float(np.clip(jpos - j0, 0.0, 1.0))
    return float((1 - w) * grid[t, j0] + w * grid[t, j1])


def _dual_bound(seed: int, n_tree: int = 200, n_path: int = 400) -> tuple[float, float]:
    from numpy.polynomial.hermite_e import hermegauss

    gh_x, gh_w = hermegauss(24)
    gh_w = gh_w / np.sqrt(2.0 * np.pi)
    grid, logu, dt, _p = _tree_grid(n_tree)
    rng = np.random.default_rng(seed)
    drift = (R - Q - 0.5 * SIG**2) * dt
    sd = SIG * np.sqrt(dt)
    duals = np.zeros(n_path)
    for k in range(n_path):
        m = 0.0
        prev_s = S0
        f_max = np.maximum(K - S0, 0.0)
        log_s = np.log(S0)
        for t in range(1, n_tree + 1):
            z = rng.normal()
            log_s += drift + sd * z
            s = float(np.exp(log_s))
            v_t = _vhat(grid, t, s, logu)
            kids = prev_s * np.exp(drift + sd * gh_x)
            ev = float(np.sum(gh_w * np.array([_vhat(grid, t, kk, logu) for kk in kids])))
            m += np.exp(-R * t * dt) * (v_t - ev)
            dual = np.maximum(K - s, 0.0) * np.exp(-R * t * dt) - m
            f_max = max(f_max, dual)
            prev_s = s
        duals[k] = f_max
    return float(duals.mean()), float(duals.std() / np.sqrt(n_path))


def bench_dual_american(
    seed: int = 4109,
    *,
    n_crr: int = 2000,
    n_tree: int = 200,
    n_path: int = 400,
) -> dict[str, float]:
    ref, _ = crr_price(n_crr)
    dual, se = _dual_bound(seed, n_tree=n_tree, n_path=n_path)
    eur = european_put()
    return {
        "synthetic_dual_bound": dual,
        "synthetic_dual_err": abs(dual - ref),
        "synthetic_dual_gap": dual - ref,
        "synthetic_dual_se": se,
        "synthetic_dual_eur_floor_err": abs(eur - ref),
    }
