"""Tabular TD-learning canon: TD(0) policy evaluation on sampled (SYNTHETIC)
transitions, SARSA on-policy control, and Q-learning off-policy control —
all with epsilon-greedy exploration and decaying step sizes.
``bench_td_learning`` uses a chain MDP with a known optimal policy and
gates learned Q/policy agreement against the exact DP solution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def td0_eval(
    p: FloatArray,
    r: FloatArray,
    pi: FloatArray,
    gamma: float = 0.99,
    episodes: int = 2000,
    alpha0: float = 0.2,
    seed: int = 0,
) -> FloatArray:
    """Sample-based TD(0) estimate of V^pi."""
    p = np.asarray(p, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_s, n_a = r.shape
    v = np.zeros(n_s)
    counts = np.zeros(n_s)
    for _ in range(episodes):
        s = int(rng.integers(n_s))
        for _ in range(40):
            a = int(rng.choice(n_a, p=pi[s]))
            s2 = int(rng.choice(n_s, p=p[s, a]))
            counts[s] += 1.0
            alpha = alpha0 / np.sqrt(counts[s])
            v[s] += alpha * (r[s, a] + gamma * v[s2] - v[s])
            s = s2
    return v


def _eps_greedy(q_row: FloatArray, eps: float, rng: np.random.Generator) -> int:
    if rng.uniform() < eps:
        return int(rng.integers(q_row.size))
    return int(np.argmax(q_row))


def _run(
    p: FloatArray,
    r: FloatArray,
    gamma: float,
    episodes: int,
    alpha0: float,
    eps0: float,
    seed: int,
    on_policy: bool,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    n_s, n_a = r.shape
    q = np.zeros((n_s, n_a))
    counts = np.zeros((n_s, n_a))
    for ep in range(episodes):
        s = int(rng.integers(n_s))
        a = _eps_greedy(q[s], eps0, rng)
        for _ in range(60):
            s2 = int(rng.choice(n_s, p=p[s, a]))
            eps = eps0 / np.sqrt(1.0 + ep)
            a2 = _eps_greedy(q[s2], eps, rng)
            counts[s, a] += 1.0
            alpha = alpha0 / np.sqrt(counts[s, a])
            target = r[s, a] + gamma * (q[s2, a2] if on_policy else q[s2].max())
            q[s, a] += alpha * (target - q[s, a])
            s, a = s2, a2
    return q


def q_learning(
    p: FloatArray,
    r: FloatArray,
    gamma: float = 0.99,
    episodes: int = 1200,
    alpha0: float = 0.3,
    eps0: float = 0.3,
    seed: int = 0,
) -> FloatArray:
    return _run(
        np.asarray(p, float), np.asarray(r, float), gamma, episodes, alpha0, eps0, seed, False
    )


def sarsa(
    p: FloatArray,
    r: FloatArray,
    gamma: float = 0.99,
    episodes: int = 1200,
    alpha0: float = 0.3,
    eps0: float = 0.3,
    seed: int = 0,
) -> FloatArray:
    return _run(
        np.asarray(p, float), np.asarray(r, float), gamma, episodes, alpha0, eps0, seed, True
    )


def bench_td_learning(seed: int = 20261231, *, episodes: int = 6000) -> dict[str, float]:
    from quant_fund.models.mdp_solvers import policy_eval, value_iteration

    rng = np.random.default_rng(seed)
    n_s, n_a = 10, 2
    p = rng.dirichlet(np.ones(n_s) * 0.5, size=(n_s, n_a))
    r = 0.3 * rng.standard_normal((n_s, n_a))
    # planted: action 1 marches toward goal state with high prob + reward
    p[:, 1, :] = 0.0
    for s in range(n_s - 1):
        p[s, 1, s + 1] = 1.0
    p[n_s - 1, :, :] = 0.0
    p[n_s - 1, :, n_s - 1] = 1.0
    r[:, 1] += 0.8
    r[n_s - 1, :] = 1.5
    gamma = 0.95
    v_star, pi_star = value_iteration(p, r, gamma)
    q_ql = q_learning(p, r, gamma, episodes=episodes, alpha0=0.5, eps0=0.2, seed=seed + 1)
    q_sa = sarsa(p, r, gamma, episodes=episodes, alpha0=0.5, eps0=0.2, seed=seed + 2)
    v_td = td0_eval(p, r, pi_star, gamma, episodes=episodes, alpha0=0.5, seed=seed + 3)
    v_pi = policy_eval(p, r, pi_star, gamma)
    agree_ql = np.mean(q_ql.argmax(1) == pi_star.argmax(1))
    agree_sa = np.mean(q_sa.argmax(1) == pi_star.argmax(1))
    return {
        "synthetic_ql_policy_agree": float(agree_ql),
        "synthetic_sarsa_policy_agree": float(agree_sa),
        "synthetic_td0_eval_err": float(np.max(np.abs(v_td - v_pi))),
        "synthetic_ql_v_err": float(np.max(np.abs(q_ql.max(axis=1) - v_star))),
    }
