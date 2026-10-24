"""KL-UCB bandit canon (Garivier & Cappé 2011): the upper (SYNTHETIC)
confidence index is the Bernoulli-KL solution
q = sup{q >= mu : kl(mu, q) <= ln t / n}, found by bisection.
Bench compares KL-UCB regret against plain UCB1 on the same
synthetic Bernoulli instance.
"""

from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def kl_bern(p: float, q: float) -> float:
    """KL divergence between Bernoulli(p) and Bernoulli(q)."""
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    q = min(max(q, 1e-12), 1.0 - 1e-12)
    return float(p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q)))


def klucb_index(mu_hat: float, n: float, t: float, c: float = 0.0) -> float:
    """Upper KL confidence bound via bisection on [mu_hat, 1)."""
    bound = (np.log(t) + c * np.log(max(np.log(t), 1.0))) / max(n, 1.0)
    lo, hi = float(mu_hat), 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kl_bern(mu_hat, mid) <= bound:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def kl_ucb(probs: FloatArray, steps: int, rng: np.random.Generator) -> float:
    """KL-UCB policy; returns pseudo-regret."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    counts = np.zeros(k)
    rewards = np.zeros(k)
    regret = 0.0
    best = float(probs.max())
    for t in range(steps):
        if t < k:
            a = t
        else:
            idx = [klucb_index(rewards[i] / counts[i], counts[i], float(t)) for i in range(k)]
            a = int(np.argmax(idx))
        r = float(rng.random() < probs[a])
        counts[a] += 1.0
        rewards[a] += r
        regret += best - probs[a]
    return float(regret)


def _ucb1_regret(probs: FloatArray, steps: int, rng: np.random.Generator) -> float:
    """UCB1 baseline (duplicated to keep this module self-contained)."""
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.size
    counts = np.zeros(k)
    means = np.zeros(k)
    regret = 0.0
    best = float(probs.max())
    for t in range(steps):
        if t < k:
            a = t
        else:
            bonus = np.sqrt(2.0 * np.log(t) / np.maximum(counts, 1.0))
            a = int(np.argmax(means + bonus))
        r = float(rng.random() < probs[a])
        counts[a] += 1.0
        means[a] += (r - means[a]) / counts[a]
        regret += best - probs[a]
    return float(regret)


def bench_kl_bandits(
    seed: int | None = None,
    *,
    steps: int = 2000,
    runs: int = 8,
) -> dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else 20261231)
    k = 10
    probs = np.array([0.72] + list(rng.uniform(0.2, 0.55, k - 1)))
    kl = float(np.mean([kl_ucb(probs, steps, rng) for _ in range(runs)]))
    u = float(np.mean([_ucb1_regret(probs, steps, rng) for _ in range(runs)]))
    return {
        "synthetic_klucb_regret": kl,
        "synthetic_ucb1_regret": u,
        "synthetic_kl_gain": u - kl,
    }
