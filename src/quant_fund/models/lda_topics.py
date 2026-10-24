"""Latent Dirichlet allocation via collapsed Gibbs sampling (SYNTHETIC)
(Griffiths & Steyvers 2004): z-di token draws with α/β
Dirichlet priors; returns topic-word φ and doc-topic θ.
Synthetic bench gates recovery of two planted topic
profiles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def lda_gibbs(
    docs: list[IntArray],
    n_topics: int,
    vocab: int,
    alpha: float = 0.3,
    beta: float = 0.05,
    it: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Collapsed Gibbs: p(z=k|rest) ∝ (n_dk+α)(n_kw+β)/(n_k+Vβ)."""
    rng = np.random.default_rng(seed)
    n_docs = len(docs)
    flat = [w for d in docs for w in d]
    z = np.asarray(rng.integers(0, n_topics, len(flat)))
    w_arr = np.asarray(flat)
    doc_of = np.asarray([i for i, d in enumerate(docs) for _ in d])
    n_dk = np.zeros((n_docs, n_topics), dtype=np.int64)
    n_kw = np.zeros((n_topics, vocab), dtype=np.int64)
    n_k = np.zeros(n_topics, dtype=np.int64)
    for t in range(len(flat)):
        k = int(z[t])
        n_dk[doc_of[t], k] += 1
        n_kw[k, w_arr[t]] += 1
        n_k[k] += 1
    for _ in range(it):
        for t in range(len(flat)):
            k = int(z[t])
            w = w_arr[t]
            d = doc_of[t]
            n_dk[d, k] -= 1
            n_kw[k, w] -= 1
            n_k[k] -= 1
            p = (n_dk[d] + alpha) * (n_kw[:, w] + beta) / (n_k + vocab * beta)
            k_new = int(rng.choice(n_topics, p=p / p.sum()))
            z[t] = k_new
            n_dk[d, k_new] += 1
            n_kw[k_new, w] += 1
            n_k[k_new] += 1
    phi = (n_kw + beta) / (n_k[:, None] + vocab * beta)
    theta = (n_dk + alpha) / (n_dk.sum(axis=1, keepdims=True) + n_topics * alpha)
    return {"phi": phi, "theta": theta, "z": z, "doc_of": doc_of}


def bench_lda_topics(seed: int = 556, *, n_docs: int = 80, it: int = 250) -> dict[str, float]:
    """SYNTHETIC: docs drawn from two planted topic profiles —
    the learned φ must recover both (up to permutation)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    vocab = 30
    # topic 0 mass on words 0-9, topic 1 on words 10-19
    phi_true = np.zeros((2, vocab)) + 0.01
    phi_true[0, :10] = 0.095
    phi_true[1, 10:20] = 0.095
    phi_true /= phi_true.sum(axis=1, keepdims=True)
    docs: list[IntArray] = []
    for i in range(n_docs):
        theta = np.array([0.9, 0.1]) if i < n_docs // 2 else np.array([0.1, 0.9])
        theta += rng.normal(0, 0.05, 2)
        theta = np.clip(theta, 0.01, 1)
        theta /= theta.sum()
        zs = rng.choice(2, 40, p=theta)
        doc = np.asarray([rng.choice(vocab, p=phi_true[k]) for k in zs], dtype=np.int64)
        docs.append(doc)
    mdl = lda_gibbs(docs, n_topics=2, vocab=vocab, it=it, seed=seed)
    phi = np.asarray(mdl["phi"])

    # match topics by L1 distance to planted profiles
    def best_match(row: FloatArray) -> float:
        return float(min(np.abs(row - phi_true[0]).sum(), np.abs(row - phi_true[1]).sum()))

    dist = np.asarray([best_match(phi[k]) for k in range(2)])
    out["synthetic_lda_phi_l1_min"] = float(dist.min())
    out["synthetic_lda_phi_l1_mean"] = float(dist.mean())
    if dist.min() > 0.4:
        raise ValueError(f"lda phi off: {dist}")
    # distinctness: the two learned topics must differ
    if np.abs(phi[0] - phi[1]).sum() < 0.5:
        raise ValueError(f"lda topics collapsed: {np.abs(phi[0] - phi[1]).sum()}")
    return out
