"""Performer — FAVOR+ random-feature kernelized attention.

Choromanski et al. 2021: softmax(q·k) is approximated by phi(q)·phi(k) with
positive random features phi(x) = exp(-||x||^2/2) exp(w·x)/sqrt(r), turning
attention into (Q' K'^T) V computed as Q' (K'^T V) — O(n r d). Bench: same
long-range retrieval fixture vs full-attention oracle. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_SEED = 20261032
_R_FEAT = 64


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _phi(x: Any, w: Any, torch: Any) -> Any:
    """FAVOR+ positive random features."""
    norm = (x**2).sum(-1, keepdim=True) / 2
    return torch.exp(x @ w - norm) / np.sqrt(w.shape[1])


def bench_performer_attn(
    seed: int = 5,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 900,
    d_model: int = 32,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    w_rand = torch.randn(d_model, _R_FEAT)  # fixed random features
    params = list(proj.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def favor_attn(xb: Any) -> Any:
        h = proj(xb)
        qp = _phi(h, w_rand, torch)  # (B,T,r)
        kp = _phi(h, w_rand, torch)
        kv = kp.transpose(1, 2) @ h  # (B,r,dm)
        num = qp @ kv  # (B,T,dm)
        den = qp @ kp.sum(1)[:, :, None]  # (B,T,1) normalizer
        return num / den.clamp(1e-6, None)

    def full_attn(xb: Any) -> Any:
        h = proj_o(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return a @ h

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(out(favor_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_o = torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt_o.zero_grad()
        loss_o.backward()
        opt_o.step()

    with torch.no_grad():
        xe = torch.tensor(xte, dtype=torch.float32)
        ye = torch.tensor(yte)
        acc_pf = float((out(favor_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
        acc_full = float((out_o(full_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
    cost = attn_dot_cost(t, "performer", _R_FEAT)
    return {
        "synthetic_performer_acc": acc_pf,
        "synthetic_performer_full_acc": acc_full,
        "synthetic_performer_acc_gap": acc_full - acc_pf,
        "synthetic_performer_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
