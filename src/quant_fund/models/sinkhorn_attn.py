"""Sinkhorn attention — learned block permutation for block-sparse attention.

Tay et al. 2020: tokens are sorted into buckets by a differentiable
doubly-stochastic sorting matrix (Sinkhorn iterations on a learned score);
each token then attends within its own + adjacent bucket — content-based
sparsity instead of positional windows. Bench: retrieval fixture vs full
oracle; reports cost ratio. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_SEED = 20261035
_N_BUCKETS = 8
_SK_ITERS = 6


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def _sinkhorn(logits: Any, iters: int, torch: Any) -> Any:
    """Doubly-stochastic normalizing iterations over the block axis."""
    m = logits
    for _ in range(iters):
        m = m - m.logsumexp(-1, keepdim=True)
        m = m - m.logsumexp(-2, keepdim=True)
    return torch.exp(m)


def bench_sinkhorn_attn(
    seed: int = 11,
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
    sort_lin = torch.nn.Linear(d_model, _N_BUCKETS)
    out = torch.nn.Linear(d_model, n_classes)
    params = list(proj.parameters()) + list(sort_lin.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def sk_attn(xb: Any) -> Any:
        h = proj(xb)  # (B,T,dm)
        scores = sort_lin(h.mean(1, keepdim=True).expand(-1, t, -1))  # (B,T,nb)
        sort_p = _sinkhorn(scores, _SK_ITERS, torch)  # (B,T,nb)
        perm = sort_p.argmax(-1)  # bucket id per token
        a_full = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        # block-sparse mask: same bucket or adjacent bucket
        same = perm[:, :, None] == perm[:, None, :]
        adj = (perm[:, :, None] - perm[:, None, :]).abs() == 1
        allow = (same | adj).float()
        a = a_full * allow
        a = a / a.sum(-1, keepdim=True).clamp(1e-9, None)
        return a @ h

    def full_attn(xb: Any) -> Any:
        h = proj_o(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return a @ h

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(out(sk_attn(xt)[:, -1]), yt)
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
        acc_sk = float((out(sk_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
        acc_full = float((out_o(full_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
    cost = attn_dot_cost(t, "sinkhorn", _N_BUCKETS)
    return {
        "synthetic_sinkhorn_acc": acc_sk,
        "synthetic_sinkhorn_full_acc": acc_full,
        "synthetic_sinkhorn_acc_gap": acc_full - acc_sk,
        "synthetic_sinkhorn_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
