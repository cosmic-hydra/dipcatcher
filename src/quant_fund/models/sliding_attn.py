"""Sliding-window (local) attention — Longformer-style O(n w) attention.

Beltagy et al. 2020: each token attends only within a fixed window, giving
linear-in-n cost and a local inductive bias; distant information must be
carried by the positional/feed-forward stack. Bench: the retrieval fixture —
expected to underperform on far-back queries (honest negative where window <
distance), reported vs full-attention oracle. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_SEED = 20261034
_WIN = 33  # odd window centered on each token


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_sliding_attn(
    seed: int = 9,
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
    params = list(proj.parameters()) + list(out.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    half = _WIN // 2
    mask = torch.full((t, t), -1e9)
    for i in range(t):
        lo, hi = max(0, i - half), min(t, i + half + 1)
        mask[i, lo:hi] = 0.0

    def slide_attn(xb: Any) -> Any:
        h = proj(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model) + mask, -1)
        return a @ h

    def full_attn(xb: Any) -> Any:
        h = proj_o(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return a @ h

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(out(slide_attn(xt)[:, -1]), yt)
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
        acc_sw = float((out(slide_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
        acc_full = float((out_o(full_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
    cost = attn_dot_cost(t, "sliding", _WIN)
    return {
        "synthetic_sliding_acc": acc_sw,
        "synthetic_sliding_full_acc": acc_full,
        "synthetic_sliding_acc_gap": acc_full - acc_sw,
        "synthetic_sliding_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
