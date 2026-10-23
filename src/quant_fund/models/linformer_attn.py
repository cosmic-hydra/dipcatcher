"""Linformer — low-rank self-attention via learned sequence projections.

Wang et al. 2020: attention scores are low-rank, so projecting K and V along
the sequence dimension (n -> k) approximates full attention at O(n k) cost.
Bench: long-range retrieval fixture — accuracy vs a full-attention oracle and
the dot-cost ratio. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_SEED = 20261031
_T_MAX = 129  # 2*64 pairs + query
_K_PROJ = 16


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_linformer_attn(
    seed: int = 3,
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
    e_k = torch.nn.Parameter(torch.randn(t, _K_PROJ) * 0.1)  # seq-dim proj for K
    e_v = torch.nn.Parameter(torch.randn(t, _K_PROJ) * 0.1)
    out = torch.nn.Linear(d_model, n_classes)
    params = list(proj.parameters()) + list(out.parameters()) + [e_k, e_v]
    opt = torch.optim.Adam(params, lr=3e-3)

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def lin_attn(xb: Any) -> Any:
        h = proj(xb)  # (B,T,dm)
        k = h.transpose(1, 2) @ e_k  # (B,dm,k)
        v = h.transpose(1, 2) @ e_v  # (B,dm,k)
        scores = h @ k / np.sqrt(d_model)  # (B,T,k)
        a = torch.softmax(scores, -1)
        return a @ v.transpose(1, 2)  # (B,T,dm)

    def full_attn(xb: Any) -> Any:
        h = proj_o(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return a @ h

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        logits = out(lin_attn(xt)[:, -1])
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logits_o = out_o(full_attn(xt)[:, -1])
        loss_o = torch.nn.functional.cross_entropy(logits_o, yt)
        opt_o.zero_grad()
        loss_o.backward()
        opt_o.step()

    with torch.no_grad():
        xe = torch.tensor(xte, dtype=torch.float32)
        ye = torch.tensor(yte)
        acc_lin = float((out(lin_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
        acc_full = float((out_o(full_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
    cost = attn_dot_cost(t, "linformer", _K_PROJ)
    return {
        "synthetic_linformer_acc": acc_lin,
        "synthetic_linformer_full_acc": acc_full,
        "synthetic_linformer_acc_gap": acc_full - acc_lin,
        "synthetic_linformer_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
