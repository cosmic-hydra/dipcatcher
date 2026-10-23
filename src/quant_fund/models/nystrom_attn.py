"""Nyström attention — landmark-based low-rank softmax approximation.

Xiong et al. 2021: pick m landmark points, compute the m x m landmark kernel
matrix, and reconstruct attention as A_qm A_mm^+ A_mk — the Nyström
approximation of the full softmax matrix at O(n m) cost. Bench: retrieval
fixture vs full-attention oracle. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]

_SEED = 20261036
_M_LAND = 16


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def bench_nystrom_attn(
    seed: int = 13,
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

    def nystrom_attn(xb: Any) -> Any:
        h = proj(xb)  # (B,T,dm)
        idx = torch.linspace(0, t - 1, _M_LAND).long()
        q_lm = h[:, idx]  # (B,m,dm)
        k_lm = h[:, idx]
        a_qm = torch.softmax(h @ k_lm.transpose(1, 2) / np.sqrt(d_model), -1)  # (B,T,m)
        a_mm = torch.softmax(q_lm @ k_lm.transpose(1, 2) / np.sqrt(d_model), -1)
        a_mk = torch.softmax(q_lm @ h.transpose(1, 2) / np.sqrt(d_model), -1)  # (B,m,T)
        # Nyström: A ~ A_qm A_mm^+ A_mk ; pseudo-inverse via solve
        a_mm_inv = torch.linalg.pinv(a_mm + 1e-4 * torch.eye(_M_LAND))
        a = a_qm @ a_mm_inv @ a_mk
        a = a / a.sum(-1, keepdim=True).clamp(1e-9, None)
        return a @ h

    def full_attn(xb: Any) -> Any:
        h = proj_o(xb)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return a @ h

    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(out(nystrom_attn(xt)[:, -1]), yt)
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
        acc_ny = float((out(nystrom_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
        acc_full = float((out_o(full_attn(xe)[:, -1]).argmax(-1) == ye).float().mean())
    cost = attn_dot_cost(t, "nystrom", _M_LAND)
    return {
        "synthetic_nystrom_acc": acc_ny,
        "synthetic_nystrom_full_acc": acc_full,
        "synthetic_nystrom_acc_gap": acc_full - acc_ny,
        "synthetic_nystrom_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
