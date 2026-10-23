"""RWKV WKV recurrent attention (Peng et al. 2023) (SYNTHETIC).

y_t = σ(r_t) ⊙ Σ_i e^{-(t-i)γ} k_i v_i / Σ_i e^{-(t-i)γ} k_i — a linear
recurrence that can do content routing through learned k/r. On the
associative-recall fixture it partially recovers lookup ability vs the
pure-convolution S4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import synth_retrieval

FloatArray = NDArray[np.float64]
_SEED = 20261231


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("rwkv_wkv needs the torch `nn` extra") from exc


def bench_rwkv_wkv(
    seed: int = 5,
    n_train: int = 120,
    n_test: int = 40,
    m_pairs: int = 8,
    n_classes: int = 4,
    iters: int = 30,
    d_model: int = 48,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    w_k = torch.nn.Linear(d_model, d_model)
    w_v = torch.nn.Linear(d_model, d_model)
    w_r = torch.nn.Linear(d_model, d_model)
    gamma = torch.nn.Parameter(0.05 * torch.randn(d_model))
    out = torch.nn.Linear(d_model, n_classes)
    params = (
        list(proj.parameters())
        + list(w_k.parameters())
        + list(w_v.parameters())
        + list(w_r.parameters())
        + list(out.parameters())
        + [gamma]
    )

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(params, lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def rwkv(xb):
        # Unrolled form of num_t = exp(gamma) * num_{t-1} + k_t v_t.
        # The linear maps run once over the sequence; the time sum is one
        # GEMM instead of a Python loop of tiny kernels (that loop was the
        # bench's wall clock under a single OpenMP thread).
        h = proj(xb)
        k = torch.sigmoid(w_k(h))
        v = w_v(h)
        r = torch.sigmoid(w_r(h))
        steps = h.shape[1]
        pos = torch.arange(steps, device=h.device)
        lag = (pos[:, None] - pos[None, :]).to(dtype=h.dtype)
        causal = (lag >= 0).to(dtype=h.dtype)
        decay = torch.exp(gamma.view(1, 1, -1) * lag.clamp(min=0).unsqueeze(-1))
        decay = decay * causal.unsqueeze(-1)
        num = torch.einsum("tid,bid->btd", decay, k * v)
        den = torch.einsum("tid,bid->btd", decay, k)
        return r * num / (den + 1e-6)

    def full_attn(xb):
        h = proj_o(xb)
        return torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(rwkv(xt)[:, -1]), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(rwkv(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float(
            (out_o(full_attn(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        )
    return {
        "synthetic_rwkv_acc": acc,
        "synthetic_rwkv_full_acc": acc_full,
        "synthetic_rwkv_acc_gap": acc_full - acc,
        "synthetic_rwkv_cost_ratio": float(t * d_model) / float(t * t),
        "synthetic_torch_available": 1.0,
    }
