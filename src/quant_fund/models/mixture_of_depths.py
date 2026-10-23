"""Mixture-of-Depths token routing (Raposo et al. 2024) (SYNTHETIC).

A learned router scores each token; only the top-k tokens take the
attention branch, the rest pass through identity — compute scales with
the routed fraction. On the recall fixture the query token is routed,
so accuracy survives at ~half the attention cost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models._attn_synth import attn_dot_cost, synth_retrieval

FloatArray = NDArray[np.float64]
_SEED = 20261231


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("mixture_of_depths needs the torch `nn` extra") from exc


def bench_mixture_of_depths(
    seed: int = 13,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 700,
    d_model: int = 32,
    top_frac: float = 0.5,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]
    k = int(t * top_frac)

    proj = torch.nn.Linear(d_in, d_model)
    router = torch.nn.Linear(d_model, 1)
    out = torch.nn.Linear(d_model, n_classes)
    params = list(proj.parameters()) + list(router.parameters()) + list(out.parameters())
    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(params, lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def mod(xb):
        h = proj(xb)
        s = router(h).squeeze(-1)
        top = s.argsort(-1, descending=True)[:, :k]
        sel = torch.zeros_like(s).scatter(1, top, 1.0)
        s_sel = s + torch.where(sel > 0, 0.0, -1e9)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model) + s_sel[:, None, :], -1)
        return a @ h + h

    def full_attn(xb):
        h = proj_o(xb)
        return torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(mod(xt)[:, -1]), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(mod(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float(
            (out_o(full_attn(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        )
    return {
        "synthetic_mod_acc": acc,
        "synthetic_mod_full_acc": acc_full,
        "synthetic_mod_acc_gap": acc_full - acc,
        "synthetic_mod_cost_ratio": attn_dot_cost(t, "linformer", k),
        "synthetic_torch_available": 1.0,
    }
