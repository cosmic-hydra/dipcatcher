"""RetNet retention with exponential decay mask (Sun et al. 2023) (SYNTHETIC).

Attention scores carry a causal decay γ^{i-j} — retention keeps full
content addressing (unlike conv scans) but discounts distant tokens.
On recall the decay costs little since the queried pair is content-keyed.
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
        raise ImportError("retnet_decay needs the torch `nn` extra") from exc


def bench_retnet_decay(
    seed: int = 9,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 700,
    d_model: int = 48,
    gamma: float = 0.92,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(list(proj.parameters()) + list(out.parameters()), lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)
    ii, jj = np.meshgrid(np.arange(t), np.arange(t), indexing="ij")
    decay = torch.tensor(np.where(jj <= ii, gamma ** (ii - jj), 0.0)).float()

    def retnet(xb):
        h = proj(xb)
        s = h @ h.transpose(1, 2) / np.sqrt(d_model)
        a = torch.softmax(s + torch.log(decay + 1e-12), -1)
        return a @ h

    def full_attn(xb):
        h = proj_o(xb)
        return torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(retnet(xt)[:, -1]), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(retnet(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float(
            (out_o(full_attn(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        )
    return {
        "synthetic_retnet_acc": acc,
        "synthetic_retnet_full_acc": acc_full,
        "synthetic_retnet_acc_gap": acc_full - acc,
        "synthetic_retnet_cost_ratio": 1.0,
        "synthetic_torch_available": 1.0,
    }
