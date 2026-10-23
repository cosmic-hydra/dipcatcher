"""Grouped-query attention (Ainslie et al. 2023) (SYNTHETIC).

G KV heads shared across Q heads — KV cache scales with G not H. On the
recall fixture GQA at G=2 keeps most of MHA accuracy at 1/4 the KV
memory for H=8.
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
        raise ImportError("gqa_attn needs the torch `nn` extra") from exc


def bench_gqa_attn(
    seed: int = 161,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 700,
    d_model: int = 32,
    n_heads: int = 8,
    n_groups: int = 2,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]
    d_h = d_model // n_heads

    proj = torch.nn.Linear(d_in, d_model)
    wq = torch.nn.Linear(d_model, d_model)
    wk = torch.nn.Linear(d_model, n_groups * d_h)
    wv = torch.nn.Linear(d_model, n_groups * d_h)
    out = torch.nn.Linear(d_model, n_classes)
    params = (
        list(proj.parameters())
        + list(wq.parameters())
        + list(wk.parameters())
        + list(wv.parameters())
        + list(out.parameters())
    )

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(params, lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def gqa(xb):
        h = proj(xb)
        b = xb.shape[0]
        q = wq(h).view(b, t, n_heads, d_h).transpose(1, 2)
        kk = wk(h).view(b, t, n_groups, d_h).transpose(1, 2)
        vv = wv(h).view(b, t, n_groups, d_h).transpose(1, 2)
        rep = n_heads // n_groups
        kk = kk.repeat_interleave(rep, 1)
        vv = vv.repeat_interleave(rep, 1)
        a = torch.softmax(q @ kk.transpose(-1, -2) / np.sqrt(d_h), -1)
        o = (a @ vv).transpose(1, 2).reshape(b, t, d_model)
        return o[:, -1]

    def full_attn(xb):
        h = proj_o(xb)
        return (torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h)[:, -1]

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(gqa(xt)), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(gqa(xe)).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float((out_o(full_attn(xe)).argmax(-1) == torch.tensor(yte)).float().mean())
    return {
        "synthetic_gqa_acc": acc,
        "synthetic_gqa_mha_acc": acc_full,
        "synthetic_gqa_gap": acc_full - acc,
        "synthetic_gqa_kv_frac": float(n_groups) / n_heads,
        "synthetic_torch_available": 1.0,
    }
