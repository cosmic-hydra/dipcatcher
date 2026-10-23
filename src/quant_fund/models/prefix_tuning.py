"""Prefix tuning — learned K/V prefixes (Li & Liang 2021) (SYNTHETIC).

Frozen backbone; each attention layer gets learned prefix keys/values
prepended to K and V. On the value-remap fixture prefix states steer
the readout mapping at ~1% of backbone params.
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
        raise ImportError("prefix_tuning needs the torch `nn` extra") from exc


def bench_prefix_tuning(
    seed: int = 143,
    n_train: int = 800,
    n_shift: int = 300,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters_base: int = 700,
    iters_adapt: int = 400,
    d_model: int = 32,
    n_prefix: int = 8,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xs, ys = synth_retrieval(n_shift, m_pairs, n_classes, np.random.default_rng(seed + 1))
    ys = (ys + 1) % n_classes
    d_in = xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(list(proj.parameters()) + list(out.parameters()), lr=3e-3)

    def attn(xb, pk=None, pv=None):
        h = proj(xb)
        k = h if pk is None else torch.cat([pk.expand(xb.shape[0], -1, -1), h], 1)
        v = h if pv is None else torch.cat([pv.expand(xb.shape[0], -1, -1), h], 1)
        a = torch.softmax(h @ k.transpose(1, 2) / np.sqrt(d_model), -1)
        return (a @ v)[:, -1]

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters_base):
        loss = torch.nn.functional.cross_entropy(out(attn(xt)), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in list(proj.parameters()) + list(out.parameters()):
        p.requires_grad_(False)
    xs_t, ys_t = torch.tensor(xs).float(), torch.tensor(ys)
    pk = torch.nn.Parameter(torch.randn(n_prefix, d_model) * 0.1)
    pv = torch.nn.Parameter(torch.randn(n_prefix, d_model) * 0.1)
    opt2 = torch.optim.Adam([pk, pv], lr=5e-3)
    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(out(attn(xs_t, pk, pv)), ys_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_pre = float((out(attn(xs_t)).argmax(-1) == ys_t).float().mean())
        acc_pref = float((out(attn(xs_t, pk, pv)).argmax(-1) == ys_t).float().mean())
    n_full = sum(p.numel() for p in list(proj.parameters()) + list(out.parameters()))
    return {
        "synthetic_prefix_acc_shift": acc_pref,
        "synthetic_prefix_frozen_acc": acc_pre,
        "synthetic_prefix_param_frac": float(2 * n_prefix * d_model) / n_full,
        "synthetic_torch_available": 1.0,
    }
