"""S4-style diagonal state-space model (Gu et al. 2021) (SYNTHETIC).

h_t = a ⊙ h_{t-1} + B x_t with learned per-dim decay a — an O(T·h)
scan vs attention's O(T²). On the associative-recall fixture the pure
convolutional scan cannot do content-addressed lookup — an honest
negative vs full attention, the documented SSM limitation.
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
        raise ImportError("s4_ssm needs the torch `nn` extra") from exc


def bench_s4_ssm(
    seed: int = 3,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 700,
    d_model: int = 48,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xte, yte = synth_retrieval(n_test, m_pairs, n_classes, rng)
    t, d_in = xtr.shape[1], xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    log_a = torch.nn.Parameter(-0.5 + 0.1 * torch.randn(d_model))
    b_in = torch.nn.Linear(d_model, d_model)
    c_out = torch.nn.Linear(d_model, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    params = (
        list(proj.parameters())
        + list(b_in.parameters())
        + list(c_out.parameters())
        + list(out.parameters())
        + [log_a]
    )

    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(params, lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def s4(xb):
        h = proj(xb)
        a = torch.exp(-torch.nn.functional.softplus(log_a))[None, :]
        state = torch.zeros(xb.shape[0], d_model)
        ys = []
        for i in range(xb.shape[1]):
            state = a * state + b_in(h[:, i])
            ys.append(c_out(state))
        return torch.stack(ys, 1)

    def full_attn(xb):
        h = proj_o(xb)
        return torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(s4(xt)[:, -1]), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(s4(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float(
            (out_o(full_attn(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        )
    cost = float(t * d_model) / float(t * t)
    return {
        "synthetic_s4_acc": acc,
        "synthetic_s4_full_acc": acc_full,
        "synthetic_s4_acc_gap": acc_full - acc,
        "synthetic_s4_cost_ratio": cost,
        "synthetic_torch_available": 1.0,
    }
