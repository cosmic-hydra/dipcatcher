"""DeltaNet fast-weight linear attention (Schlag et al. 2021) (SYNTHETIC).

F_t = F_{t-1}(I − β k kᵀ) + β v kᵀ — the delta rule removes the stale
value before writing, giving true associative recall O(T·d²). On the
recall fixture it approaches full attention at linear cost, unlike the
decay-only RWKV.
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
        raise ImportError("delta_net needs the torch `nn` extra") from exc


def bench_delta_net(
    seed: int = 11,
    n_train: int = 800,
    n_test: int = 240,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters: int = 700,
    d_model: int = 16,
    beta: float = 1.0,
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
    w_q = torch.nn.Linear(d_model, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    params = (
        list(proj.parameters())
        + list(w_k.parameters())
        + list(w_v.parameters())
        + list(w_q.parameters())
        + list(out.parameters())
    )
    proj_o = torch.nn.Linear(d_in, d_model)
    out_o = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(params, lr=3e-3)
    opt_o = torch.optim.Adam(list(proj_o.parameters()) + list(out_o.parameters()), lr=3e-3)

    def delta(xb):
        h = proj(xb)
        eye = torch.eye(d_model)[None]
        f = torch.zeros(xb.shape[0], d_model, d_model)
        ys = []
        for i in range(xb.shape[1]):
            k = torch.nn.functional.normalize(w_k(h[:, i]), dim=-1)
            v = w_v(h[:, i])
            f = f @ (eye - beta * torch.einsum("bi,bj->bij", k, k)) + beta * torch.einsum(
                "bi,bj->bij", v, k
            )
            ys.append(torch.einsum("bij,bj->bi", f, w_q(h[:, i])))
        return torch.stack(ys, 1)

    def full_attn(xb):
        h = proj_o(xb)
        return torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1) @ h

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(
            out(delta(xt)[:, -1]), yt
        ) + torch.nn.functional.cross_entropy(out_o(full_attn(xt)[:, -1]), yt)
        opt.zero_grad()
        opt_o.zero_grad()
        loss.backward()
        opt.step()
        opt_o.step()
    xe = torch.tensor(xte).float()
    with torch.no_grad():
        acc = float((out(delta(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean())
        acc_full = float(
            (out_o(full_attn(xe)[:, -1]).argmax(-1) == torch.tensor(yte)).float().mean()
        )
    return {
        "synthetic_delta_acc": acc,
        "synthetic_delta_full_acc": acc_full,
        "synthetic_delta_acc_gap": acc_full - acc,
        "synthetic_delta_cost_ratio": float(t * d_model * d_model) / float(t * t * 48),
        "synthetic_torch_available": 1.0,
    }
