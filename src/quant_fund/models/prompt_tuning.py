"""Prompt tuning — learnable soft tokens (Lester et al. 2021) (SYNTHETIC).

Frozen attention backbone + p learned embeddings prepended to every
sequence. On the value-remap fixture (same keys, rotated value classes)
soft prompts drive the readout remap while the backbone never moves.
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
        raise ImportError("prompt_tuning needs the torch `nn` extra") from exc


def _remap(y: np.ndarray) -> np.ndarray:
    return (y + 1) % int(y.max() + 1)


def bench_prompt_tuning(
    seed: int = 141,
    n_train: int = 800,
    n_shift: int = 300,
    m_pairs: int = 48,
    n_classes: int = 4,
    iters_base: int = 700,
    iters_adapt: int = 400,
    d_model: int = 32,
    n_prompt: int = 8,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    xtr, ytr = synth_retrieval(n_train, m_pairs, n_classes, rng)
    xs, ys = synth_retrieval(n_shift, m_pairs, n_classes, np.random.default_rng(seed + 1))
    ys = _remap(ys)
    d_in = xtr.shape[2]

    proj = torch.nn.Linear(d_in, d_model)
    out = torch.nn.Linear(d_model, n_classes)
    opt = torch.optim.Adam(list(proj.parameters()) + list(out.parameters()), lr=3e-3)

    def attn(xb, prompt=None):
        h = proj(xb)
        if prompt is not None:
            h = torch.cat([prompt.expand(xb.shape[0], -1, -1), h], 1)
        a = torch.softmax(h @ h.transpose(1, 2) / np.sqrt(d_model), -1)
        return (a @ h)[:, -1]

    xt, yt = torch.tensor(xtr).float(), torch.tensor(ytr)
    for _i in range(iters_base):
        loss = torch.nn.functional.cross_entropy(out(attn(xt)), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in list(proj.parameters()) + list(out.parameters()):
        p.requires_grad_(False)
    xs_t, ys_t = torch.tensor(xs).float(), torch.tensor(ys)
    with torch.no_grad():
        acc_frozen = float((out(attn(xs_t)).argmax(-1) == ys_t).float().mean())
    prompt = torch.nn.Parameter(torch.randn(n_prompt, d_model) * 0.1)
    opt2 = torch.optim.Adam([prompt], lr=5e-3)
    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(out(attn(xs_t, prompt)), ys_t)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        acc_prompt = float((out(attn(xs_t, prompt)).argmax(-1) == ys_t).float().mean())
    for p in list(proj.parameters()) + list(out.parameters()):
        p.requires_grad_(True)
    opt3 = torch.optim.Adam(list(proj.parameters()) + list(out.parameters()), lr=3e-3)
    for _i in range(iters_adapt):
        loss = torch.nn.functional.cross_entropy(out(attn(xs_t)), ys_t)
        opt3.zero_grad()
        loss.backward()
        opt3.step()
    with torch.no_grad():
        acc_full = float((out(attn(xs_t)).argmax(-1) == ys_t).float().mean())
    n_full = sum(p.numel() for p in list(proj.parameters()) + list(out.parameters()))
    return {
        "synthetic_prompt_acc_shift": acc_prompt,
        "synthetic_prompt_frozen_acc": acc_frozen,
        "synthetic_prompt_full_acc": acc_full,
        "synthetic_prompt_param_frac": float(n_prompt * d_model) / n_full,
        "synthetic_torch_available": 1.0,
    }
