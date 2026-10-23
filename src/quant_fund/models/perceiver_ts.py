"""Perceiver-IO forecaster: latent cross-attention for long inputs (SYNTHETIC).

Jaegle et al. 2021: a small latent array cross-attends to the (long)
input, then self-attends among latents — compute scales linearly in
input length instead of quadratically.

Bench: long synthetic windows (T=256) where the target is a slow
moving-average + local spike interaction; perceiver (32 latents) vs
full self-attention (baseline) and mean-pool MLP — metric MAE +
attention-cost proxy (n_attn_dot products).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("perceiver_ts requires the `nn` extra (make sync)") from exc


def synth_long_window(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """(n, win); y = tanh(mean(x) + 1.5*max_spike) — global + local cue."""
    x = np.cumsum(0.2 * rng.standard_normal((n, win)), 1)
    x -= x.mean(1, keepdims=True)
    spike_idx = rng.integers(0, win, n)
    x[np.arange(n), spike_idx] += 1.2 * rng.standard_normal(n)
    y = np.tanh(x.mean(1) + 1.5 * np.abs(x).max(1))
    return x.astype(np.float64), y.astype(np.float64)


def bench_perceiver_ts(
    seed: int = 20261231,
    n: int = 240,
    win: int = 256,
    n_lat: int = 32,
    iters: int = 800,
) -> dict[str, float]:
    """32-latent perceiver vs full attention and pooled MLP (SYNTH)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, y = synth_long_window(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)  # (B,T,1)
    yb = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    tr = slice(0, 3 * n // 4)
    te = slice(3 * n // 4, n)
    d = 32

    in_emb = torch.nn.Linear(1, d)
    lat = torch.nn.Parameter(torch.randn(1, n_lat, d) * 0.05)
    x_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    s_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    head = torch.nn.Linear(d, 1)
    params = torch.nn.ModuleList([in_emb, x_attn, s_attn, head])
    opt = torch.optim.Adam(list(params.parameters()) + [lat], lr=1e-3)
    for _i in range(iters):
        z = xb[tr]
        kv = in_emb(z)
        lv = lat.expand(z.shape[0], -1, -1)
        lv, _ = x_attn(lv, kv, kv)
        lv, _ = s_attn(lv, lv, lv)
        pred = head(lv.mean(1))
        loss = torch.mean((pred.squeeze(1) - yb[tr].squeeze(1)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        kv = in_emb(xb[te])
        lv = lat.expand(xb[te].shape[0], -1, -1)
        lv, _ = x_attn(lv, kv, kv)
        lv, _ = s_attn(lv, lv, lv)
        perc_mae = float(torch.mean(torch.abs(head(lv.mean(1)).squeeze(1) - yb[te].squeeze(1))))

    # full self-attention baseline
    full_attn = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    f_head = torch.nn.Linear(d, 1)
    f_emb = torch.nn.Linear(1, d)
    pf = torch.nn.ModuleList([full_attn, f_head, f_emb])
    optf = torch.optim.Adam(pf.parameters(), lr=1e-3)
    for _i in range(iters):
        h0 = f_emb(xb[tr])
        h, _ = full_attn(h0, h0, h0)
        loss = torch.mean((f_head(h.mean(1)).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optf.zero_grad()
        loss.backward()
        optf.step()
    with torch.no_grad():
        h0 = f_emb(xb[te])
        h, _ = full_attn(h0, h0, h0)
        fa_mae = float(torch.mean(torch.abs(f_head(h.mean(1)).squeeze(1) - yb[te].squeeze(1))))
    pool = torch.nn.Sequential(torch.nn.Linear(win, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    pp = torch.nn.ModuleList([pool])
    optp = torch.optim.Adam(pp.parameters(), lr=1e-3)
    for _i in range(iters):
        loss = torch.mean((pool(xb[tr].squeeze(-1)).squeeze(1) - yb[tr].squeeze(1)) ** 2)
        optp.zero_grad()
        loss.backward()
        optp.step()
    with torch.no_grad():
        mp_mae = float(
            torch.mean(torch.abs(pool(xb[te].squeeze(-1)).squeeze(1) - yb[te].squeeze(1)))
        )
    return {
        "synthetic_perceiver_mae": perc_mae,
        "synthetic_perceiver_fullattn_mae": fa_mae,
        "synthetic_perceiver_poolmlp_mae": mp_mae,
        "synthetic_perceiver_margin_vs_fullattn": fa_mae - perc_mae,
        "synthetic_perceiver_dot_ratio": (n_lat * win + n_lat * n_lat) / (win * win),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_perceiver_ts()))
