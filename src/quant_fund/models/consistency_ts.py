"""Consistency model for few-step TS generation (SYNTHETIC).

Song et al. 2023: train a single network to map any point on the
PF-ODE trajectory to its endpoint — generate in 1-2 steps instead of
diffusion's hundreds. Distillation-lite: teacher is the analytic
linear ODE toward a real sample (self-distillation).

Bench: bimodal windows; 2-step consistency sampler (MMD) vs the
8-step flow baseline and Gaussian.
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
        raise ImportError("consistency_ts requires the `nn` extra (make sync)") from exc


def synth_bimodal(n: int, win: int, rng: np.random.Generator) -> FloatArray:
    x = np.zeros((n, win))
    for i in range(n):
        if i % 2 == 0:
            x[i] = np.sin(np.linspace(0, 2 * np.pi, win) + rng.uniform(0, 1))
        else:
            x[i] = np.cumsum(0.4 * rng.standard_normal(win))
            x[i] -= x[i].mean()
    return x.astype(np.float64)


def _mmd(x: FloatArray, y: FloatArray, bw: float) -> float:
    def k(a: FloatArray, b: FloatArray) -> FloatArray:
        d = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T
        return np.exp(-d / (2 * bw**2))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def bench_consistency_ts(
    seed: int = 20261231,
    n: int = 320,
    win: int = 24,
    iters: int = 1400,
) -> dict[str, float]:
    """Consistency-model 2-step sampler vs Gaussian (MMD, SYNTH)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs = synth_bimodal(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    te = xs[3 * n // 4 :]
    T_max = 4.0
    n_sig = 16
    sigs = np.geomspace(0.05, T_max, n_sig)

    net = torch.nn.Sequential(
        torch.nn.Linear(win + 1, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, win),
    )
    tgt_net = torch.nn.Sequential(
        torch.nn.Linear(win + 1, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, win),
    )
    for a_, b_ in zip(tgt_net.parameters(), net.parameters(), strict=True):
        a_.data.copy_(b_.data)
    params = torch.nn.ModuleList([net])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    for _i in range(iters):
        b_idx = rng.integers(0, 3 * n // 4, 128)
        x0 = xb[b_idx]
        k = rng.integers(1, n_sig, len(b_idx))
        sig_hi = torch.tensor(sigs[k], dtype=torch.float32).unsqueeze(1)
        sig_lo = torch.tensor(sigs[k - 1], dtype=torch.float32).unsqueeze(1)
        z = torch.randn_like(x0)
        x_hi = x0 + sig_hi * z
        x_lo = x0 + sig_lo * z
        f_hi = net(torch.cat([x_hi, sig_hi], 1))
        with torch.no_grad():
            f_lo = tgt_net(torch.cat([x_lo, sig_lo], 1))
        # CT: x0 boundary anchor + cross-level self-consistency —
        # the x0 term is what stops the collapse seen in pure
        # self-distillation
        loss = torch.mean((f_hi - x0) ** 2) + 0.5 * torch.mean((f_hi - f_lo) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        for a_, b_ in zip(tgt_net.parameters(), net.parameters(), strict=True):
            a_.data.copy_(0.99 * a_.data + 0.01 * b_.data)

    m = len(te)
    with torch.no_grad():
        z = torch.randn(m, win) * T_max
        t = torch.full((m, 1), T_max)
        x = net(torch.cat([z, t], 1))  # 1-step
        z2 = x + torch.randn(m, win) * sigs[3]
        x = net(torch.cat([z2, torch.full((m, 1), float(sigs[3]))], 1))
        gen = x.numpy()
    mmd_c = _mmd(gen, te, bw=1.0)
    gauss = np.random.default_rng(seed + 1).normal(
        xs[: 3 * n // 4].mean(0), xs[: 3 * n // 4].std(0), (m, win)
    )
    mmd_g = _mmd(gauss.astype(np.float64), te, bw=1.0)
    return {
        "synthetic_consistency_mmd": mmd_c,
        "synthetic_consistency_gauss_mmd": mmd_g,
        "synthetic_consistency_margin_vs_gauss": mmd_g - mmd_c,
        "synthetic_consistency_steps": 2.0,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_consistency_ts()))
