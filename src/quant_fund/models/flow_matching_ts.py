"""Conditional flow matching for TS generation (rectified flow) (SYNTHETIC).

Lipman et al. 2023 / Liu et al. 2022: regress the straight-line vector
field `dx/dt = x1 - x0` along the interpolation path; generate by
Euler-integrating the learned field from noise — far fewer steps than
score diffusion.

Bench: synthetic window generation conditioned on a regime bit;
sample quality via MMD to held-out real windows vs the DDPM-style
baseline from ts_diffusion's noise-regression equivalent, and vs a
Gaussian-moment-match sampler. Metric: MMD (lower better) + steps.
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
        raise ImportError("flow_matching_ts requires the `nn` extra (make sync)") from exc


def synth_regime_windows(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win) + binary regime: reg 0 = smooth sine, reg 1 = jagged."""
    x = np.zeros((n, win))
    reg = np.zeros(n)
    for i in range(n):
        if i % 2 == 0:
            x[i] = np.sin(np.linspace(0, 2 * np.pi, win) + rng.uniform(0, 1))
        else:
            x[i] = np.cumsum(0.4 * rng.standard_normal(win))
            x[i] -= x[i].mean()
        reg[i] = float(i % 2)
    return x.astype(np.float64), reg.astype(np.float64)


def _mmd(x: FloatArray, y: FloatArray, bw: float) -> float:
    def k(a: FloatArray, b: FloatArray) -> FloatArray:
        d = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None] - 2 * a @ b.T
        return np.exp(-d / (2 * bw**2))

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def bench_flow_matching_ts(
    seed: int = 20261231,
    n: int = 320,
    win: int = 24,
    iters: int = 1200,
    gen_steps: int = 8,
) -> dict[str, float]:
    """Rectified-flow sampler vs moment-matched Gaussian (MMD, SYNTH)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, reg = synth_regime_windows(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    cb = torch.tensor(reg, dtype=torch.float32).unsqueeze(1)
    te = xs[3 * n // 4 :]

    net = torch.nn.Sequential(
        torch.nn.Linear(win + 2, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, win),
    )
    params = torch.nn.ModuleList([net])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    for _i in range(iters):
        b_idx = rng.integers(0, 3 * n // 4, 128)
        x1 = xb[b_idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(b_idx), 1)
        xt = t * x1 + (1 - t) * x0
        v = net(torch.cat([xt, t, cb[b_idx]], 1))
        loss = torch.mean((v - (x1 - x0)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m = len(te)
    with torch.no_grad():
        x = torch.randn(m, win)
        cond = cb[3 * n // 4 :]
        for s in range(gen_steps):
            tt = torch.full((m, 1), s / gen_steps)
            x = x + net(torch.cat([x, tt, cond], 1)) / gen_steps
        gen = x.numpy()
    mmd_flow = _mmd(gen, te, bw=1.0)
    gauss = np.random.default_rng(seed + 1).normal(
        xs[: 3 * n // 4].mean(0), xs[: 3 * n // 4].std(0), (m, win)
    )
    mmd_gauss = _mmd(gauss.astype(np.float64), te, bw=1.0)
    return {
        "synthetic_flow_mmd": mmd_flow,
        "synthetic_flow_gauss_mmd": mmd_gauss,
        "synthetic_flow_margin_vs_gauss": mmd_gauss - mmd_flow,
        "synthetic_flow_steps": float(gen_steps),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_flow_matching_ts()))
