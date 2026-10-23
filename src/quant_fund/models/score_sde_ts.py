"""Score-based SDE diffusion for TS generation (SYNTHETIC).

Song et al. 2021: learn the score of noise-perturbed data at all
noise scales (VP SDE), then sample by solving the reverse SDE.
Complements the DDPM-style `ts_diffusion` (discrete steps, MSE on
noise) — this one is the continuous-time score matching view.

Bench: bimodal synthetic windows; score-SDE sampler quality (MMD)
vs Gaussian baseline; also reports the denoising-score-match loss.
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
        raise ImportError("score_sde_ts requires the `nn` extra (make sync)") from exc


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


def bench_score_sde_ts(
    seed: int = 20261231,
    n: int = 320,
    win: int = 24,
    iters: int = 1200,
    gen_steps: int = 60,
    sig_min: float = 0.02,
    sig_max: float = 2.0,
) -> dict[str, float]:
    """Score-matching SDE sampler vs Gaussian baseline (MMD, SYNTH)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs = synth_bimodal(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    te = xs[3 * n // 4 :]

    net = torch.nn.Sequential(
        torch.nn.Linear(win + 1, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, 96),
        torch.nn.ReLU(),
        torch.nn.Linear(96, win),
    )
    params = torch.nn.ModuleList([net])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    for _i in range(iters):
        b_idx = rng.integers(0, 3 * n // 4, 128)
        x = xb[b_idx]
        sig = torch.tensor(
            np.exp(
                np.linspace(np.log(sig_min), np.log(sig_max), 20)[rng.integers(0, 20, len(b_idx))]
            ),
            dtype=torch.float32,
        ).unsqueeze(1)
        z = torch.randn_like(x)
        xt = x + sig * z
        score = net(torch.cat([xt, torch.log(sig)], 1))
        loss = torch.mean((sig * score + z) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    m = len(te)
    with torch.no_grad():
        x = torch.randn(m, win) * sig_max
        dts = np.log(sig_max / sig_min) / gen_steps
        for s in range(gen_steps):
            sig_t = sig_max * np.exp(-dts * s)
            tt = torch.full((m, 1), float(np.log(sig_t)))
            sc = net(torch.cat([x, tt], 1))
            # reverse Euler-Maruyama for the VE SDE: dx = -g^2 score dt + g dW
            g2 = float(2 * sig_t**2 * dts)
            x = x + g2 * sc + float(np.sqrt(g2)) * torch.randn_like(x)
        gen = x.numpy()
    mmd_sde = _mmd(gen, te, bw=1.0)
    gauss = np.random.default_rng(seed + 1).normal(
        xs[: 3 * n // 4].mean(0), xs[: 3 * n // 4].std(0), (m, win)
    )
    mmd_gauss = _mmd(gauss.astype(np.float64), te, bw=1.0)
    return {
        "synthetic_scoresde_mmd": mmd_sde,
        "synthetic_scoresde_gauss_mmd": mmd_gauss,
        "synthetic_scoresde_margin_vs_gauss": mmd_gauss - mmd_sde,
        "synthetic_scoresde_steps": float(gen_steps),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_score_sde_ts()))
