"""Kolmogorov-Arnold Network forecaster (torch).

Edges are learnable univariate functions — a Fourier/harmonic basis
expansion with per-edge coefficients — rather than fixed scalar weights:
out_j = phi_j( sum_i f_ij(x_i) ) with f_ij a learned basis expansion.
Requires the ``nn`` extra; SYNTHETIC series only.

Bench: next-step regression on a series whose map is sharply nonlinear
in each lag coordinate (harmonic + sign structure) — KAN's univariate
edge functions should fit it with far fewer effective degrees than an
MLP of comparable size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("kan_forecaster torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_nonlinear(n: int, lag: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """y_t = sin(3*x1) + 0.7*|x2|^0.5*sign(x2) + 0.4*tanh(4*x3) + noise."""
    x = 0.8 * np.cumsum(0.15 * rng.standard_normal((n + lag, 3)).T, axis=1).T
    x += 0.3 * rng.standard_normal(x.shape)
    xs = np.array([x[t - lag : t] for t in range(lag, n + lag)])[:, :, 0]
    y = (
        np.sin(3 * xs[:, 0])
        + 0.7 * np.sign(xs[:, 1]) * np.sqrt(np.abs(xs[:, 1]))
        + 0.4 * np.tanh(4 * xs[:, 2])
        + 0.05 * rng.standard_normal(n)
    )
    return xs, y


def bench_kan_forecaster(
    seed: int = 89,
    *,
    n: int = 4000,
    steps: int = 1500,
    mlp_steps: int = 800,
    batch: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    lag, tr = 8, int(n * 0.8)
    xs_np, y = synth_nonlinear(n, lag, rng)
    xs_np = (xs_np - xs_np[:tr].mean(0)) / (xs_np[:tr].std(0) + 1e-9)
    xs = torch.tensor(xs_np, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)

    basis = 7
    w1 = torch.nn.Parameter(0.2 * torch.randn(lag, 16, basis))
    w2 = torch.nn.Parameter(0.2 * torch.randn(16, 1, basis))
    ks = torch.arange(1, basis // 2 + 1, dtype=torch.float32)

    def kan_edges(xb, w_):
        feats = [xb.unsqueeze(-1)]
        feats.append(torch.sin(xb.unsqueeze(-1) * ks))
        feats.append(torch.cos(xb.unsqueeze(-1) * ks))
        phi = torch.cat(feats, -1)[:, :, :basis]  # (B,in,basis)
        return torch.einsum("bik,iok->bo", phi, w_)

    def kan(xb):
        h = torch.tanh(kan_edges(xb, w1))
        return kan_edges(h, w2).squeeze(-1)

    opt = torch.optim.Adam([w1, w2], lr=5e-3)
    for _ in range(steps):
        idx = torch.randint(0, tr, (batch,))
        loss = torch.mean((kan(xs[idx]) - ys[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    mlp = torch.nn.Sequential(
        torch.nn.Linear(lag, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    mopt = torch.optim.Adam(mlp.parameters(), lr=3e-3)
    for _ in range(mlp_steps):
        idx = torch.randint(0, tr, (batch,))
        loss = torch.mean((mlp(xs[idx]).squeeze(-1) - ys[idx]) ** 2)
        mopt.zero_grad()
        loss.backward()
        mopt.step()

    with torch.no_grad():
        mae_kan = float(torch.mean(torch.abs(kan(xs[tr:]) - ys[tr:])))
        mae_mlp = float(torch.mean(torch.abs(mlp(xs[tr:]).squeeze(-1) - ys[tr:])))
    # ridge floor
    xr = np.hstack([xs_np, np.ones((n, 1))])
    wr = np.asarray(
        np.linalg.solve(xr[:tr].T @ xr[:tr] + 1e-3 * np.eye(xr.shape[1]), xr[:tr].T @ y[:tr])
    )
    mae_r = float(np.mean(np.abs(xr[tr:] @ wr - y[tr:])))
    return {
        "synthetic_kan_mae": mae_kan,
        "synthetic_kan_mlp_mae": mae_mlp,
        "synthetic_kan_ridge_mae": mae_r,
        "synthetic_kan_margin_vs_mlp": mae_mlp - mae_kan,
        "synthetic_kan_margin_vs_ridge": mae_r - mae_kan,
        "synthetic_kan_params": float(w1.numel() + w2.numel()),
        "synthetic_kan_mlp_params": float(sum(p.numel() for p in mlp.parameters())),
        "synthetic_torch_available": 1.0,
    }
