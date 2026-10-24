"""TSMixer: alternating time-mix and channel-mix MLPs (torch).

Residual blocks alternate an MLP applied along the time axis with an MLP
applied along the channel axis — the Google TSMixer recipe that matches
transformers on long-horizon panels at a fraction of the cost. Requires
the ``nn`` extra; SYNTHETIC panel only.

Bench: multivariate panel with cross-channel lead-lag + seasonal
structure; next-step MAE vs a channel-mixed flat MLP and AR ridge.
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
        raise ImportError("ts_mixer torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_mixer_panel(n: int, rng: np.random.Generator) -> FloatArray:
    """4 channels: c0 leads c1 by 3 steps, c2 seasonal, c3 cross-product."""
    x = np.zeros((n, 4))
    for t in range(1, n):
        x[t, 0] = 0.85 * x[t - 1, 0] + 0.3 * rng.standard_normal()
        x[t, 2] = np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal()
        x[t, 1] = (
            0.7 * x[t - 1, 1]
            + 0.6 * np.tanh(2 * x[t - 3 if t > 3 else t - 1, 0])
            + 0.1 * rng.standard_normal()
        )
        x[t, 3] = 0.5 * x[t, 1] * x[t, 2] + 0.2 * rng.standard_normal()
    return x


def bench_ts_mixer(
    seed: int = 91,
    *,
    t_len: int = 3000,
    win: int = 40,
    steps: int = 600,
    mlp_steps: int = 300,
    batch: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    T = t_len
    x = synth_mixer_panel(T, rng)
    xs = torch.tensor(np.array([x[t - win : t] for t in range(win, T)]), dtype=torch.float32)
    ys = torch.tensor(x[win:], dtype=torch.float32)
    tr = int(len(xs) * 0.85)

    # TSMixer block: time-mix (C x win->win) + channel-mix (win x C->C)
    tmix1 = torch.nn.Linear(win, win)
    tmix2 = torch.nn.Linear(win, win)
    cmix1 = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    cmix2 = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    head = torch.nn.Linear(win, 1)
    mods = torch.nn.ModuleList([tmix1, tmix2, cmix1, cmix2, head])

    def forward(xb):  # (B, win, C)
        h = xb + tmix1(xb.transpose(1, 2)).transpose(1, 2)  # time-mix residual
        h = h + cmix1(h)  # channel-mix residual
        h = h + tmix2(h.transpose(1, 2)).transpose(1, 2)
        h = h + cmix2(h)
        return head(h.transpose(1, 2)).squeeze(-1)  # (B, C)

    opt = torch.optim.Adam(mods.parameters(), lr=2e-3)
    for _ in range(steps):
        idx = torch.randint(0, tr, (batch,))
        loss = torch.mean(torch.abs(forward(xs[idx]) - ys[idx]))
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        mae = float(torch.mean(torch.abs(forward(xs[tr:]) - ys[tr:])))
    # flat MLP baseline
    mlp = torch.nn.Sequential(
        torch.nn.Linear(win * 4, 128), torch.nn.ReLU(), torch.nn.Linear(128, 4)
    )
    mopt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    xf = xs.reshape(len(xs), -1)
    for _ in range(mlp_steps):
        idx = torch.randint(0, tr, (batch,))
        loss = torch.mean(torch.abs(mlp(xf[idx]) - ys[idx]))
        mopt.zero_grad()
        loss.backward()
        mopt.step()
    with torch.no_grad():
        mae_m = float(torch.mean(torch.abs(mlp(xf[tr:]) - ys[tr:])))
    # AR ridge on each channel
    errs = []
    for c in range(4):
        c0 = x[:, c]
        xa = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(2, win + tr)])
        w = np.asarray(np.linalg.solve(xa.T @ xa + 1e-4 * np.eye(3), xa.T @ c0[2 : win + tr]))
        pred = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(win + tr, T)]) @ w
        errs.append(float(np.mean(np.abs(pred - c0[win + tr :]))))
    ar = float(np.mean(errs))
    return {
        "synthetic_tsmixer_mae": mae,
        "synthetic_tsmixer_flatmlp_mae": mae_m,
        "synthetic_tsmixer_ar_mae": ar,
        "synthetic_tsmixer_margin_vs_mlp": mae_m - mae,
        "synthetic_tsmixer_margin_vs_ar": ar - mae,
        "synthetic_torch_available": 1.0,
    }
