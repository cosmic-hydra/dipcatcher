"""Chart-pattern CNN: candlestick-image move classifier (torch).

OHLC windows are rasterized into a 2-D image (time x price-bucket, one
channel per O/H/L/C) and a small CNN classifies the next-bar direction —
the image-based technical-analysis approach. Requires the ``nn`` extra;
SYNTHETIC charts only.

Bench: synthetic paths where the label is a nonlinear spatial pattern
(engulfing/volatility-cone structure) — CNN should beat a logistic on
flattened OHLC features.
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
        raise ImportError("cnn_alpha torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_charts(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """(n, win, 4) OHLC + label = nonlinear engulfing/cone pattern."""
    x = np.zeros((n, win, 4))
    y = np.zeros(n)
    for i in range(n):
        drift = rng.choice([-0.02, 0.0, 0.02])
        vol = 0.05 + 0.15 * rng.random()
        c = 100 + np.cumsum(drift + vol * rng.standard_normal(win))
        o = np.concatenate([[c[0]], c[:-1]]) + 0.3 * vol * rng.standard_normal(win)
        h = np.maximum(o, c) + abs(0.5 * vol * rng.standard_normal(win))
        lo = np.minimum(o, c) - abs(0.5 * vol * rng.standard_normal(win))
        x[i] = np.stack([o, h, lo, c], 1)
        # label: U-shaped recovery — min in middle third, ends recover
        # above the early-quarter level: spatial composition a
        # flattened linear model can't express
        i_min = int(np.argmin(c))
        u_shape = float(
            win // 3 <= i_min <= 2 * win // 3
            and c[-1] > c[i_min] + 0.5 * (c[0] - c[i_min])
            and c[-1] > np.mean(c[: win // 4])
        )
        y[i] = float(u_shape + 0.25 * rng.standard_normal() > 0.5)
    return x, y


def _rasterize(x: FloatArray, rows: int = 24) -> FloatArray:
    """(n, win, 4) OHLC -> (n, 4, rows, win) binary-ish image."""
    n, win, _ = x.shape
    out = np.zeros((n, 4, rows, win))
    for i in range(n):
        lo, hi = x[i, :, 2].min(), x[i, :, 1].max()
        span = max(hi - lo, 1e-9)
        for t in range(win):
            for c, v in enumerate(x[i, t]):
                r = int(np.clip((v - lo) / span * (rows - 1), 0, rows - 1))
                out[i, c, rows - 1 - r, t] = 1.0
    return out


def bench_cnn_alpha(
    seed: int = 101,
    *,
    n: int = 3000,
    win: int = 24,
    steps: int = 300,
    batch: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    tr = int(n * 0.8)
    x, y = synth_charts(n, win, rng)
    imgs = torch.tensor(_rasterize(x), dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)

    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 24, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((3, 4)),
        torch.nn.Flatten(),
        torch.nn.Linear(24 * 12, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.randint(0, tr, (batch,))
        loss = bce(net(imgs[idx]).squeeze(-1), ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((net(imgs[tr:]).squeeze(-1) > 0) == (ys[tr:] > 0.5)).float().mean())

    # logistic on flattened OHLC
    xf = x.reshape(n, -1)
    xf = np.hstack([xf, np.ones((n, 1))])
    lam = 1e-2 * np.eye(xf.shape[1])
    w = np.asarray(np.linalg.solve(xf[:tr].T @ xf[:tr] + lam, xf[:tr].T @ y[:tr]))
    acc_l = float(np.mean((xf[tr:] @ w > 0.5) == (y[tr:] > 0.5)))
    return {
        "synthetic_cnnalpha_acc": acc,
        "synthetic_cnnalpha_logistic_acc": acc_l,
        "synthetic_cnnalpha_margin": acc - acc_l,
        "synthetic_torch_available": 1.0,
    }
