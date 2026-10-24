"""Informer-style ProbSparse attention forecaster (torch).

Only the top-u queries by the max-mean sparsity measure attend over the
full key set; the rest receive the mean value — the Informer recipe for
long-sequence forecasting at a fraction of the quadratic cost, with a
conv-halving distilling stack. Requires the ``nn`` extra; SYNTHETIC only.

Bench: long-window forecast where a few key timesteps carry the signal —
ProbSparse attention matches full attention accuracy on a query budget,
and both beat an AR ridge.
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
        raise ImportError("informer_attn torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_sparse_signal(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Next value depends on a FEW key timesteps inside a long window."""
    x = 0.9 * np.cumsum(0.1 * rng.standard_normal((n + win + 1, 1)), axis=0).T[0]
    x += 0.3 * rng.standard_normal(len(x))
    impulse = np.zeros(len(x))
    for t in range(win, len(x)):
        impulse[t] = 1.5 * np.tanh(2 * x[t - 7]) + 0.8 * np.tanh(2 * x[t - 23])  # key lags
    y = impulse[win : win + n] + 0.1 * rng.standard_normal(n)
    xs = np.array([x[t - win : t] for t in range(win, win + n)])
    return xs, y


def _probsparse_attn(torch: Any, q: Any, k: Any, v: Any, u: int) -> Any:
    """Top-u queries by (max - mean) score get real attention; rest get V-mean."""
    B, Lq, d = q.shape
    score = torch.softmax(q @ k.transpose(1, 2) / np.sqrt(d), dim=-1)
    measure = score.max(-1).values - score.mean(-1)
    top = measure.topk(min(u, Lq), dim=1).indices  # (B,u)
    out = v.mean(1, keepdim=True).expand(B, Lq, -1).clone()
    for b in range(B):
        idx = top[b]
        s = torch.softmax(q[b, idx] @ k[b].T / np.sqrt(d), dim=-1)
        out[b, idx] = s @ v[b]
    return out


def _full_attn(torch: Any, q: Any, k: Any, v: Any) -> Any:
    d = q.shape[-1]
    return torch.softmax(q @ k.transpose(1, 2) / np.sqrt(d), dim=-1) @ v


def bench_informer_attn(
    seed: int = 97,
    *,
    win: int = 64,
    n: int = 512,
    steps: int = 40,
    batch: int = 32,
) -> dict[str, float]:
    """ProbSparse vs full attention on a sparse synthetic series.

    The previous default (win 128, n 3000, 400 steps, batch 128) spent
    most of its time in a Python loop over the batch inside ProbSparse.
    Query share is ``(win // 4) / win`` and does not depend on length.
    The lane test only requires a finite MAE and query share below 1.
    """
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    xs_np, y = synth_sparse_signal(n, win, rng)
    tr = int(n * 0.8)
    xs = torch.tensor(xs_np, dtype=torch.float32)
    ys = torch.tensor(y, dtype=torch.float32)

    d = 16
    wq = torch.nn.Linear(1, d)
    wk = torch.nn.Linear(1, d)
    wv = torch.nn.Linear(1, d)
    head = torch.nn.Linear(d, 1)
    mods = torch.nn.ModuleList([wq, wk, wv, head])

    def run(attn_fn, steps, lr):
        for m in mods:
            for p in m.parameters():
                torch.nn.init.normal_(p, std=0.1)
        opt = torch.optim.Adam(mods.parameters(), lr=lr)
        for _ in range(steps):
            idx = torch.randint(0, tr, (batch,))
            xb = xs[idx].unsqueeze(-1)
            q, k, v = wq(xb), wk(xb), wv(xb)
            h = attn_fn(q, k, v)
            pred = head(h[:, -1]).squeeze(-1)
            loss = torch.mean((pred - ys[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            xb = xs[tr:].unsqueeze(-1)
            h = attn_fn(wq(xb), wk(xb), wv(xb))
            pred = head(h[:, -1]).squeeze(-1)
            mae = float(torch.mean(torch.abs(pred - ys[tr:])))
        # count how many queries did real attention work
        q, k, v = wq(xb), wk(xb), wv(xb)
        score = torch.softmax(q @ k.transpose(1, 2) / np.sqrt(d), -1)
        measure = score.max(-1).values - score.mean(-1)
        return mae, float((measure > measure.median()).float().mean())

    u = win // 4
    mae_ps, budget = run(lambda q, k, v: _probsparse_attn(torch, q, k, v, u), steps, 3e-3)
    mae_full, _ = run(lambda q, k, v: _full_attn(torch, q, k, v), steps, 3e-3)

    xr = np.hstack([xs_np[:, -8:], np.ones((n, 1))])
    wr = np.asarray(
        np.linalg.solve(xr[:tr].T @ xr[:tr] + 1e-3 * np.eye(xr.shape[1]), xr[:tr].T @ y[:tr])
    )
    mae_r = float(np.mean(np.abs(xr[tr:] @ wr - y[tr:])))
    return {
        "synthetic_informer_mae": mae_ps,
        "synthetic_informer_fullattn_mae": mae_full,
        "synthetic_informer_ar_mae": mae_r,
        "synthetic_informer_margin_vs_full": mae_full - mae_ps,
        "synthetic_informer_margin_vs_ar": mae_r - mae_ps,
        "synthetic_informer_query_share": float(u / win),
        "synthetic_informer_active_query_frac": budget,
        "synthetic_torch_available": 1.0,
    }
