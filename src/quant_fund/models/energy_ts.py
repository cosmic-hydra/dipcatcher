"""Energy-based model for TS anomaly scoring (SYNTHETIC).

Trains an energy function E(x) via contrastive divergence (Langevin
negative sampling); normal windows get low energy, corrupted/anomalous
windows high energy. No generative decoder needed.

Bench: smooth sine windows vs anomalies (spikes, level shifts, frozen
signals); metric = ROC-AUC of E vs a reconstruction-error MLP
autoencoder baseline.
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
        raise ImportError("energy_ts requires the `nn` extra (make sync)") from exc


def synth_normal_anom(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Half smooth sinusoid (label 0), half corrupted (label 1)."""
    x = np.zeros((n, win))
    y = np.zeros(n)
    for i in range(n):
        base = np.sin(np.linspace(0, 2 * np.pi, win) + rng.uniform(0, 1))
        if i % 2 == 0:
            x[i] = base + 0.1 * rng.standard_normal(win)
        else:
            y[i] = 1.0
            kind = rng.integers(3)
            if kind == 0:
                x[i] = base + 0.1 * rng.standard_normal(win)
                x[i, rng.integers(0, win)] += 1.1  # subtle spike
            elif kind == 1:
                x[i] = base + 0.35  # mild level shift
            else:
                x[i] = np.convolve(base, np.ones(5) / 5, mode="same") + 0.05 * rng.standard_normal(
                    win
                )  # over-smoothed
    return x.astype(np.float64), y.astype(np.float64)


def _auc(scores: FloatArray, y: FloatArray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores)) + 1
    pos = y == 1
    n_pos = pos.sum()
    n_neg = len(y) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bench_energy_ts(
    seed: int = 20261231,
    n: int = 320,
    win: int = 24,
    iters: int = 700,
    langevin_steps: int = 15,
) -> dict[str, float]:
    """EBM energy ranking vs AE recon-error baseline (AUC, SYNTH)."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, y = synth_normal_anom(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    pos_mask = y == 0

    ebm = torch.nn.Sequential(
        torch.nn.Linear(win, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    params = torch.nn.ModuleList([ebm])
    opt = torch.optim.Adam(params.parameters(), lr=1e-3)
    tr = np.where(pos_mask & (np.arange(n) < 3 * n // 4))[0]
    for _i in range(iters):
        b_idx = tr[rng.integers(0, len(tr), 64)]
        x_pos = xb[b_idx]
        x_neg = x_pos + 0.1 * torch.randn_like(x_pos)
        for _l in range(langevin_steps):
            x_neg = x_neg.requires_grad_(True)
            e_neg = ebm(x_neg).sum()
            grad = torch.autograd.grad(e_neg, x_neg)[0]
            x_neg = (
                x_neg - 0.02 * torch.clamp(grad, -2, 2) + 0.02 * torch.randn_like(x_neg)
            ).detach()
        e_pos = ebm(x_pos).mean()
        e_neg2 = ebm(x_neg).mean()
        loss = e_pos - e_neg2 + 0.01 * (e_pos**2 + e_neg2**2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        e_scores = ebm(xb).squeeze(1).numpy()  # anomaly = HIGH energy
    auc_ebm = _auc(e_scores, y)

    ae = torch.nn.Sequential(torch.nn.Linear(win, 16), torch.nn.ReLU(), torch.nn.Linear(16, win))
    pa = torch.nn.ModuleList([ae])
    opta = torch.optim.Adam(pa.parameters(), lr=2e-3)
    for _i in range(iters):
        b_idx = tr[rng.integers(0, len(tr), 64)]
        loss = torch.mean((ae(xb[b_idx]) - xb[b_idx]) ** 2)
        opta.zero_grad()
        loss.backward()
        opta.step()
    with torch.no_grad():
        ae_err = torch.mean((ae(xb) - xb) ** 2, 1).numpy()
    auc_ae = _auc(ae_err.astype(np.float64), y)
    return {
        "synthetic_energy_auc": auc_ebm,
        "synthetic_energy_ae_auc": auc_ae,
        "synthetic_energy_margin_vs_ae": auc_ebm - auc_ae,
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_energy_ts()))
