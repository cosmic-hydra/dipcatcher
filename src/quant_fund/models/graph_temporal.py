"""Spatio-temporal graph forecaster (torch).

A fixed dependency graph over the panel couples a graph convolution
(neighbor mixing) with a GRU per node — the DCRNN/ST-GCN recipe in
miniature. Requires the ``nn`` extra; SYNTHETIC panel only.

Bench: a panel where each node's next value depends on its own history
AND neighbors' current state (directed lead-lag graph); joint model vs
per-node independent GRUs and AR ridge.
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
        raise ImportError("graph_temporal torch path requires the `nn` extra (make sync)") from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_graph_panel(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """6-node directed lead graph: node j's driver feeds node j+1.

    Returns panel (n,6) + adjacency (6,6).
    """
    adj = np.zeros((6, 6))
    for j in range(5):
        adj[j, j + 1] = 1.0  # j -> j+1
    adj[5, 0] = 0.5
    x = np.zeros((n, 6))
    for t in range(1, n):
        drive = np.sign(x[t - 1]) * x[t - 1] ** 2  # nonlinear neighbor coupling
        x[t] = 0.3 * x[t - 1] + 0.9 * np.tanh(0.5 * adj.T @ drive) + 0.06 * rng.standard_normal(6)
    return x, adj


def bench_graph_temporal(
    seed: int = 107,
    *,
    t_len: int = 3000,
    win: int = 20,
    joint_steps: int = 600,
    ind_steps: int = 200,
    batch: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    T = t_len
    x, adj = synth_graph_panel(T, rng)
    tr_t = int(T * 0.85)
    adj_t = torch.tensor(adj, dtype=torch.float32)
    xs = torch.tensor(np.array([x[t - win : t] for t in range(win, T)]), dtype=torch.float32)
    ys = torch.tensor(x[win:], dtype=torch.float32)
    tr = len(xs) - (T - tr_t)

    # joint: graph-mix -> GRU -> head
    gmix = torch.nn.Linear(6, 6, bias=False)
    gmix.weight.data = adj_t.clone() + torch.eye(6)
    gru = torch.nn.GRU(6, 32, batch_first=True)
    head = torch.nn.Linear(32, 6)
    mods = torch.nn.ModuleList([gmix, gru, head])

    def joint(xb):  # (B, win, 6)
        h = torch.tanh(xb @ gmix.weight.T)
        _, hn = gru(h)
        return head(hn[-1])

    opt = torch.optim.Adam(mods.parameters(), lr=5e-3)
    for _ in range(joint_steps):
        idx = torch.randint(0, tr, (batch,))
        loss = torch.mean((joint(xs[idx]) - ys[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        mae_joint = float(torch.mean(torch.abs(joint(xs[tr:]) - ys[tr:])))

    # per-node independent GRUs
    maes = []
    for c in range(6):
        gru_i = torch.nn.GRU(1, 16, batch_first=True)
        head_i = torch.nn.Linear(16, 1)
        mods_i = torch.nn.ModuleList([gru_i, head_i])
        opt_i = torch.optim.Adam(mods_i.parameters(), lr=3e-3)
        xc = xs[:, :, c : c + 1]
        yc = ys[:, c]
        for _ in range(ind_steps):
            idx = torch.randint(0, tr, (min(batch, tr),))
            _, hn = gru_i(xc[idx])
            loss = torch.mean((head_i(hn[-1]).squeeze(-1) - yc[idx]) ** 2)
            opt_i.zero_grad()
            loss.backward()
            opt_i.step()
        with torch.no_grad():
            _, hn = gru_i(xc[tr:])
            maes.append(float(torch.mean(torch.abs(head_i(hn[-1]).squeeze(-1) - yc[tr:]))))
    mae_ind = float(np.mean(maes))

    errs = []
    for c in range(6):
        c0 = x[:, c]
        xa = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(2, win + tr)])
        w = np.asarray(np.linalg.solve(xa.T @ xa + 1e-4 * np.eye(3), xa.T @ c0[2 : win + tr]))
        pred = np.array([[c0[t - 1], c0[t - 2], 1.0] for t in range(win + tr, T)]) @ w
        errs.append(float(np.mean(np.abs(pred - c0[win + tr :]))))
    ar = float(np.mean(errs))
    return {
        "synthetic_graphtemporal_mae": mae_joint,
        "synthetic_graphtemporal_ind_mae": mae_ind,
        "synthetic_graphtemporal_ar_mae": ar,
        "synthetic_graphtemporal_margin_vs_ind": mae_ind - mae_joint,
        "synthetic_graphtemporal_margin_vs_ar": ar - mae_joint,
        "synthetic_torch_available": 1.0,
    }
