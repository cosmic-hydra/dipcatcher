"""Masked autoencoder pretraining for time series (torch).

Windows are split into patches; a random half are masked, the encoder
sees only visible patches, and a lightweight decoder reconstructs the
masked ones — the MAE recipe applied to 1-D series. The pretrained
encoder is then evaluated by a downstream probe. Requires the ``nn``
extra; SYNTHETIC series only.

Bench: linear probe on the MAE embedding vs a probe on the raw window
and vs a same-architecture encoder trained from scratch with labels —
pretraining should win the low-label regime.
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
        raise ImportError(
            "mask_autoencoder torch path requires the `nn` extra (make sync)"
        ) from exc


def torch_available() -> bool:
    try:
        _torch()
        return True
    except ImportError:
        return False


def synth_mae_windows(n: int, win: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Windows from 3 dynamical regimes (drift/mean-revert/oscillator);
    label = regime id."""
    x = np.zeros((n, win))
    reg = np.zeros(n, dtype=int)
    for i in range(n):
        r = int(rng.integers(0, 3))
        reg[i] = r
        eps = rng.standard_normal(win)
        if r == 0:
            s = np.cumsum(0.25 + 0.4 * eps)
        elif r == 1:
            s = np.zeros(win)
            for t in range(1, win):
                s[t] = -0.55 * s[t - 1] + 0.45 * eps[t]
        else:
            s = np.cumsum(0.5 * eps) + 1.2 * np.sin(
                2 * np.pi * np.arange(win) / 16 + rng.uniform(0, 6.28)
            )
        x[i] = s - s[0]
    return x, reg


def bench_mask_autoencoder(
    seed: int = 103,
    *,
    n: int = 5000,
    steps: int = 600,
    probe_steps: int = 300,
    batch: int = 256,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    torch = _torch()
    torch.manual_seed(seed)
    win, plen = 40, 8
    n_p = win // plen
    x, reg = synth_mae_windows(n, win, rng)
    tr, tr_lab = int(n * 0.8), 60  # low-label regime
    xs = torch.tensor(x, dtype=torch.float32)

    d = 24
    emb = torch.nn.Linear(plen, d)
    enc = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d, 2, 48, batch_first=True, dropout=0.0), 1
    )
    dec = torch.nn.Linear(d, plen)
    mods = torch.nn.ModuleList([emb, enc, dec])
    opt = torch.optim.Adam(mods.parameters(), lr=2e-3)

    for _ in range(steps):
        idx = torch.randint(0, tr, (batch,))
        p = xs[idx].reshape(-1, n_p, plen)
        mask = torch.rand(batch, n_p) < 0.5
        mask[:, 0] = False  # keep at least one visible
        vis = torch.where(mask.unsqueeze(-1), torch.zeros_like(p), p)
        h = enc(emb(vis))
        rec = dec(h)
        loss = torch.mean(((rec - p) ** 2) * mask.unsqueeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def embed(xb):
        p = xb.reshape(-1, n_p, plen)
        h = enc(emb(p))
        return h.mean(1)

    with torch.no_grad():
        e = embed(xs).numpy()

    # ridge probes
    def class_acc(feats, regv, trn):
        t = np.eye(3)[regv[:trn]]
        xa = np.hstack([feats, np.ones((len(feats), 1))])
        w = np.asarray(
            np.linalg.solve(xa[:trn].T @ xa[:trn] + 1e-2 * np.eye(xa.shape[1]), xa[:trn].T @ t)
        )
        return float(np.mean(np.argmax(xa[tr : tr + 800] @ w, 1) == regv[tr : tr + 800]))

    acc_mae = class_acc(e, reg, tr_lab)
    acc_raw = class_acc(x, reg, tr_lab)

    # scratch encoder trained ONLY on labels
    emb2 = torch.nn.Linear(plen, d)
    enc2 = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(d, 2, 48, batch_first=True, dropout=0.0), 1
    )
    head2 = torch.nn.Linear(d, 3)
    mods2 = torch.nn.ModuleList([emb2, enc2, head2])
    opt2 = torch.optim.Adam(mods2.parameters(), lr=2e-3)
    tgt = torch.tensor(reg, dtype=torch.long)
    for _ in range(probe_steps):
        idx = torch.randint(0, tr_lab, (min(64, tr_lab),))
        p = xs[idx].reshape(-1, n_p, plen)
        logits = head2(enc2(emb2(p)).mean(1))
        loss = torch.nn.functional.cross_entropy(logits, tgt[idx])
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    with torch.no_grad():
        logits = head2(enc2(emb2(xs[tr:].reshape(-1, n_p, plen))).mean(1))
        acc_scratch = float((logits.argmax(1) == tgt[tr:]).float().mean())

    return {
        "synthetic_mae_probe_acc": acc_mae,
        "synthetic_mae_raw_probe_acc": acc_raw,
        "synthetic_mae_scratch_acc": acc_scratch,
        "synthetic_mae_margin_vs_raw": acc_mae - acc_raw,
        "synthetic_mae_margin_vs_scratch": acc_mae - acc_scratch,
        "synthetic_torch_available": 1.0,
    }
