"""SPDNet-style covariance-manifold classifier.

Huang & Van Gool 2017: symmetric positive-definite matrices live on a curved
manifold — naive Euclidean vectorization loses the geometry. Layers:
BiMap (congruent transform W^T S W) + ReEig (eigenvalue floor/relu). Bench:
classify return-regime SPD covariances vs a Euclidean MLP on the flattened
matrix. SYNTHETIC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SEED = 20261015
_D_COV = 6


def _torch() -> Any:
    import torch

    _ = torch.nn.Linear  # torch + nn extra required
    return torch


def synth_spd_data(n: int, rng: np.random.Generator) -> tuple[FloatArray, NDArray[np.int64]]:
    """Regime labels from Wishart-like covariances with distinct spectra."""
    d = _D_COV
    covs = np.zeros((n, d, d))
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        regime = i % 2
        base = rng.normal(0, 1, (d + 4, d))
        cov = base.T @ base / (d + 4) + 0.3 * np.eye(d)
        if regime == 1:
            u, _ = np.linalg.qr(rng.normal(0, 1, (d, d)))
            cov = u @ np.diag(np.diag(cov) * rng.uniform(1.5, 4.0, d)) @ u.T
        covs[i] = cov
        y[i] = regime
    return covs.astype(np.float64), y


def _bimap(s: Any, w: Any) -> Any:
    return w.T @ s @ w


def _reeig(s: Any, eps: float, torch: Any) -> Any:
    evals, evecs = torch.linalg.eigh(s)
    evals = torch.nn.functional.relu(evals - eps) + eps
    return evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)


def bench_spd_net(
    seed: int = 21,
    n_train: int = 800,
    n_test: int = 200,
    iters: int = 500,
    eps: float = 1e-3,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed + _SEED)
    ctr, ytr = synth_spd_data(n_train, rng)
    cte, yte = synth_spd_data(n_test, rng)

    w1 = torch.nn.Parameter(torch.eye(_D_COV) + 0.05 * torch.randn(_D_COV, _D_COV))
    w2 = torch.nn.Parameter(torch.eye(_D_COV) + 0.05 * torch.randn(_D_COV, _D_COV))
    clf = torch.nn.Linear(_D_COV * _D_COV, 2)
    params = [w1, w2] + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)

    st = torch.tensor(ctr, dtype=torch.float32)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        h = _reeig(_bimap(st, w1), eps, torch)
        h = _reeig(_bimap(h, w2), eps, torch)
        logits = clf(h.reshape(n_train, -1))
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Euclidean baseline: flatten raw cov into MLP
    mlp = torch.nn.Sequential(
        torch.nn.Linear(_D_COV * _D_COV, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    opt_m = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(mlp(st.reshape(n_train, -1)), yt)
        opt_m.zero_grad()
        loss.backward()
        opt_m.step()

    with torch.no_grad():
        se = torch.tensor(cte, dtype=torch.float32)
        ye = torch.tensor(yte)
        h = _reeig(_bimap(se, w1), eps, torch)
        h = _reeig(_bimap(h, w2), eps, torch)
        acc_spd = float((clf(h.reshape(n_test, -1)).argmax(-1) == ye).float().mean())
        acc_mlp = float((mlp(se.reshape(n_test, -1)).argmax(-1) == ye).float().mean())
    return {
        "synthetic_spd_acc": acc_spd,
        "synthetic_spd_mlp_acc": acc_mlp,
        "synthetic_spd_acc_gain": acc_spd - acc_mlp,
        "synthetic_torch_available": 1.0,
    }
