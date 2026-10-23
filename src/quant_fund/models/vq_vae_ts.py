"""VQ-VAE for time series: discrete latent codebook (SYNTHETIC).

van den Oord et al. 2017: encoder outputs are snapped to a learned
codebook of prototypes (commitment + codebook losses); the discrete
latents form a compressed vocabulary of market shapes.

Bench: synthetic windows drawn from a mixture of 4 shape prototypes;
the codebook should recover ~4 active codes, decode with lower error
than a k-means baseline, and beat a PCA-4 recon.
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
        raise ImportError("vq_vae_ts requires the `nn` extra (make sync)") from exc


def synth_prototype_mix(
    n: int, win: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """(n, win) windows from 4 latent prototypes + class label."""
    protos = np.stack(
        [
            np.sin(np.linspace(0, np.pi, win)),
            -np.sin(np.linspace(0, np.pi, win)),
            np.sin(np.linspace(0, 3 * np.pi, win)),
            np.linspace(-1, 1, win),
        ]
    )
    x = np.zeros((n, win))
    lab = np.zeros(n)
    for i in range(n):
        k = int(rng.integers(4))
        lab[i] = k
        x[i] = protos[k] + 0.2 * rng.standard_normal(win)
    return x.astype(np.float64), lab.astype(np.float64)


def bench_vq_vae_ts(
    seed: int = 20261231,
    n: int = 320,
    win: int = 32,
    K: int = 8,
    iters: int = 900,
) -> dict[str, float]:
    """VQ-VAE codebook recovery + recon vs PCA-4."""
    torch = _torch()
    torch.manual_seed(int(seed))  # audit sweep: seeded determinism
    rng = np.random.default_rng(seed)
    xs, lab = synth_prototype_mix(n, win, rng)
    xb = torch.tensor(xs, dtype=torch.float32)
    te = slice(3 * n // 4, n)
    tr = slice(0, 3 * n // 4)

    enc = torch.nn.Sequential(torch.nn.Linear(win, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16))
    emb = torch.nn.Embedding(K, 16)
    dec = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, win))
    params = torch.nn.ModuleList([enc, emb, dec])
    opt = torch.optim.Adam(params.parameters(), lr=2e-3)
    for _i in range(iters):
        z = enc(xb[tr])
        # cosine-normalized codebook matching — stops the classic
        # VQ collapse where a few codes dominate
        zn = torch.nn.functional.normalize(z, dim=-1)
        en = torch.nn.functional.normalize(emb.weight, dim=-1)
        dist = 1 - zn @ en.T
        q = dist.argmin(-1)
        zq = emb(q)
        recon = dec(zq)
        loss = (
            torch.mean((recon - xb[tr]) ** 2)
            + torch.mean((zq.detach() - z) ** 2)  # commitment
            + 0.1 * torch.mean((zq - z.detach()) ** 2)  # codebook
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if _i % 200 == 199:
            # revive dead codes from random encoder outputs
            with torch.no_grad():
                used = torch.bincount(q, minlength=K) > 0
                dead = (~used).nonzero().squeeze(1)
                if dead.numel() > 0:
                    pick = torch.randint(0, z.shape[0], (dead.numel(),))
                    emb.weight.data[dead] = z[pick]
    with torch.no_grad():
        z = enc(xb)
        zn = torch.nn.functional.normalize(z, dim=-1)
        en = torch.nn.functional.normalize(emb.weight, dim=-1)
        q = (1 - zn @ en.T).argmin(-1)
        rec = dec(emb(q))
        vq_mse = float(torch.mean((rec[te] - xb[te]) ** 2))
        codes = q[te].numpy()
        n_active = float(len(np.unique(codes)))
        # mutual info between code index and true prototype class
        mi = 0.0
        for k in range(4):
            for c in np.unique(codes):
                p_kc = np.mean((lab[te] == k) & (codes == c))
                p_k = np.mean(lab[te] == k)
                p_c = np.mean(codes == c)
                if p_kc > 1e-9:
                    mi += p_kc * np.log(p_kc / (p_k * p_c))

    # PCA-4 baseline recon
    X = xs[tr]
    U, Sv, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    proj = (xs[te] - X.mean(0)) @ Vt[:4].T @ Vt[:4] + X.mean(0)
    pca_mse = float(np.mean((proj - xs[te]) ** 2))
    return {
        "synthetic_vqvae_mse": vq_mse,
        "synthetic_vqvae_pca_mse": pca_mse,
        "synthetic_vqvae_margin_vs_pca": pca_mse - vq_mse,
        "synthetic_vqvae_active_codes": n_active,
        "synthetic_vqvae_code_mi": float(mi),
        "synthetic_vqvae_mi_ratio": float(mi / np.log(4)),
        "synthetic_vqvae_bits_per_window": float(np.log2(K)) / (win * 32),
        "synthetic_torch_available": 1.0,
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(bench_vq_vae_ts()))
