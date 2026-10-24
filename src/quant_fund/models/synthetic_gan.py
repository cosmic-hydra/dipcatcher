"""Time-series GAN (Exec-Summary synthetic-data item). Vanilla minimax (SYNTHETIC)
GAN on return windows: 1-hidden-layer generator and discriminator with
manual backprop; label smoothing + instance noise for stability.

Synthetic bench: post-training discriminator AUC near 0.5, stylized-fact
gaps (vol clustering, kurtosis) vs source series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


def _tanh(x: FloatArray) -> FloatArray:
    return np.tanh(x)


@dataclass
class Generator:
    z_dim: int
    out_dim: int
    hid: int = 48

    def __post_init__(self) -> None:
        rng = np.random.default_rng(0)
        self.w1 = 0.1 * rng.standard_normal((self.hid, self.z_dim))
        self.b1 = np.zeros(self.hid)
        self.w2 = 0.1 * rng.standard_normal((self.out_dim, self.hid))
        self.b2 = np.zeros(self.out_dim)

    def forward(self, z: FloatArray) -> FloatArray:
        return _tanh(self.w1 @ z + self.b1) @ self.w2.T + self.b2

    def step(self, z: FloatArray, d_err: FloatArray, lr: float) -> None:
        h = _tanh(self.w1 @ z + self.b1)
        g_out = d_err  # dL/d(out) from discriminator
        self.w2 -= lr * np.outer(g_out, h)
        self.b2 -= lr * g_out
        dh = (1 - h**2) * (g_out @ self.w2)
        self.w1 -= lr * np.outer(dh, z)
        self.b1 -= lr * dh


@dataclass
class Discriminator:
    in_dim: int
    hid: int = 32

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1)
        self.w1 = 0.1 * rng.standard_normal((self.hid, self.in_dim))
        self.b1 = np.zeros(self.hid)
        self.w2 = 0.1 * rng.standard_normal(self.hid)
        self.b2 = 0.0

    def forward(self, x: FloatArray) -> tuple[float, FloatArray]:
        h = _tanh(self.w1 @ x + self.b1)
        logit = float(h @ self.w2 + self.b2)
        return logit, h

    def step(self, x: FloatArray, label: float, lr: float) -> tuple[float, FloatArray]:
        logit, h = self.forward(x)
        p = 1.0 / (1.0 + np.exp(-logit))
        err = p - label
        gw2 = err * h
        gh = (1 - h**2) * (err * self.w2)
        self.w2 -= lr * gw2
        self.b2 -= lr * err
        self.w1 -= lr * np.outer(gh, x)
        self.b1 -= lr * gh
        # backprop to x (for generator): d logit / dx
        dlogit_dx = self.w1.T @ ((1 - h**2) * self.w2)
        return float(-(label * np.log(p + 1e-9) + (1 - label) * np.log(1 - p + 1e-9))), dlogit_dx

    def prob(self, x: FloatArray) -> float:
        logit, _ = self.forward(x)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def step_wasserstein(
        self, x: FloatArray, side: float, lr: float, clip: float = 0.05
    ) -> tuple[float, FloatArray]:
        """Wasserstein critic step: score(x) pushed up (real) or down (fake)."""
        logit, h = self.forward(x)
        err = -side  # dL/dlogit where L = -side * score
        gw2 = err * h
        gh = (1 - h**2) * (err * self.w2)
        self.w2 -= lr * gw2
        self.b2 -= lr * err
        self.w1 -= lr * np.outer(gh, x)
        self.b1 -= lr * gh
        self.w1 = np.clip(self.w1, -clip, clip)
        self.w2 = np.clip(self.w2, -clip, clip)
        self.b1 = np.clip(self.b1, -clip, clip)
        dlogit_dx = self.w1.T @ ((1 - h**2) * self.w2)
        return float(-side * logit), dlogit_dx


def garch_windows(n: int, w: int, rng: np.random.Generator) -> FloatArray:
    # r[0] is the lag for the first variance update and the first window.
    # np.empty leaves that element as heap garbage, which is often zero on
    # Linux and non-finite on Windows, so the critic loss diverges there.
    r = np.zeros(n + w)
    v = np.empty(n + w)
    v[0] = 1e-4
    for t in range(1, n + w):
        v[t] = 0.05e-4 + 0.9 * v[t - 1] + 0.09 * r[t - 1] ** 2
        r[t] = np.sqrt(v[t]) * rng.standard_normal()
    return np.array([r[i : i + w] for i in range(n)])


def vol_cluster(x: FloatArray) -> float:
    return float(np.corrcoef(np.abs(x[:-1]), np.abs(x[1:]))[0, 1])


def kurt(x: FloatArray) -> float:
    m = x - x.mean()
    return float(np.mean(m**4) / (np.mean(m**2) ** 2 + 1e-12) - 3.0)


def bench_synthetic_gan(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    w = 24
    wins = garch_windows(400, w, rng)
    scale = wins.std()
    wins_n = wins / scale
    G = Generator(8, w)
    D = Discriminator(w)
    d_loss = g_loss = 0.0
    for _ in range(1200):
        for _ in range(5):  # n_critic
            i = int(rng.integers(len(wins_n)))
            real = wins_n[i]
            z = rng.standard_normal(8)
            fake = G.forward(z)
            dl, _ = D.step_wasserstein(real, 1.0, 0.005)
            dl2, _ = D.step_wasserstein(fake, -1.0, 0.005)
            d_loss = dl + dl2
        z = rng.standard_normal(8)
        fake = G.forward(z)
        _, dx = D.step_wasserstein(fake, 0.0, 0.0)  # grad only
        G.step(z, -dx, 0.01)
        logit, _ = D.forward(fake)
        g_loss = -float(logit)
    # eval: discriminator AUC on fresh fakes
    ps_real = np.array([D.prob(wins_n[i]) for i in range(200)])
    ps_fake = np.array([D.prob(G.forward(rng.standard_normal(8))) for _ in range(200)])
    y = np.concatenate([np.ones(200), np.zeros(200)])
    p = np.concatenate([ps_real, ps_fake])
    order = np.argsort(p)
    ranks = np.empty(len(y))
    ranks[order] = np.arange(1, len(y) + 1)
    auc = float((ranks[y == 1].sum() - 200 * 201 / 2) / (200 * 200))
    gen = np.concatenate([G.forward(rng.standard_normal(8)) for _ in range(30)]) * scale
    real = wins.reshape(-1)
    return {
        "synthetic_gan_d_loss": float(d_loss),
        "synthetic_gan_g_loss": float(g_loss),
        "synthetic_gan_d_auc": auc,
        "synthetic_gan_volcluster_gap": abs(vol_cluster(gen) - vol_cluster(real)),
        "synthetic_gan_kurtosis_gap": abs(kurt(gen) - kurt(real)),
        "synthetic_gan_std_ratio": float(gen.std() / real.std()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_synthetic_gan(), indent=1))
