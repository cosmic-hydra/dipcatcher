"""POMCP (Silver & Veness 2010) — Monte-Carlo tree search over (SYNTHETIC)
histories with an unweighted particle belief and UCB1 action
selection. Each simulation draws a state from B(h), simulates via the
generative model, and backs up returns along the history tree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.qmdp import POMDP, tiger_pomdp

FloatArray = NDArray[np.float64]


class POMCP:
    """POMCP planner: particle belief + UCT over (history, action)
    nodes."""

    def __init__(self, p: POMDP, n_particles: int = 256, c: float = 8.0, depth: int = 10):
        self.p = p
        self.c = c
        self.depth = depth
        self.particles = np.zeros(n_particles, dtype=np.int64)
        self.n_init = n_particles
        self._reset(rng=np.random.default_rng(0))

    def _reset(self, rng: np.random.Generator) -> None:
        self.particles = rng.choice(self.p.n_s, self.n_init, p=self.p.s0_dist)
        self.tree: dict[tuple, list[float]] = {}
        self.n_counts: dict[tuple, list[int]] = {}

    def _gen(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, int, float]:
        sp = int(rng.choice(self.p.n_s, p=self.p.t[a, s]))
        o = int(rng.choice(self.p.n_o, p=self.p.o[a, sp]))
        return sp, o, float(self.p.r[a, s])

    def _simulate(self, s: int, h: tuple, depth: int, rng: np.random.Generator) -> float:
        if depth <= 0:
            return 0.0
        if h not in self.tree:
            # expand: rollout a random action to estimate the leaf
            self.tree[h] = [0.0] * self.p.n_a
            self.n_counts[h] = [0] * self.p.n_a
            a0 = int(rng.integers(self.p.n_a))
            sp, _o, r = self._gen(s, a0, rng)
            ret = r + self.p.gamma * self._rollout(sp, depth - 1, rng)
            self.n_counts[h][a0] = 1
            self.tree[h][a0] = ret
            return ret
        # UCB1 select
        nh = sum(self.n_counts[h])
        ucb = [
            self.tree[h][a] + self.c * np.sqrt(np.log(nh + 1) / (self.n_counts[h][a] + 1e-9))
            for a in range(self.p.n_a)
        ]
        a = int(np.argmax(ucb))
        sp, o, r = self._gen(s, a, rng)
        ret = r + self.p.gamma * self._simulate(sp, h + ((a, o),), depth - 1, rng)
        self.n_counts[h][a] += 1
        self.tree[h][a] += (ret - self.tree[h][a]) / self.n_counts[h][a]
        return float(ret)

    def _rollout(self, s: int, depth: int, rng: np.random.Generator) -> float:
        # heuristic rollout: mostly listen (action 0) when available —
        # standard POMCP domain prior; keeps leaf estimates sane on
        # heavy-tailed reward problems like Tiger
        ret, disc = 0.0, 1.0
        listen = 0 if self.p.n_a > 2 else -1
        for _ in range(depth):
            if listen >= 0 and rng.random() < 0.8:
                a = listen
            else:
                a = int(rng.integers(self.p.n_a))
            s, _o, r = self._gen(s, a, rng)
            ret += disc * r
            disc *= self.p.gamma
        return ret

    def plan(self, rng: np.random.Generator, n_sims: int = 200) -> int:
        h = ()
        for _ in range(n_sims):
            s = int(self.particles[rng.integers(len(self.particles))])
            self._simulate(s, h, self.depth, rng)
        if h not in self.tree:
            return int(rng.integers(self.p.n_a))
        return int(np.argmax(self.tree[h]))

    def step(self, a: int, o: int, rng: np.random.Generator) -> None:
        """Update the particle belief after real (a, o): resample
        particles consistent with the observation."""
        new: list[int] = []
        tries = 0
        while len(new) < self.n_init and tries < self.n_init * 30:
            tries += 1
            s = int(self.particles[rng.integers(len(self.particles))])
            sp, so, _r = self._gen(s, a, rng)
            if so == o:
                new.append(sp)
        if len(new) < 4:  # particle depletion → reset
            self._reset(rng)
            return
        self.particles = np.array(new + list(rng.choice(new, self.n_init - len(new))))


def _episode(
    p: POMDP, agent: POMCP | None, rng: np.random.Generator, horizon: int, n_sims: int
) -> float:
    s = int(rng.choice(p.n_s, p=p.s0_dist))
    ret, disc = 0.0, 1.0
    for _ in range(horizon):
        if agent is None:
            a = int(rng.integers(p.n_a))
        else:
            a = agent.plan(rng, n_sims=n_sims)
        sp = int(rng.choice(p.n_s, p=p.t[a, s]))
        o = int(rng.choice(p.n_o, p=p.o[a, sp]))
        ret += disc * p.r[a, s]
        disc *= p.gamma
        if agent is not None:
            agent.step(a, o, rng)
        s = sp
    return ret


def bench_pomcp(
    seed: int = 20261231,
    *,
    n_episodes: int = 12,
    n_sims: int = 300,
    n_particles: int = 200,
) -> dict[str, float]:
    """SYNTHETIC: POMCP on Tiger — listens first at the uniform
    belief, and its UCT policy beats the random-action baseline by a
    wide margin (Tiger's ±100 reward scale is intentionally brutal)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p = tiger_pomdp()
    # random baseline
    rand_ret = float(np.mean([_episode(p, None, rng, 12, 0) for _ in range(30)]))
    out["synthetic_pomcp_random_return"] = rand_ret
    agent = POMCP(p, n_particles=n_particles, c=4.0, depth=10)
    rets = []
    for _ in range(n_episodes):
        agent._reset(rng)
        rets.append(_episode(p, agent, rng, 12, n_sims))
    out["synthetic_pomcp_return"] = float(np.mean(rets))
    out["synthetic_pomcp_beats_random"] = float(np.mean(rets) > rand_ret + 20.0)
    agent._reset(rng)
    out["synthetic_pomcp_first_listens"] = float(agent.plan(rng, n_sims=150) == 0)
    return out


if __name__ == "__main__":
    print(bench_pomcp())
