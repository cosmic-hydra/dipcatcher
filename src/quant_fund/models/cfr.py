"""Counterfactual regret minimization — self-play equilibrium (SYNTHETIC)
computation for extensive-form games, instantiated on Kuhn poker
(canonical 3-card, 1-street game with a known Nash equilibrium).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_DEALS = [(i, j) for i in range(3) for j in range(3) if i != j]


def _kuhn_util(hist: str, c1: int, c2: int, player: int) -> float:
    """Terminal utility for `player` (the player to move) — Kuhn poker.

    Terminal histories: pp (check–check), bb (bet–call), pbb
    (check–bet–call), bp (bet–fold), pbp (check–bet–fold).
    """
    u1: float
    if hist in ("pp",):
        u1 = 1.0 if c1 > c2 else -1.0
    elif hist in ("bb", "pbb"):
        u1 = 2.0 if c1 > c2 else -2.0
    elif hist == "bp":
        u1 = 1.0  # p1 bet, p2 folded
    else:  # "pbp"
        u1 = -1.0  # p2 bet, p1 folded
    return u1 if player == 1 else -u1


def _is_terminal(h: str) -> bool:
    return h in ("pp", "bb", "pbb", "bp", "pbp")


def cfr_kuhn(iterations: int, rng: np.random.Generator) -> dict[str, FloatArray]:  # noqa: C901
    """Vanilla CFR on Kuhn poker.

    Info sets: player 1 sees {card}_{history}; player 2 likewise.
    Returns the average strategy dict keyed by info set.
    """
    regrets: dict[str, FloatArray] = {}
    strategy_sum: dict[str, FloatArray] = {}

    def infoset(player: int, card: int, hist: str) -> str:
        return f"p{player}c{card}:{hist}"

    def get_strategy(key: str, n_act: int) -> FloatArray:
        r = regrets.get(key, np.zeros(n_act))
        pos = np.maximum(r, 0.0)
        if pos.sum() <= 0:
            return np.full(n_act, 1.0 / n_act)
        out: FloatArray = np.asarray(pos / pos.sum())
        return out

    # cfr returns the utility of the player to move (hence the − on
    # recursion); reach probabilities p1/p2 track each player.
    def cfr(hist: str, c1: int, c2: int, p1: float, p2: float) -> float:
        if _is_terminal(hist):
            return _kuhn_util(hist, c1, c2, len(hist) % 2 + 1 if len(hist) else 1)
        player = 1 if len(hist) % 2 == 0 else 2
        card = c1 if player == 1 else c2
        key = infoset(player, card, hist)
        acts = ["p", "b"]
        strat = get_strategy(key, 2)
        util = np.empty(2)
        for a_i, a in enumerate(acts):
            util[a_i] = -cfr(
                hist + a,
                c1,
                c2,
                p1 * strat[a_i] if player == 1 else p1,
                p2 * strat[a_i] if player == 2 else p2,
            )
        node_util = float(strat @ util)
        r = regrets.setdefault(key, np.zeros(2))
        r += (p2 if player == 1 else p1) * (util - node_util)
        s = strategy_sum.setdefault(key, np.zeros(2))
        s += (p1 if player == 1 else p2) * strat
        return float(node_util)

    for _ in range(iterations):
        for c1, c2 in _DEALS:
            cfr("", c1, c2, 1.0, 1.0)
    avg = {k: v / max(v.sum(), 1e-12) for k, v in strategy_sum.items()}
    return avg


def kuhn_exploitability(avg: dict[str, FloatArray]) -> float:
    """Best-response exploitability: u1(BR1, σ2) + u2(σ1, BR2).

    The best response is a pure policy over the responder's OWN info
    sets (own card + public history) — enumerated exhaustively over
    Kuhn's 2^6 pure policies per player. Zero at equilibrium.
    """
    actions = ["p", "b"]
    # each player's own info sets: own card × decision histories
    infos = {1: ["", "pb"], 2: ["p", "b"]}

    def play(hist: str, c1: int, c2: int, brp: int, brs: dict[tuple[int, str], int]) -> float:
        if _is_terminal(hist):
            return _kuhn_util(hist, c1, c2, 1)
        cur = 1 if len(hist) % 2 == 0 else 2
        if cur == brp:
            card = c1 if brp == 1 else c2
            a = brs[(card, hist)]
            return play(hist + actions[a], c1, c2, brp, brs)
        card = c1 if cur == 1 else c2
        s = avg.get(f"p{cur}c{card}:{hist}", np.array([0.5, 0.5]))
        return float(sum(s[i] * play(hist + a, c1, c2, brp, brs) for i, a in enumerate(actions)))

    def best_response(brp: int) -> float:
        best = -1e9
        keys = [(card, h) for card in range(3) for h in infos[brp]]
        for bits in range(2 ** len(keys)):
            brs = {k: (bits >> i) & 1 for i, k in enumerate(keys)}
            tot = 0.0
            for c1, c2 in _DEALS:
                u1 = play("", c1, c2, brp, brs)
                tot += u1 if brp == 1 else -u1
            best = max(best, tot / len(_DEALS))
        return best

    return float(best_response(1) + best_response(2))


def bench_cfr(seed: int = 20261231, *, n_iter: int = 60_000) -> dict[str, float]:
    """SYNTHETIC: CFR on Kuhn poker — exploitability shrinks toward the
    known equilibrium value (P1 EV = −1/18). The lane default is 60k
    iterations; the research catalog passes a shorter run."""
    rng = np.random.default_rng(seed)
    avg = cfr_kuhn(n_iter, rng)
    out: dict[str, float] = {}
    out["synthetic_cfr_exploitability"] = kuhn_exploitability(avg)
    # canonical equilibrium signature: P1 with card 0 (weakest) checks
    # ~2/3 after check–bet sequence; check first action freq ≈ 2/3
    s = avg.get("p1c0:", np.array([0.5, 0.5]))
    out["synthetic_cfr_p1c0_check"] = float(s[0])
    out["synthetic_cfr_converged"] = float(out["synthetic_cfr_exploitability"] < 0.02)
    return out


if __name__ == "__main__":
    print(bench_cfr())
