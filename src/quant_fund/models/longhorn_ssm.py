"""Longhorn (Liu et al. 2024) — SSM designed from online regression: (SYNTHETIC)
the state update solves an online least-squares objective
h_t = h_{t-1} + beta_t k_t (x_t - k_t^T h_{t-1}) — closed-form Kalman-like
gain; no random SSM init needed. Induction recall bench.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("longhorn_ssm requires torch (pip install -e .[nn])") from exc
    return torch


def bench_longhorn_ssm(seed: int = 2275, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pk = torch.nn.Linear(D, D)
    pb = torch.nn.Linear(D, D)
    pq = torch.nn.Linear(D, D)
    head = torch.nn.Linear(D, VOCAB)
    params = [p for m in (emb, pk, pb, pq, head) for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        s = torch.zeros(B_, D)
        for t in range(T_):
            k = torch.tanh(pk(h[:, t]))
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(pb(h[:, t]))
            pred = (s * k).sum(-1, keepdim=True)
            s = s + beta * k * (h[:, t] - pred)
        q = torch.tanh(pq(h[:, -1]))
        return s * q

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(head(scan(emb(X))), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_longhorn_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_longhorn_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
