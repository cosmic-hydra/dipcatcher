"""Titans neural memory (Behrouz et al. 2025) — memory is a model (SYNTHETIC)
trained online by its own surprise: M_t = eta_t M_{t-1} - theta_t * grad
on associative loss ||M^T k - v||². Momentum-style surprise writes.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("titans_memory requires torch (pip install -e .[nn])") from exc
    return torch


def bench_titans_memory(seed: int = 2263, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pk = torch.nn.Linear(D, D)
    pv = torch.nn.Linear(D, D)
    pq = torch.nn.Linear(D, D)
    pe = torch.nn.Linear(D, 1)  # retention eta
    pt = torch.nn.Linear(D, 1)  # lr theta
    head = torch.nn.Linear(D, VOCAB)
    params = (
        list(emb.parameters())
        + list(pk.parameters())
        + list(pv.parameters())
        + list(pq.parameters())
        + list(pe.parameters())
        + list(pt.parameters())
        + list(head.parameters())
    )
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        M = torch.zeros(B_, D, D)
        for t in range(T_):
            k = torch.tanh(pk(h[:, t]))
            v = pv(h[:, t])
            eta = torch.sigmoid(pe(h[:, t]))
            theta = torch.sigmoid(pt(h[:, t]))
            surprise = torch.einsum("bi,bij->bj", k, M) - v  # prediction error
            grad = torch.einsum("bi,bj->bij", k, surprise)
            M = eta[:, :, None] * M - theta[:, :, None] * grad
        q = torch.tanh(pq(h[:, -1]))
        return torch.einsum("bi,bij->bj", q, M)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(head(scan(emb(X))), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_titans_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_titans_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
