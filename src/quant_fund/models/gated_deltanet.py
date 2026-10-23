"""Gated DeltaNet (Yang et al. 2024) — delta-rule memory with an (SYNTHETIC)
output gate: S = S W + beta (v - S^T k) ⊗ k, o = g ⊙ (q^T S). The gate
adds non-recall forgetting control. Induction recall bench.
"""

from __future__ import annotations

from quant_fund.models._lm_synth import VOCAB, attn_baseline, recall_batch


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("gated_deltanet requires torch (pip install -e .[nn])") from exc
    return torch


def bench_gated_deltanet(seed: int = 2269, iters: int = 800, D: int = 16) -> dict[str, float]:
    torch = _torch()
    x, y = recall_batch(seed)
    torch.manual_seed(seed)
    emb = torch.nn.Embedding(VOCAB, D)
    pw = torch.nn.Linear(D, D)
    pk = torch.nn.Linear(D, D)
    pv = torch.nn.Linear(D, D)
    pq = torch.nn.Linear(D, D)
    pb = torch.nn.Linear(D, D)
    pg = torch.nn.Linear(D, D)
    head = torch.nn.Linear(D, VOCAB)
    params = [p for m in (emb, pw, pk, pv, pq, pb, pg, head) for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=0.005)
    X = torch.tensor(x)
    Y = torch.tensor(y)

    def scan(h):
        B_, T_, _ = h.shape
        S = torch.zeros(B_, D, D)
        for t in range(T_):
            w = torch.sigmoid(pw(h[:, t]))
            k = torch.tanh(pk(h[:, t]))
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            v = pv(h[:, t])
            beta = torch.sigmoid(pb(h[:, t]))
            pred = torch.einsum("bi,bij->bj", k, S)
            delta = torch.einsum("bi,bj->bij", k, (v - pred))
            S = S * w[:, :, None] + beta[:, :, None] * delta
        q = torch.tanh(pq(h[:, -1]))
        g = torch.sigmoid(pg(h[:, -1]))
        return g * torch.einsum("bi,bij->bj", q, S)

    for _ in range(iters):
        loss = torch.nn.functional.cross_entropy(head(scan(emb(X))), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(scan(emb(X))).argmax(-1) == Y).float().mean().item()
    base = attn_baseline(seed)
    return {
        "synthetic_gdn_recall": float(acc),
        "synthetic_attn_recall": base,
        "synthetic_gdn_gain": float(acc) - base,
        "synthetic_torch_available": 1.0,
    }
